#!/usr/bin/env python3
"""Bird species classification: frozen EfficientNet-B0 embeddings + RBF SVM.

Pipeline:
  1. ImageNet-pretrained EfficientNet-B0 (torchvision) is used as a frozen
     feature extractor: the classifier head is removed and every image becomes
     a 1280-dimensional embedding.
  2. A StandardScaler + RBF SVM is trained on those embeddings, exactly like the
     previous handcrafted-feature baseline but with CNN features.

Evaluation:
  * Stratified 5-fold CV (the historical protocol, kept for comparison).
  * Grouped 5-fold CV (StratifiedGroupKFold): augmented copies of the same
    source photo always stay in the same fold. This is the primary, honest
    metric.
  * One final fit on the whole training split, evaluated once on the held-out
    test split.

Artifacts written to results/:
  models/best_model.joblib   trained StandardScaler + SVC pipeline
  model_config.json          backbone, preprocessing, classes, versions
  cv_results.csv             CV and test metrics
  classification_report.txt  per-class reports
  best_model.json            summary used by the README/report
  figures/                   class distribution, samples, confusion matrices
  embeddings_train.npy       cached embeddings (gitignored)
  embeddings_test.npy        cached embeddings (gitignored)

Usage:
  python classification_model.py
  python classification_model.py --recompute-embeddings
  python classification_model.py --predict path/to/bird.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5
CLASSES = [
    "bald_eagle",
    "bee_eater",
    "bustard",
    "european_gallinule",
    "sparrow",
]

BACKBONE_NAME = "efficientnet_b0"
WEIGHTS_NAME = "EfficientNet_B0_Weights.IMAGENET1K_V1"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBEDDING_VERSION = 1

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
TEST_DIR = PROJECT_DIR / "dataset" / "test"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_CONFIG_PATH = RESULTS_DIR / "model_config.json"
CV_RESULTS_CSV = RESULTS_DIR / "cv_results.csv"
CLASSIFICATION_REPORT_TXT = RESULTS_DIR / "classification_report.txt"
BEST_MODEL_JSON = RESULTS_DIR / "best_model.json"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def list_images(split: str) -> tuple[list[Path], list[str]]:
    """List images class by class in a deterministic order."""
    root = TRAIN_DIR if split == "train" else TEST_DIR
    paths: list[Path] = []
    labels: list[str] = []
    for class_name in CLASSES:
        for path in sorted((root / class_name).glob("*.jpg")):
            paths.append(path)
            labels.append(class_name)
    return paths, labels


def source_group(path: Path) -> str:
    """Group augmented copies with their source photo (`<id>_aug.jpg`)."""
    stem = path.stem
    if "_aug" in stem:
        stem = stem.split("_aug", 1)[0]
    return stem


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_backbone(device: torch.device) -> torch.nn.Module:
    try:
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception as exc:  # network / cache problem on first run
        raise SystemExit(
            "Could not load the EfficientNet-B0 ImageNet weights. The first run "
            "downloads them with torchvision (about 20 MB); check your internet "
            "connection or set TORCH_HOME to a folder that already has them."
        ) from exc
    backbone.classifier = torch.nn.Identity()
    return backbone.eval().to(device)


@torch.inference_mode()
def embed_images(
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
    description: str,
) -> np.ndarray:
    backbone = build_backbone(device)
    loader = DataLoader(
        ImagePathDataset(paths, build_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    for images in tqdm(loader, desc=description, unit="batch"):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = backbone(images)
        else:
            features = backbone(images)
        chunks.append(features.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def cache_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha1()
    digest.update(f"{EMBEDDING_VERSION}|{BACKBONE_NAME}|{WEIGHTS_NAME}|{IMAGE_SIZE}".encode())
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_extract_embeddings(
    split: str,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
    force: bool,
) -> np.ndarray:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = RESULTS_DIR / f"embeddings_{split}.npy"
    meta_path = RESULTS_DIR / f"embeddings_{split}.meta.json"
    fingerprint = cache_fingerprint(paths)

    if not force and npy_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("fingerprint") == fingerprint
            and meta.get("shape", [None])[0] == len(paths)
        ):
            print(f"Loaded cached {split} embeddings: {npy_path}")
            return np.load(npy_path)

    print(
        f"Extracting {split} embeddings for {len(paths)} images with "
        f"{BACKBONE_NAME} on {device} ..."
    )
    embeddings = embed_images(
        paths, device, batch_size, workers, f"{split} embeddings"
    )
    np.save(npy_path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "shape": list(embeddings.shape),
                "backbone": BACKBONE_NAME,
                "weights": WEIGHTS_NAME,
                "image_size": IMAGE_SIZE,
                "embedding_version": EMBEDDING_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {npy_path} {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Classifier and evaluation
# ---------------------------------------------------------------------------


def make_pipeline() -> Pipeline:
    """StandardScaler + RBF SVM, matching the previous baseline classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    cache_size=1000,
                ),
            ),
        ]
    )


def summarize(y_true, y_pred, name: str) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print(f"\n=== {name} ===")
    print(
        f"accuracy {accuracy_score(y_true, y_pred) * 100:.2f}% | "
        f"macro F1 {f1_score(y_true, y_pred, average='macro'):.4f}"
    )
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            target_names=CLASSES,
            digits=4,
            zero_division=0,
        )
    )
    print("confusion matrix (rows=true, cols=pred):")
    print(matrix)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in CLASSES
        },
        "confusion_matrix": matrix.tolist(),
    }


def cross_validate(X, y, splits, tag: str) -> tuple[np.ndarray, dict]:
    oof = np.empty(len(y), dtype=object)
    fold_scores: list[float] = []
    started = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        model = make_pipeline()
        fit_started = time.perf_counter()
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[val_idx])
        oof[val_idx] = predictions
        score = accuracy_score(y[val_idx], predictions)
        fold_scores.append(float(score))
        print(
            f"[{tag}] fold {fold}/{len(splits)}: accuracy {score * 100:.2f}% "
            f"({time.perf_counter() - fit_started:.1f}s)"
        )
    metrics = summarize(y, oof, tag)
    metrics["fold_accuracies"] = fold_scores
    metrics["fold_accuracy_mean"] = float(np.mean(fold_scores))
    metrics["fold_accuracy_std"] = float(np.std(fold_scores))
    metrics["fit_seconds"] = time.perf_counter() - started
    print(
        f"[{tag}] CV accuracy {metrics['fold_accuracy_mean'] * 100:.2f}% "
        f"+- {metrics['fold_accuracy_std'] * 100:.2f}% "
        f"| total {metrics['fit_seconds']:.1f}s"
    )
    return oof, metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_class_distribution(labels: list[str]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = [labels.count(name) for name in CLASSES]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASSES, counts, color="#1A73E8")
    ax.bar_label(bars, padding=2)
    ax.set_title("Training images per species")
    ax.set_ylabel("images")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def save_sample_images(paths: list[Path], labels: list[str], per_class: int = 3) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(3 * per_class, 2.6 * len(CLASSES)))
    for row, class_name in enumerate(CLASSES):
        class_paths = [p for p, label in zip(paths, labels) if label == class_name][
            :per_class
        ]
        for col in range(per_class):
            axis = axes[row, col]
            axis.axis("off")
            if col < len(class_paths):
                with Image.open(class_paths[col]) as image:
                    axis.imshow(image.convert("RGB"))
                axis.set_title(class_name, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_images.png", dpi=120)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_image(image_path: str, device: torch.device) -> None:
    if not MODEL_PATH.is_file() or not MODEL_CONFIG_PATH.is_file():
        raise SystemExit(
            f"Model artifacts not found in {RESULTS_DIR}. "
            "Run `python classification_model.py` first."
        )
    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    pipeline = joblib.load(MODEL_PATH)
    embeddings = embed_images([Path(image_path)], device, 1, 0, "embedding")
    prediction = pipeline.predict(embeddings)[0]
    scores = pipeline.decision_function(embeddings)
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    if scores is not None:
        ordered = np.argsort(np.ravel(scores))[::-1]
        print("Decision scores:")
        for index in ordered:
            print(f"  {pipeline.classes_[index]:<22} {np.ravel(scores)[index]: .4f}")
    print(f"(model: {config['backbone']}, {config['weights']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate frozen EfficientNet-B0 embeddings + RBF SVM."
    )
    parser.add_argument(
        "--predict",
        metavar="IMAGE",
        help="Classify one image with the saved model and exit.",
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Recompute cached CNN embeddings even if they exist.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for the frozen backbone (default: auto).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Embedding batch size."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers for embedding extraction (0 disables).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.predict:
        predict_image(args.predict, device)
        return

    train_paths, y_train = list_images("train")
    test_paths, y_test = list_images("test")
    if not train_paths:
        raise SystemExit(f"No training images found under {TRAIN_DIR}")
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    print(
        f"Training images: {len(train_paths)} | test images: {len(test_paths)} "
        f"| source groups: {len({source_group(p) for p in train_paths})}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train = load_or_extract_embeddings(
        "train",
        train_paths,
        device,
        args.batch_size,
        args.workers,
        args.recompute_embeddings,
    )
    X_test = load_or_extract_embeddings(
        "test",
        test_paths,
        device,
        args.batch_size,
        args.workers,
        args.recompute_embeddings,
    )
    print(f"Embedding matrix: train {X_train.shape} | test {X_test.shape}")

    groups = np.asarray([source_group(path) for path in train_paths])
    stratified_splits = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        ).split(X_train, y_train)
    )
    grouped_splits = list(
        StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        ).split(X_train, y_train, groups)
    )

    print("\n--- Stratified 5-fold CV (historical protocol) ---")
    oof_stratified, stratified_metrics = cross_validate(
        X_train, y_train, stratified_splits, "Stratified 5-fold"
    )
    print("\n--- Grouped 5-fold CV (primary, no augmentation leakage) ---")
    oof_grouped, grouped_metrics = cross_validate(
        X_train, y_train, grouped_splits, "Grouped 5-fold"
    )

    save_class_distribution(y_train.tolist())
    save_sample_images(train_paths, y_train.tolist())
    save_confusion_matrix(
        y_train,
        oof_stratified,
        "Stratified 5-fold OOF confusion matrix",
        "confusion_matrix_stratified.png",
    )
    save_confusion_matrix(
        y_train,
        oof_grouped,
        "Grouped 5-fold OOF confusion matrix (no _aug leakage)",
        "confusion_matrix_grouped.png",
    )

    print("\nFitting final model on the whole training split ...")
    final_model = make_pipeline()
    fit_started = time.perf_counter()
    final_model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started
    test_predictions = final_model.predict(X_test)
    test_metrics = summarize(y_test, test_predictions, "Held-out test split")
    print(f"Final fit: {fit_seconds:.1f}s on {len(X_train)} images")

    joblib.dump(final_model, MODEL_PATH, compress=3)
    print(f"Saved model: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    config = {
        "model_type": "frozen CNN embeddings + RBF SVM",
        "backbone": BACKBONE_NAME,
        "weights": WEIGHTS_NAME,
        "image_size": IMAGE_SIZE,
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
        "embedding_dim": int(X_train.shape[1]),
        "embedding_version": EMBEDDING_VERSION,
        "classes": CLASSES,
        "classifier": "StandardScaler + SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    MODEL_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = [
        {
            "model": "EfficientNet-B0 frozen embeddings + RBF SVM",
            "protocol": "Stratified 5-fold CV",
            "cv_accuracy_mean": stratified_metrics["fold_accuracy_mean"],
            "cv_accuracy_std": stratified_metrics["fold_accuracy_std"],
            "oof_accuracy": stratified_metrics["accuracy"],
            "oof_macro_f1": stratified_metrics["macro_f1"],
            "fit_seconds": stratified_metrics["fit_seconds"],
        },
        {
            "model": "EfficientNet-B0 frozen embeddings + RBF SVM",
            "protocol": "Grouped 5-fold CV (primary)",
            "cv_accuracy_mean": grouped_metrics["fold_accuracy_mean"],
            "cv_accuracy_std": grouped_metrics["fold_accuracy_std"],
            "oof_accuracy": grouped_metrics["accuracy"],
            "oof_macro_f1": grouped_metrics["macro_f1"],
            "fit_seconds": grouped_metrics["fit_seconds"],
        },
        {
            "model": "EfficientNet-B0 frozen embeddings + RBF SVM",
            "protocol": "Held-out test split (final fit)",
            "cv_accuracy_mean": "",
            "cv_accuracy_std": "",
            "oof_accuracy": test_metrics["accuracy"],
            "oof_macro_f1": test_metrics["macro_f1"],
            "fit_seconds": fit_seconds,
        },
    ]
    pd.DataFrame(rows).to_csv(CV_RESULTS_CSV, index=False)

    report_sections = []
    for name, y_true, y_pred, metrics in (
        (
            "Stratified 5-fold CV (historical protocol)",
            y_train,
            oof_stratified,
            stratified_metrics,
        ),
        (
            "Grouped 5-fold CV (primary, no _aug leakage)",
            y_train,
            oof_grouped,
            grouped_metrics,
        ),
        ("Held-out test split", y_test, test_predictions, test_metrics),
    ):
        report_sections.append(
            f"{name}\n"
            f"Accuracy: {metrics['accuracy'] * 100:.2f}%\n"
            f"Macro F1: {metrics['macro_f1']:.4f}\n\n"
            + classification_report(
                y_true,
                y_pred,
                labels=CLASSES,
                target_names=CLASSES,
                digits=4,
                zero_division=0,
            )
        )
    CLASSIFICATION_REPORT_TXT.write_text(
        "\n".join(report_sections), encoding="utf-8"
    )

    summary = {
        "model": "EfficientNet-B0 (frozen, ImageNet-1K) + RBF SVM",
        "backbone": BACKBONE_NAME,
        "weights": WEIGHTS_NAME,
        "cv_protocol": "StratifiedGroupKFold(5), grouped by source photo",
        "cv_accuracy_mean": grouped_metrics["fold_accuracy_mean"],
        "cv_accuracy_std": grouped_metrics["fold_accuracy_std"],
        "oof_accuracy": grouped_metrics["accuracy"],
        "oof_macro_f1": grouped_metrics["macro_f1"],
        "stratified_cv_accuracy_mean": stratified_metrics["fold_accuracy_mean"],
        "stratified_cv_accuracy_std": stratified_metrics["fold_accuracy_std"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "sparrow_recall_grouped_cv": grouped_metrics["per_class"]["sparrow"]["recall"],
        "train_images": len(train_paths),
        "test_images": len(test_paths),
        "source_groups": len(set(groups)),
        "trained_at": config["trained_at"],
    }
    BEST_MODEL_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        f"Grouped 5-fold CV: {grouped_metrics['fold_accuracy_mean'] * 100:.2f}% "
        f"+- {grouped_metrics['fold_accuracy_std'] * 100:.2f}% | "
        f"macro F1 {grouped_metrics['macro_f1']:.4f}"
    )
    print(
        f"Stratified 5-fold CV: {stratified_metrics['fold_accuracy_mean'] * 100:.2f}% "
        f"+- {stratified_metrics['fold_accuracy_std'] * 100:.2f}% | "
        f"macro F1 {stratified_metrics['macro_f1']:.4f}"
    )
    print(
        f"Held-out test: {test_metrics['accuracy'] * 100:.2f}% | "
        f"macro F1 {test_metrics['macro_f1']:.4f}"
    )
    print(f"Model: {MODEL_PATH}")
    print(f"Config: {MODEL_CONFIG_PATH}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
