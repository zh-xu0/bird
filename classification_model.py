#!/usr/bin/env python3
"""BIRD - Bird Image Recognition Detector.

EGCI486 Image Processing final project.
Copyright (C) 2026 Zachary Xu and Jonathan Reppert.
Licensed under the GNU General Public License v3.0 or later; see LICENSE.

Simple, beginner-friendly pipeline in the style of the EGCI486 labs:

  1. Extract 311 simple colour and shape features from every training image:
     the original 87 features plus 2x2 spatial HSV histograms.
  2. Compare StandardScaler + SVM (RBF, balanced class weights) with
     StandardScaler + K-NN.
  3. Tune a wider SVM parameter grid with 5-fold stratified cross-validation.
  4. Report CV accuracy, macro F1 and a confusion matrix.
  5. Save the best model and classify one image with --predict.

The test folder (dataset/test) is never read by this script.

Usage:
    python classification_model.py                     # train + evaluate
    python classification_model.py --predict bird.jpg  # classify one image
    python classification_model.py --recompute-features  # ignore cached CSV
"""

import os

# Keep the CPU quiet. This feature set is small, so there is no benefit in
# letting BLAS or OpenCV use every core.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import joblib
import matplotlib
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

cv2.setNumThreads(1)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5
N_JOBS = -1  # use every logical CPU core (this machine has 4 cores / 8 threads)

CLASSES = [
    "bald_eagle",
    "bee_eater",
    "bustard",
    "european_gallinule",
    "sparrow",
]

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
FEATURES_CSV = RESULTS_DIR / "features_train.csv"
FEATURE_COLUMNS_JSON = RESULTS_DIR / "feature_columns.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# Bump this if you change extract_features(), so old CSV files are rebuilt.
FEATURE_VERSION = 2


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features_from_array(image_bgr):
    """Return the 311-feature dictionary for one BGR image array."""
    image = cv2.resize(image_bgr, (256, 256))  # dataset images are 256 x 256

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = {}

    # RGB statistics (same idea as LAB08).
    for i, name in enumerate("RGB"):
        channel = rgb[:, :, i]
        features[f"{name}_mean"] = float(channel.mean())
        features[f"{name}_std"] = float(channel.std())
        features[f"{name}_min"] = float(channel.min())
        features[f"{name}_max"] = float(channel.max())

    # HSV statistics.
    for i, name in enumerate("HSV"):
        channel = hsv[:, :, i]
        features[f"HSV_{name}_mean"] = float(channel.mean())
        features[f"HSV_{name}_std"] = float(channel.std())

    # HSV histograms, normalised so every image is on the same scale.
    for channel_index, name, bins, value_range in (
        (0, "H", 24, (0, 180)),
        (1, "S", 16, (0, 256)),
        (2, "V", 16, (0, 256)),
    ):
        hist, _ = np.histogram(
            hsv[:, :, channel_index], bins=bins, range=value_range
        )
        hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
        for bin_index, value in enumerate(hist):
            features[f"hist_{name}_{bin_index}"] = float(value)

    # 2x2 spatial HSV histograms: where the colours are, not just the averages.
    height, width = hsv.shape[:2]
    for row in range(2):
        y0 = row * height // 2
        y1 = (row + 1) * height // 2
        for col in range(2):
            x0 = col * width // 2
            x1 = (col + 1) * width // 2
            cell = hsv[y0:y1, x0:x1]
            for channel_index, name, bins, value_range in (
                (0, "H", 24, (0, 180)),
                (1, "S", 16, (0, 256)),
                (2, "V", 16, (0, 256)),
            ):
                hist, _ = np.histogram(
                    cell[:, :, channel_index], bins=bins, range=value_range
                )
                hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
                for bin_index, value in enumerate(hist):
                    features[f"hist_{name}_r{row}c{col}_{bin_index}"] = float(value)

    # Lab colour statistics.
    for i, name in enumerate(("L", "a", "b")):
        channel = lab[:, :, i]
        features[f"Lab_{name}_mean"] = float(channel.mean())
        features[f"Lab_{name}_std"] = float(channel.std())

    # Hu moments (same log transform as LAB08).
    hu = cv2.HuMoments(cv2.moments(cv2.bitwise_not(gray))).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    for i, value in enumerate(hu):
        features[f"hu{i + 1}"] = float(value)

    return features


def extract_features(image_path):
    """Read an image file and return its 311-feature dictionary."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return extract_features_from_array(image)


def get_train_images():
    """List every training image, class by class. The test folder is not used."""
    images = []
    for class_name in CLASSES:
        images.extend(
            sorted(glob.glob(str(TRAIN_DIR / class_name / "*.jpg")))
        )
    return images


def _extract_one(image_path):
    # Avoid OpenCV spawning extra threads inside each worker.
    cv2.setNumThreads(1)
    return extract_features(image_path)


def build_feature_table(force=False, n_jobs=N_JOBS):
    """Extract features once and cache them in a CSV file (LAB08 style)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if FEATURES_CSV.exists() and not force:
        table = pd.read_csv(FEATURES_CSV)
        if (
            "feature_version" in table.columns
            and (table["feature_version"] == FEATURE_VERSION).all()
        ):
            n_features = table.shape[1] - 3
            print(
                f"Loaded cached features: {FEATURES_CSV} "
                f"({len(table)} images, {n_features} features)"
            )
            return table
        print("Cached feature file is outdated; extracting again.")

    image_paths = get_train_images()
    if not image_paths:
        raise FileNotFoundError(f"No training images found in {TRAIN_DIR}")

    print(f"Extracting features from {len(image_paths)} training images ...")
    started = time.time()
    if n_jobs == 1:
        feature_rows = [
            extract_features(image_path)
            for image_path in tqdm(image_paths, desc="Features")
        ]
    else:
        feature_rows = Parallel(n_jobs=n_jobs, verbose=5, batch_size=16)(
            delayed(_extract_one)(image_path) for image_path in image_paths
        )

    rows = []
    for image_path, features in zip(image_paths, feature_rows):
        features["path"] = image_path
        features["class"] = Path(image_path).parent.name
        features["feature_version"] = FEATURE_VERSION
        rows.append(features)

    table = pd.DataFrame(rows)
    table.to_csv(FEATURES_CSV, index=False)
    print(
        f"Saved {FEATURES_CSV} "
        f"({table.shape[0]} images, {table.shape[1] - 3} features) "
        f"in {time.time() - started:.1f}s"
    )
    return table


def get_feature_columns(table):
    """All numeric feature columns, i.e. everything except the metadata."""
    return [
        column
        for column in table.columns
        if column not in ("path", "class", "feature_version")
    ]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_class_distribution(table):
    counts = table["class"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="#2a78d6")
    ax.set_title("Training images per class")
    ax.set_xlabel("")
    ax.set_ylabel("Number of images")
    ax.bar_label(ax.containers[0], padding=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def save_sample_images(per_class=3):
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(7, 9))
    for row, class_name in enumerate(CLASSES):
        image_paths = sorted((TRAIN_DIR / class_name).glob("*.jpg"))[:per_class]
        for col in range(per_class):
            axes[row, col].axis("off")
            if col < len(image_paths):
                image = cv2.imread(str(image_paths[col]))
                axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_ylabel(
            class_name, rotation=0, ha="right", va="center", fontsize=8
        )
    fig.suptitle("Example training images (three per class)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_images.png", dpi=150)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, title, filename):
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    display.plot(cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_model_comparison(results):
    labels = [result["model"] for result in results]
    means = np.array([result["cv_accuracy_mean"] for result in results]) * 100
    stds = np.array([result["cv_accuracy_std"] for result in results]) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means, yerr=stds, capsize=5, color=["#2a78d6", "#eb6834"])
    ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model comparison")
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 1.5, f"{mean:.1f}%", ha="center")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def build_models():
    """Return (label, short name, pipeline, parameter grid) for each model."""
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    svm_grid = {
        "clf__C": [0.1, 1, 10, 100, 1000],
        "clf__gamma": ["scale", 0.01, 0.001],
    }

    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )
    knn_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
    }

    return [
        ("SVM (RBF)", "svm_rbf", svm, svm_grid),
        ("K-NN", "knn", knn, knn_grid),
    ]


def train_and_evaluate(table, n_jobs=N_JOBS):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = get_feature_columns(table)
    FEATURE_COLUMNS_JSON.write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )

    X = table[feature_columns].to_numpy(dtype=np.float32)
    y = table["class"].to_numpy()
    print(f"\nTraining matrix: {X.shape[0]} images x {X.shape[1]} features")

    save_class_distribution(table)
    save_sample_images()

    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    results = []
    fitted_models = {}
    report_sections = []

    for label, short_name, pipeline, param_grid in build_models():
        print(f"\n=== {label} ===")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            refit=True,
            verbose=1,
        )
        started = time.time()
        search.fit(X, y)
        elapsed = time.time() - started

        best_index = search.best_index_
        fold_scores = np.array(
            [
                search.cv_results_[f"split{fold}_test_score"][best_index]
                for fold in range(N_SPLITS)
            ]
        )
        print(f"Best parameters: {search.best_params_}")
        print(
            f"5-fold CV accuracy: {fold_scores.mean() * 100:.2f}% "
            f"+- {fold_scores.std() * 100:.2f}%"
        )

        # Out-of-fold predictions give an honest confusion matrix without
        # touching the test folder.
        y_pred = cross_val_predict(
            clone(search.best_estimator_), X, y, cv=cv, n_jobs=n_jobs
        )
        oof_accuracy = accuracy_score(y, y_pred)
        oof_macro_f1 = f1_score(y, y_pred, average="macro")
        report = classification_report(
            y, y_pred, labels=CLASSES, target_names=CLASSES, digits=4
        )
        print(
            f"Out-of-fold accuracy: {oof_accuracy * 100:.2f}% | "
            f"macro F1: {oof_macro_f1:.4f}\n"
        )
        print(report)

        fitted_models[short_name] = search.best_estimator_
        results.append(
            {
                "model": label,
                "key": short_name,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "cv_accuracy_mean": float(fold_scores.mean()),
                "cv_accuracy_std": float(fold_scores.std()),
                "oof_accuracy": float(oof_accuracy),
                "oof_macro_f1": float(oof_macro_f1),
                "fit_seconds": float(elapsed),
            }
        )
        report_sections.append(
            f"{label}\n"
            f"Best parameters: {search.best_params_}\n"
            f"5-fold CV accuracy: {fold_scores.mean() * 100:.2f}% "
            f"+- {fold_scores.std() * 100:.2f}%\n"
            f"Out-of-fold accuracy: {oof_accuracy * 100:.2f}%\n"
            f"Macro F1: {oof_macro_f1:.4f}\n\n"
            f"{report}\n"
        )

        save_confusion_matrix(
            y,
            y_pred,
            f"{label} - out-of-fold confusion matrix",
            f"confusion_matrix_{short_name}.png",
        )
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{short_name}.joblib")

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    save_model_comparison(results)
    (RESULTS_DIR / "classification_report.txt").write_text(
        "\n".join(report_sections), encoding="utf-8"
    )

    best = max(results, key=lambda row: row["cv_accuracy_mean"])
    joblib.dump(fitted_models[best["key"]], BEST_MODEL_PATH)
    (RESULTS_DIR / "best_model.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for result in results:
        print(
            f"{result['model']:<10} "
            f"CV accuracy {result['cv_accuracy_mean'] * 100:6.2f}% "
            f"+- {result['cv_accuracy_std'] * 100:.2f}% | "
            f"macro F1 {result['oof_macro_f1']:.4f}"
        )
    print(f"\nBest model: {best['model']}")
    print(f"Saved to:   {BEST_MODEL_PATH}")
    print(f"Results:    {RESULTS_DIR}")
    print("\nThe test folder was not used.")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(image_path):
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python classification_model.py` first."
        )

    model = joblib.load(BEST_MODEL_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    features = extract_features(image_path)

    # Keep the trained column order, then use NumPy exactly as in training.
    frame = pd.DataFrame([features])[feature_columns]
    row = frame.to_numpy(dtype=np.float32)

    predicted_class = model.predict(row)[0]

    print(f"\nImage: {image_path}")
    print(f"Predicted class: {predicted_class}")

    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(row))[0]
        score_type = "probability"
    elif hasattr(model, "decision_function"):
        scores = np.atleast_1d(np.asarray(model.decision_function(row))[0])
        score_type = "decision score (not a probability)"
    else:
        return predicted_class

    print(f"Scores ({score_type}):")
    ranked = sorted(
        zip(model.classes_, scores), key=lambda item: item[1], reverse=True
    )
    for class_name, score in ranked:
        print(f"  {class_name:<22} {float(score):>10.4f}")
    return predicted_class


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a simple bird species classifier."
    )
    parser.add_argument(
        "--predict",
        metavar="IMAGE",
        help="Classify one image using the saved model.",
    )
    parser.add_argument(
        "--recompute-features",
        action="store_true",
        help="Ignore the cached feature CSV and extract features again.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help=f"Parallel workers for grid search (default {N_JOBS}).",
    )
    args = parser.parse_args()

    if args.predict:
        predict(args.predict)
        return

    table = build_feature_table(
        force=args.recompute_features, n_jobs=args.n_jobs
    )
    train_and_evaluate(table, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
