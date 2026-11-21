"""EfficientNet-B0 embedding + RBF SVM solver for the bird CAPTCHA game.

The game calls :func:`solve_with_model`, which returns the grid positions the
model predicts as the requested bird. Each tile is read from the original
dataset image referenced by the grid (zip entry or file path), resized to
224x224, normalized with ImageNet statistics, embedded by the frozen
EfficientNet-B0 backbone, and classified by the saved StandardScaler + SVC
pipeline.

Artifacts are produced by the project's ``classification_model.py``:
  results/models/best_model.joblib
  results/model_config.json

Set ``BIRD_DEVICE=cpu`` or ``BIRD_DEVICE=cuda`` to override device selection,
and ``BIRD_SOLVE_BATCH_SIZE`` to change the embedding batch size (default 32).
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

from PIL import Image

GAME_DIR = Path(__file__).resolve().parent
PROJECT_DIR = GAME_DIR.parent
MODEL_PATH = PROJECT_DIR / "results" / "models" / "best_model.joblib"
CONFIG_PATH = PROJECT_DIR / "results" / "model_config.json"
BATCH_SIZE = max(1, int(os.environ.get("BIRD_SOLVE_BATCH_SIZE", "32")))

_MODEL = None
_BACKBONE = None
_TRANSFORM = None
_DEVICE = None
_CONFIG: dict | None = None


def _torch_modules():
    try:
        import torch
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "The Solve button requires PyTorch and torchvision. Install them with "
            "`pip install -r requirements.txt` and try again."
        ) from exc
    return torch, models, transforms


def _load_solver():
    """Load the trained SVM, the frozen backbone and the preprocessing once."""
    global _MODEL, _BACKBONE, _TRANSFORM, _DEVICE, _CONFIG
    if _MODEL is not None:
        return _MODEL, _BACKBONE, _TRANSFORM, _DEVICE, _CONFIG

    if not MODEL_PATH.is_file() or not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifacts not found under {PROJECT_DIR / 'results'}. "
            "Run `python classification_model.py` in the project folder first."
        )

    import joblib

    torch, models, transforms = _torch_modules()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _CONFIG = config

    try:
        enum_name, member_name = config["weights"].split(".", 1)
        weights = getattr(getattr(models, enum_name), member_name)
    except (AttributeError, KeyError, ValueError) as exc:
        raise RuntimeError(
            "Unsupported model configuration in results/model_config.json. "
            "Retrain with classification_model.py."
        ) from exc

    factory = getattr(models, config["backbone"], None)
    if factory is None:
        raise RuntimeError(
            f"torchvision does not provide backbone '{config['backbone']}'."
        )

    try:
        backbone = factory(weights=weights)
    except Exception as exc:  # first-run download / cache problem
        raise RuntimeError(
            "Could not load the EfficientNet-B0 ImageNet weights. The first "
            "Solve downloads them with torchvision (about 20 MB); check your "
            "internet connection, or set TORCH_HOME to a cache that has them."
        ) from exc

    backbone.classifier = torch.nn.Identity()

    requested = os.environ.get("BIRD_DEVICE", "auto").lower()
    if requested == "cpu":
        device = torch.device("cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BIRD_DEVICE=cuda was set, but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = int(config.get("image_size", 224))
    mean = tuple(config.get("normalize_mean", (0.485, 0.456, 0.406)))
    std = tuple(config.get("normalize_std", (0.229, 0.224, 0.225)))
    transform = transforms.Compose(
        [
            transforms.Resize((size, size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    _MODEL = joblib.load(MODEL_PATH)
    _BACKBONE = backbone.eval().to(device)
    _TRANSFORM = transform
    _DEVICE = device
    return _MODEL, _BACKBONE, _TRANSFORM, _DEVICE, _CONFIG


def _open_image(zip_file, is_zip: bool, image_ref: str) -> Image.Image | None:
    """Load one original dataset image as RGB, from a zip entry or a path."""
    try:
        if is_zip:
            with zip_file.open(image_ref) as handle:
                data = handle.read()
            with Image.open(io.BytesIO(data)) as image:
                return image.convert("RGB")
        with Image.open(image_ref) as image:
            return image.convert("RGB")
    except (OSError, KeyError):
        return None


def _embed_batch(tensors):
    torch, _, _ = _torch_modules()
    batch = torch.stack(tensors).to(_DEVICE)
    with torch.inference_mode():
        if _DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = _BACKBONE(batch)
        else:
            features = _BACKBONE(batch)
    return features.float().cpu().numpy()


def solve_with_model(
    grid_result,
    target_bird: str | None = None,
) -> list[tuple[int, int]]:
    """Return tiles that the model predicts as the requested bird."""
    model, _backbone, transform, _device, _config = _load_solver()
    target = (
        target_bird or getattr(grid_result, "target_bird", "sparrow")
    ).lower()

    if grid_result.source:
        source = Path(grid_result.source)
        is_zip = bool(grid_result.is_zip)
    else:
        from CaptchaSparrow import _resolve_dataset_path

        source = _resolve_dataset_path()
        is_zip = source.is_file() and zipfile.is_zipfile(source)

    zip_file = zipfile.ZipFile(source, "r") if is_zip else None
    predicted: list[tuple[int, int]] = []
    tensors = []
    positions: list[tuple[int, int]] = []

    def flush() -> None:
        if not tensors:
            return
        embeddings = _embed_batch(tensors)
        labels = model.predict(embeddings)
        for position, label in zip(positions, labels):
            if str(label).lower() == target:
                predicted.append(position)
        tensors.clear()
        positions.clear()

    try:
        for row in range(grid_result.grid_size):
            for col in range(grid_result.grid_size):
                image = _open_image(
                    zip_file, is_zip, grid_result.image_paths[row][col]
                )
                if image is None:
                    continue
                tensors.append(transform(image))
                positions.append((row, col))
                if len(tensors) >= BATCH_SIZE:
                    flush()
        flush()
    finally:
        if zip_file is not None:
            zip_file.close()

    return predicted
