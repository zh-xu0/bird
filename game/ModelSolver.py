"""SVM hint solver for the bird CAPTCHA game.

This module is deliberately free of tkinter/PIL UI imports so it can be tested
headlessly.  It loads the trained scikit-learn pipeline from the project's
`results/models/best_model.joblib` and classifies the original source image of
each captcha tile (zip entry or normal file path).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np

from CaptchaSparrow import CaptchaGridResult

GAME_DIR = Path(__file__).resolve().parent
PROJECT_DIR = GAME_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

MODEL_PATH = PROJECT_DIR / "results" / "models" / "best_model.joblib"
FEATURE_COLUMNS_PATH = PROJECT_DIR / "results" / "feature_columns.json"

_MODEL = None
_FEATURE_COLUMNS: list[str] | None = None
_EXTRACT_FEATURES = None


def _load_model():
    """Load the trained SVM and the feature extractor once."""
    global _MODEL, _FEATURE_COLUMNS, _EXTRACT_FEATURES
    if _MODEL is not None:
        return _MODEL, _FEATURE_COLUMNS, _EXTRACT_FEATURES

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. Run "
            "`python classification_model.py` in the project folder first."
        )

    try:
        from classification_model import extract_features_from_array
    except ImportError as exc:
        raise RuntimeError(
            "The SVM requires OpenCV, scikit-learn and joblib. Install "
            "requirements.txt and run the game with ./run_game.sh."
        ) from exc

    _MODEL = joblib.load(MODEL_PATH)
    _FEATURE_COLUMNS = json.loads(
        FEATURE_COLUMNS_PATH.read_text(encoding="utf-8")
    )
    _EXTRACT_FEATURES = extract_features_from_array
    return _MODEL, _FEATURE_COLUMNS, _EXTRACT_FEATURES


def _read_source_image(zip_file, is_zip: bool, image_ref: str):
    """Load one original dataset image, from a zip entry or a normal path."""
    if is_zip:
        with zip_file.open(image_ref) as handle:
            data = np.frombuffer(handle.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.imread(str(image_ref))


def solve_with_model(
    grid_result: CaptchaGridResult,
    target_bird: str | None = None,
) -> list[tuple[int, int]]:
    """Return tiles that the trained SVM predicts as the target bird."""
    model, feature_columns, extract_features = _load_model()
    target = (
        target_bird or getattr(grid_result, "target_bird", "sparrow")
    ).lower()

    if grid_result.source:
        source = Path(grid_result.source)
        is_zip = grid_result.is_zip
    else:
        from CaptchaSparrow import _resolve_dataset_path

        source = _resolve_dataset_path()
        is_zip = source.is_file() and zipfile.is_zipfile(source)

    zip_file = zipfile.ZipFile(source, "r") if is_zip else None
    positions: list[tuple[int, int]] = []
    vectors: list[list[float]] = []

    try:
        for row in range(grid_result.grid_size):
            for col in range(grid_result.grid_size):
                image = _read_source_image(
                    zip_file, is_zip, grid_result.image_paths[row][col]
                )
                if image is None:
                    continue
                features = extract_features(image)
                vectors.append([features[name] for name in feature_columns])
                positions.append((row, col))
    finally:
        if zip_file is not None:
            zip_file.close()

    if not vectors:
        return []

    X = np.asarray(vectors, dtype=np.float32)
    predictions = model.predict(X)
    return [
        position
        for position, label in zip(positions, predictions)
        if str(label).lower() == target
    ]
