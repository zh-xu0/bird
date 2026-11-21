# BIRD — Bird Image Recognition Detector

A five-class bird image classifier and a reCAPTCHA-style game, built for the
EGCI486 Image Processing final project.

**Authors:** Zachary Xu and Jonathan Reppert  
**License:** GNU General Public License v3.0 or later (see [LICENSE](LICENSE))

The project has two parts:

1. **Model training** — handcrafted OpenCV features plus scikit-learn SVMs,
   evaluated with stratified 5-fold cross-validation.
2. **Presentation game** — a bird CAPTCHA with a best-effort SVM Hint button.

The five classes are bald eagle, bee-eater, bustard, European gallinule, and
sparrow.

## Results

| Model | 5-fold CV accuracy | Macro F1 |
|---|---:|---:|
| **SVM (RBF, balanced)** | **73.31% ± 1.57%** | **0.7050** |
| K-NN | 63.16% ± 1.10% | 0.6021 |

Best SVM settings: `C=10`, `gamma="scale"`.

The selected model is saved at
[`results/models/best_model.joblib`](results/models/best_model.joblib).
Metrics, the classification report, and figures are in
[`results/`](results/).

## Repository layout

```text
BIRD/
├── classification_model.py    # feature extraction, training, CV, prediction
├── requirements.txt
├── README.md
├── LICENSE
├── results/
│   ├── best_model.json
│   ├── cv_results.csv
│   ├── classification_report.txt
│   ├── feature_columns.json
│   ├── models/
│   │   └── best_model.joblib
│   └── figures/
├── game/
│   ├── CaptchaSparrow.py      # captcha grid generator
│   ├── CaptchSparrowApp.py    # Tkinter UI, 3 attempts, Hint button
│   ├── ModelSolver.py         # SVM inference for the Hint button
│   ├── run_game.sh            # launcher for Linux/macOS
│   ├── run_game.bat           # launcher for Windows
│   ├── demo_dataset/test/     # 10 demo images per species
│   ├── bird_samples/          # header reference photo per species
│   └── README.md
└── docs/
    └── SparrowDetector_ProjectProposal.pdf
```

`docs/SparrowDetector_ProjectProposal.pdf` is the original proposal, kept for
reference.

## Dataset

The full dataset is **not committed** because the raw archive is about 195 MB,
which exceeds GitHub's 100 MB file limit. Everything else — training code,
trained model, metrics, game, and a small demo dataset — is included.

The images come from [images.cv](https://images.cv):

- [bald eagle](https://images.cv/dataset/bald-eagle-image-classification-dataset)
- [bee-eater](https://images.cv/dataset/bee-eater-image-classification-dataset)
- [bustard](https://images.cv/dataset/bustard-image-classification-dataset)
- [European gallinule](https://images.cv/dataset/european-gallinule-image-classification-dataset)
- [sparrow](https://images.cv/search-labeled-image-dataset?query=sparrow)

The full training set has 7,340 images (from 5,645 unique source photos) and
the held-out test set has 310 images, all 256×256 JPEGs. To retrain, place the
images at:

```text
dataset/
├── train/
│   ├── bald_eagle/
│   ├── bee_eater/
│   ├── bustard/
│   ├── european_gallinule/
│   └── sparrow/
└── test/
    └── (same class folders)
```

For the full game, place `birds_dataset.zip` in `game/` and pass it with
`--dataset`; it is excluded from Git by `.gitignore`.

## Setup

Tested with Python 3.14 on Linux; the launchers and setup below also cover
Windows.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Tkinter is a system package, not a pip package.
# Arch/CachyOS:
sudo pacman -S tk
# Debian/Ubuntu: sudo apt install python3-tk
# macOS (Homebrew Python): brew install python-tk
```

Windows (PowerShell or Command Prompt):

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tkinter is bundled with the official python.org Windows installer (enable
**tcl/tk and IDLE**). Virtual environments are OS-specific, so create a fresh
`.venv` on each platform instead of copying one across.

## Train and evaluate

From the repository root, with `dataset/train/` populated:

```bash
python classification_model.py
```

This extracts 311 features per image, tunes the SVM and K-NN grids with
stratified 5-fold cross-validation, writes the metrics/figures to `results/`,
and saves `results/models/best_model.joblib` and
`results/models/best_model.json`.

Useful options:

```bash
python classification_model.py --recompute-features   # rebuild the feature cache
python classification_model.py --predict bird.jpg     # classify one image
python classification_model.py --n-jobs 2             # quieter / fewer workers
```

The feature table `results/features_train.csv` is regenerated as needed and is
ignored by Git.

## Run the CAPTCHA game

The game is the presentation/demo. It randomly asks for one of the five birds,
gives three Verify attempts per grid, and shows a reference photo in the blue
header. The **Hint** button runs the trained SVM and highlights predicted
tiles in green.

```bash
./game/run_game.sh
```

On Windows:

```bat
game\run_game.bat
```

Both launchers use the included demo dataset by default, so the game runs from
a fresh clone. See [game/README.md](game/README.md) for the full-dataset
options.

## Method summary

- **Features (311 per image):** RGB statistics, HSV statistics, global HSV
  histograms, 2×2 spatial HSV histograms, Lab statistics, and seven Hu moments.
- **Preprocessing:** `StandardScaler`.
- **Models:** SVM (RBF, balanced class weights) and K-NN.
- **Validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  on the training split only. The test split is never used for feature
  selection, tuning, or model selection.
- **Metric:** accuracy for grid search; macro F1 and per-class scores reported.

## Limitations

- The dataset includes `_aug` copies of some source photos. Plain stratified
  K-fold can place an original and its augmented copy in different folds, so
  the CV scores may be slightly optimistic. A grouped split by source image
  would give a more conservative estimate.
- The game's Hint is best-effort: overall CV accuracy is about 73%, and
  sparrow recall is only about 47%. The UI says so when it finds no target.
- The model uses CPU only. The full training run takes a few minutes with all
  available cores; the GPU is not needed for this feature set.

## License

Copyright (C) 2026 Zachary Xu and Jonathan Reppert.

This project is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See [LICENSE](LICENSE) for the full text.

Dataset images are sourced from images.cv and remain subject to their
respective terms.
