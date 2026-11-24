# BIRD (Bird Image Recognition Detector)

A five-class bird image classifier and a reCAPTCHA-style game, built for the
EGCI486 Image Processing final project.

**Authors:** Zachary Xu and Jonathan Reppert  
**License:** GNU General Public License v3.0 or later (see [LICENSE](LICENSE))

The project has two parts:

1. **Model training** — a frozen ImageNet-pretrained EfficientNet-B0 feature
   extractor, with a scikit-learn RBF SVM trained on its 1,280-dimensional
   embeddings, evaluated with grouped and stratified 5-fold cross-validation.
2. **Presentation game** — a bird CAPTCHA with a model-backed Solve button.

The five classes are bald eagle, bee-eater, bustard, European gallinule, and
sparrow.

## Results

| Pipeline | Protocol | Accuracy | Macro F1 |
|---|---|---:|---:|
| **EfficientNet-B0 embeddings + RBF SVM** | **Grouped 5-fold CV (primary)** | **98.24% ± 0.13%** | **0.9801** |
| EfficientNet-B0 embeddings + RBF SVM | Stratified 5-fold CV | 98.90% ± 0.41% | 0.9876 |
| EfficientNet-B0 embeddings + RBF SVM | Held-out test (310 images) | 99.68% | 0.9955 |
| Previous 311-feature SVM baseline | Grouped 5-fold CV | 64.05% ± 0.82% | 0.5927 |

Grouped CV keeps every `_aug` copy of a source photo in the same fold, so it
is the honest estimate; the old 73.31% stratified result was inflated by
augmentation leakage. An end-to-end fine-tune of EfficientNet-B0 was also
evaluated and matched the frozen pipeline (98.91% grouped validation, 99.68%
test), so the project keeps the simpler frozen-embedding model.

The selected model is saved at
[`results/models/best_model.joblib`](results/models/best_model.joblib), with its
preprocessing contract in
[`results/model_config.json`](results/model_config.json). Metrics, the
classification report, and figures are in [`results/`](results/).

## Repository layout

```text
BIRD/
├── classification_model.py    # CNN embeddings, SVM training, CV, prediction
├── demo.mp4                   # a demo of our project    
├── requirements.txt
├── README.md
├── CREDITS.md                 # model/data/software attribution and licenses
├── LICENSE
├── results/
│   ├── best_model.json
│   ├── model_config.json      # backbone, preprocessing, classes
│   ├── cv_results.csv
│   ├── classification_report.txt
│   ├── models/
│   │   └── best_model.joblib
│   └── figures/
├── game/
│   ├── CaptchaSparrow.py      # captcha grid generator
│   ├── CaptchSparrowApp.py    # Tkinter UI, 3 attempts, Solve button
│   ├── ModelSolver.py         # EfficientNet-B0 + SVM inference for Solve
│   ├── run_game.sh            # launcher for Linux/macOS
│   ├── run_game.bat           # launcher for Windows
│   ├── demo_dataset/test/     # 10 demo images per species
│   ├── bird_samples/          # header reference photo per species
│   └── README.md
└── docs/
    └── project_proposal.pdf
    └── project_report.pdf
    └── slides.pdf
    
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

Tested with Python 3.14 on Linux; the launchers and setup also cover Windows
and macOS. The pinned PyTorch wheels support Python 3.11–3.15.

### Option A — virtual environment (recommended, default)

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate          # optional: the launchers find .venv automatically
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
.venv\Scripts\activate             # optional: the launchers find .venv automatically
pip install -r requirements.txt
```

Virtual environments are OS-specific, so create a fresh `.venv` on each
platform instead of copying one across.

### Option B — no virtual environment

If the Python you use already has the packages (system Python, conda, or
another environment), install into that interpreter and run the launcher. The
scripts prefer `.venv` when it exists; on Linux/macOS you can bypass it with
`BIRD_PYTHON`.

Linux/macOS:

```bash
python3 -m pip install -r requirements.txt
./game/run_game.sh                        # uses .venv if present, else python3
BIRD_PYTHON=python3 ./game/run_game.sh    # force the system interpreter
```

Windows:

```bat
py -3 -m pip install -r requirements.txt
game\run_game.bat
```

The Windows launcher prefers a portable `Python\python.exe` next to the
project, then `.venv`, then `python` from PATH.

Tkinter must still be available: it is bundled with the python.org Windows
installer (enable **tcl/tk and IDLE**) and with python.org macOS installers;
on Linux install the system package shown above. If pip reports a permissions
error, add `--user`. On distributions that mark the system Python as externally
managed (PEP 668, e.g. Arch/CachyOS), pip may refuse Option B — use Option A
or your distro's packages instead.

`requirements.txt` includes PyTorch and torchvision. The first training run
or first Solve downloads the EfficientNet-B0 ImageNet weights with torchvision
(about 20 MB). An NVIDIA GPU is optional; the model runs on CPU too.

## Train and evaluate

From the repository root, with `dataset/train/` populated:

```bash
python classification_model.py
```

This extracts 1,280-dimensional EfficientNet-B0 embeddings (cached in
`results/embeddings_*.npy`, which Git ignores), runs stratified and grouped
5-fold cross-validation, fits the SVM on the full training split, evaluates it
once on the held-out test split, and writes the metrics and figures to
`results/`. It saves `results/models/best_model.joblib` and
`results/model_config.json`.

Useful options:

```bash
python classification_model.py --recompute-embeddings  # rebuild the embedding cache
python classification_model.py --predict bird.jpg      # classify one image
python classification_model.py --device cpu            # force CPU inference
python classification_model.py --workers 2             # fewer loader workers
```

The embedding cache is regenerated as needed and is ignored by Git.

## Run the CAPTCHA game

The game is the presentation/demo. It randomly asks for one of the five birds,
gives three Verify attempts per grid, and shows a reference photo in the blue
header. The **Solve** button runs the EfficientNet-B0 + SVM pipeline on the
tiles and highlights predicted tiles in green. Inference runs on a background
thread so the window stays responsive, including on 30×30 grids.

Run from the repository root.

Linux/macOS:

```bash
./game/run_game.sh
```

Windows:

```bat
game\run_game.bat
```

You can double-click `run_game.bat` in Explorer. On Linux/macOS,
`run_game.sh` is executable; in a file manager choose **Run** or **Run in
Terminal** (some file managers default to opening text files in an editor).

Both scripts run `game/CaptchSparrowApp.py` directly and choose an interpreter
in this order:

1. `BIRD_PYTHON` / `PYTHON` environment override (Linux/macOS)
2. a portable Python next to the project (`../Python` or `../../Python`)
3. the repository `.venv` (or `game/.venv`)
4. system `python3` / `python` (Windows: `python`)

The scripts `cd` into `game/`, so `--dataset` paths are relative to it:

```bash
./game/run_game.sh --dataset birds_dataset.zip
./game/run_game.sh --dataset ../dataset/test
```

On Windows, use `game\run_game.bat` with the same arguments.

The game uses the included demo dataset by default, so it runs from a fresh
clone. See [game/README.md](game/README.md) for the full-dataset options.

## Method summary

- **Feature extractor:** EfficientNet-B0 pretrained on ImageNet-1K
  (`torchvision`, `EfficientNet_B0_Weights.IMAGENET1K_V1`), frozen, classifier
  head removed. Each image becomes a 1,280-dimensional embedding.
- **Preprocessing:** resize to 224×224 RGB, ImageNet mean/std normalization.
- **Classifier:** `StandardScaler` + RBF SVM (`C=10`, `gamma="scale"`,
  class-balanced weights), trained on the embeddings.
- **Validation:** grouped `StratifiedGroupKFold(n_splits=5)` by source photo is
  the primary metric (no `_aug` leakage); stratified 5-fold is reported for
  comparison. The test split is used only for the final one-shot evaluation.
- **Fine-tuning:** end-to-end fine-tuning was evaluated and did not improve on
  the frozen pipeline, so the simpler model is shipped.

## Limitations

- The five species are visually distinct and the images are clean 256×256
  crops, so accuracy is near the ceiling for this dataset. The 310-image test
  split has only one misclassified image; treat differences below about 1%
  between models as noise.
- The earlier handcrafted-feature baseline reported 73.31% with stratified CV.
  That figure was inflated by `_aug` copies crossing folds; its honest grouped
  score is 64.05%. The old pipeline and metrics are preserved in a local backup
  outside the repository (not committed).
- The frozen backbone is ImageNet-pretrained, not bird-specific. It works very
  well for these five classes but may need adaptation for fine-grained species.
- On CPU, a 30×30 Solve takes roughly 15–20 seconds; on a CUDA GPU it takes a
  couple of seconds. The UI stays responsive while it runs.

## License

Copyright (C) 2026 Zachary Xu and Jonathan Reppert.

This project is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See [LICENSE](LICENSE) for the full text.

The pretrained EfficientNet-B0 weights, PyTorch/torchvision, scikit-learn, and
the images.cv dataset have their own terms. See
[CREDITS.md](CREDITS.md) for citations, licenses, and attribution. Dataset
images are sourced from images.cv and remain subject to their respective terms;
they are not covered by this project's GPL license.
