# Bird CAPTCHA Game

A reCAPTCHA-style game. Each round randomly asks for one of five birds —
bald eagle, bee-eater, bustard, European gallinule, or sparrow — and you click
every tile containing that bird.

You get **3 Verify attempts per grid**. A wrong answer keeps the same grid so
you can adjust; after the third wrong answer, a new grid is generated. A small
example photo of the requested bird is shown in the blue header. The **Solve**
button embeds every tile with a frozen EfficientNet-B0 CNN, classifies the
embeddings with an RBF SVM, and highlights the tiles predicted as the requested
bird in green. Inference runs in a background thread, so the window stays
responsive.

## Run

The game runs on Linux, macOS and Windows. Install the dependencies once, then
use the unified `game_launch.py` from the repository root. The launcher works with
either a virtual environment or a plain Python installation.

### Option A — virtual environment (recommended, default)

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo pacman -S tk        # Arch/CachyOS: Tkinter system package
# Debian/Ubuntu:  sudo apt install python3-tk
# macOS (Homebrew Python):  brew install python-tk
```

Windows (PowerShell or Command Prompt), from the repository root:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tkinter is bundled with the official python.org Windows installer — make sure
**tcl/tk and IDLE** is checked. Virtual environments are not portable between
operating systems, so create a fresh `.venv` on each machine.

Activation is optional afterwards: `game_launch.py` finds `.venv` automatically.

### Option B — no virtual environment

If the Python you run already has the packages (system Python, conda, or
another environment), install into that interpreter instead:

Linux / macOS:

```bash
python3 -m pip install -r requirements.txt
```

Windows:

```bat
py -3 -m pip install -r requirements.txt
```

Then launch with `--no-venv` (see below). If pip reports a permissions error,
add `--user`. On distributions that mark the system Python as externally
managed (PEP 668, e.g. Arch/CachyOS), pip may refuse — use Option A or your
distro's packages.

### Launch

From the repository root:

```bash
python game_launch.py          # Linux / macOS
py game_launch.py              # Windows
```

`game_launch.py` uses the project `.venv` when it exists; otherwise it uses the
interpreter you invoked. Add `--no-venv` to ignore `.venv` and use the current
interpreter:

```bash
python game_launch.py --no-venv
```

Set `BIRD_PYTHON=/path/to/python` to choose a specific interpreter.

The old launchers still work as thin convenience wrappers around `game_launch.py`:

```bash
./game/run_game.sh        # Linux / macOS
game\run_game.bat         # Windows (also double-clickable)
```

`requirements.txt` includes PyTorch and torchvision. On the first Solve (or the
first training run) torchvision downloads the EfficientNet-B0 ImageNet weights,
about 20 MB. The game works on CPU; an NVIDIA GPU is detected automatically.
Set `BIRD_DEVICE=cpu` to force CPU or `BIRD_DEVICE=cuda` to require the GPU.

## Dataset options

The repository includes a small **demo dataset** in `demo_dataset/test/` with
10 test images per species, so the game runs from a fresh clone. It is separate
from the model's training data.

To use the full dataset zip (excluded from the repository because it is about
195 MB), place `birds_dataset.zip` in this `game/` folder and run from the
repository root:

```bash
python game_launch.py --dataset game/birds_dataset.zip
```

On Windows, use `py game_launch.py` instead. Other options (the same arguments work
with the wrapper scripts too):

```bash
# Use the training split instead (the model has seen these images)
python game_launch.py --dataset game/birds_dataset.zip --split train

# Use an extracted dataset folder
python game_launch.py --dataset dataset/test

# Ignore the project .venv and use the Python running game_launch.py
python game_launch.py --no-venv
```

## Files

| File | Purpose |
|---|---|
| `CaptchaSparrow.py` | Builds the grid, chooses the target bird and returns its true positions |
| `CaptchSparrowApp.py` | Tkinter UI, tile selection, three Verify attempts and the Solve button |
| `ModelSolver.py` | Loads the EfficientNet-B0 backbone and `../results/models/best_model.joblib`, embeds each tile and predicts the requested bird |
| `../game_launch.py` | Unified launcher for Linux, macOS and Windows; auto-detects `.venv` or uses the current Python |
| `run_game.sh` / `run_game.bat` | Optional convenience wrappers around `game_launch.py` |
| `demo_dataset/test/` | Small demo dataset (10 images per species) |
| `bird_samples/` | Example photo of each bird shown in the blue header |
| `captcha_grid.png` | Example generated grid |

## Model details

- Feature extractor: EfficientNet-B0 pretrained on ImageNet-1K (`torchvision`),
  frozen, classifier head removed — each tile becomes a 1,280-d embedding.
- Classifier: `../results/models/best_model.joblib` (StandardScaler + RBF SVM,
  `C=10`, `gamma="scale"`, balanced class weights).
- Preprocessing is defined in `../results/model_config.json`
  (224×224 resize, ImageNet normalization) and applied by `ModelSolver.py`.
- The solver reads each tile's **original** image from the zip/folder, not the
  downscaled grid tile, so small grids and the 30×30 absurd mode use the same
  image quality.
- Tiles are embedded in batches (32 by default; override with
  `BIRD_SOLVE_BATCH_SIZE`) and inference runs off the Tk main thread.

Model quality: 98.24% grouped 5-fold CV and 99.68% on the 310-image held-out
test split. The Solve button is very accurate for these five species, though it
can still make rare mistakes, so Verify remains the final check.
See the repository [README](../README.md) and [CREDITS](../CREDITS.md) for
metrics, citations, and licenses.

`birds_dataset.zip` is about 195 MB, so it is intentionally not committed.
Use the included demo dataset by default, or keep the full zip outside Git.

## Regenerate the model

From the repository root, with the full training dataset in `dataset/train/`:

```bash
python classification_model.py
```

This writes `results/models/best_model.joblib` and `results/model_config.json`.
The game loads those two files. Embedding extraction is cached in
`results/embeddings_*.npy` (Git-ignored); pass `--recompute-embeddings` to
rebuild it.
