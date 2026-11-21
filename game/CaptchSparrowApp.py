from __future__ import annotations

import argparse
import queue
import random
import sys
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError as exc:
    if sys.platform == "win32":
        tk_help = (
            "Tkinter ships with the official python.org Windows installer; "
            "re-run the installer and enable 'tcl/tk and IDLE'."
        )
    elif sys.platform == "darwin":
        tk_help = (
            "Install Tk support, e.g. 'brew install python-tk', "
            "or use the python.org installer."
        )
    else:
        tk_help = (
            "On Arch/CachyOS install it with 'sudo pacman -S tk'; "
            "on Debian/Ubuntu use 'sudo apt install python3-tk'."
        )
    raise SystemExit(f"Tkinter could not load. {tk_help}\n({exc})") from exc

from PIL import Image, ImageOps, ImageTk

from CaptchaSparrow import CaptchaGridResult, generate_bird_captcha_grid
from ModelSolver import solve_with_model

BIRD_NAMES = {
    "bald_eagle": "Bald eagle",
    "bee_eater": "Bee-eater",
    "bustard": "Bustard",
    "european_gallinule": "European gallinule",
    "sparrow": "Sparrow",
}
TARGET_BIRDS = list(BIRD_NAMES)

GAME_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = GAME_DIR / "bird_samples"
DEMO_DATASET = GAME_DIR / "demo_dataset"
REFERENCE_SIZE = 92
_REFERENCE_CACHE: dict[str, Image.Image] = {}


def _reference_image(bird: str) -> Image.Image | None:
    """Load and cache a square example photo for one bird."""
    if bird not in _REFERENCE_CACHE:
        path = REFERENCE_DIR / f"{bird}.jpg"
        if not path.is_file():
            return None
        with Image.open(path) as image:
            _REFERENCE_CACHE[bird] = ImageOps.fit(
                image.convert("RGB"),
                (REFERENCE_SIZE, REFERENCE_SIZE),
                method=Image.Resampling.LANCZOS,
            )
    return _REFERENCE_CACHE[bird]


# ==============================================================================
# RECAPTCHA-STYLE TKINTER UI
# ==============================================================================
class BirdCaptchaApp:
    BLUE = "#1A73E8"
    BLUE_DARK = "#1557B0"
    BG = "#FFFFFF"
    BORDER = "#DADCE0"
    TEXT_MUTED = "#5F6368"
    SUCCESS_GREEN = "#188038"
    ERROR_RED = "#D93025"

    def __init__(
        self,
        root: tk.Tk,
        dataset_path: str | None = None,
        split_folder: str = "test",
    ) -> None:
        self.root = root
        self.dataset_path = dataset_path
        self.split_folder = split_folder
        self.root.title("ReCAPTCHA - Bird Detector")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self.grid_result: CaptchaGridResult | None = None
        self.selected: set[tuple[int, int]] = set()
        self.overlay_items: dict[tuple[int, int], list[int]] = {}
        self.photo: ImageTk.PhotoImage | None = None
        self.reference_photo: ImageTk.PhotoImage | None = None
        self.max_attempts = 3
        self.attempts_left = self.max_attempts
        self.target_bird = "sparrow"
        self.target_display = BIRD_NAMES[self.target_bird]
        self._solve_queue: queue.Queue = queue.Queue()
        self._solve_generation = 0
        self._solving = False

        self._build_ui()
        self.load_new_grid(grid_size=None)

    def _build_ui(self) -> None:
        outer = tk.Frame(
            self.root,
            bg=self.BG,
            highlightbackground=self.BORDER,
            highlightthickness=1,
            padx=8,
            pady=8,
        )
        outer.pack(padx=10, pady=10)

        # 1. Blue ReCAPTCHA Header with an example photo of the target bird
        header = tk.Frame(outer, bg=self.BLUE, padx=18, pady=14)
        header.pack(fill="x")

        header_row = tk.Frame(header, bg=self.BLUE)
        header_row.pack(fill="x")

        text_column = tk.Frame(header_row, bg=self.BLUE)
        text_column.pack(side="left", fill="x", expand=True)

        tk.Label(
            text_column,
            text="Select all squares with",
            bg=self.BLUE,
            fg="#E8F0FE",
            font=("Segoe UI", 11),
            anchor="w",
        ).pack(fill="x")

        self.title_label = tk.Label(
            text_column,
            text="Bird",
            bg=self.BLUE,
            fg="#FFFFFF",
            font=("Segoe UI", 22, "bold"),
            anchor="w",
        )
        self.title_label.pack(fill="x")

        self.subtitle_label = tk.Label(
            text_column,
            text="",
            bg=self.BLUE,
            fg="#D2E3FC",
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.subtitle_label.pack(fill="x", pady=(4, 0))

        reference_frame = tk.Frame(header_row, bg=self.BLUE)
        reference_frame.pack(side="right", padx=(14, 0), anchor="center")

        self.reference_label = tk.Label(
            reference_frame,
            bg=self.BLUE,
            bd=0,
            highlightthickness=2,
            highlightbackground="#FFFFFF",
        )
        self.reference_label.pack()

        tk.Label(
            reference_frame,
            text="example",
            bg=self.BLUE,
            fg="#D2E3FC",
            font=("Segoe UI", 8),
        ).pack(pady=(2, 0))

        # 2. Interactive Image Grid Canvas
        self.canvas = tk.Canvas(
            outer,
            bg="#F1F3F4",
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack(pady=(8, 4))
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # 3. Inline Status / Feedback Banner
        self.status_label = tk.Label(
            outer,
            text="Click tiles to select the requested bird, then press Verify.",
            bg=self.BG,
            fg=self.TEXT_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=(2, 6))

        # 4. Bottom Action Bar (Reset, Solve, Verify)
        tk.Frame(outer, bg=self.BORDER, height=1).pack(fill="x", pady=(0, 8))

        footer = tk.Frame(outer, bg=self.BG)
        footer.pack(fill="x")

        self.reset_btn = tk.Button(
            footer,
            text="Reset",
            command=self._on_reset,
            bg="#F1F3F4",
            fg="#202124",
            activebackground="#E8EAED",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=7,
            cursor="hand2",
        )
        self.reset_btn.pack(side="left")

        self.solve_btn = tk.Button(
            footer,
            text="Solve",
            command=self._on_solve,
            bg="#F1F3F4",
            fg="#202124",
            activebackground="#E8EAED",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=7,
            cursor="hand2",
        )
        self.solve_btn.pack(side="left", padx=(8, 0))

        self.verify_btn = tk.Button(
            footer,
            text="VERIFY",
            command=self._on_verify,
            bg=self.BLUE,
            fg="#FFFFFF",
            activebackground=self.BLUE_DARK,
            activeforeground="#FFFFFF",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=20,
            pady=7,
            cursor="hand2",
        )
        self.verify_btn.pack(side="right")

        self.root.bind("<Return>", lambda _event: self._on_verify())

    def load_new_grid(self, grid_size: int | None = None) -> None:
        # Invalidate any in-flight Solve result for the previous grid.
        self._solve_generation += 1
        self.selected.clear()
        self.overlay_items.clear()

        self.target_bird = random.choice(TARGET_BIRDS)
        self.target_display = BIRD_NAMES[self.target_bird]
        self.attempts_left = self.max_attempts

        self.grid_result = generate_bird_captcha_grid(
            dataset_path=self.dataset_path,
            split_folder=self.split_folder,
            target_bird=self.target_bird,
            grid_size=grid_size,
        )
        img = self.grid_result.image
        self.photo = ImageTk.PhotoImage(img)

        self.canvas.config(width=img.width, height=img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.title_label.config(text=self.target_display)
        reference = _reference_image(self.target_bird)
        if reference is not None:
            self.reference_photo = ImageTk.PhotoImage(reference)
            self.reference_label.config(image=self.reference_photo, text="")
        else:
            self.reference_photo = None
            self.reference_label.config(
                image="",
                text="No sample",
                fg="#D2E3FC",
                font=("Segoe UI", 8),
            )
        self._update_subtitle()
        self.status_label.config(
            text=(
                f"Click tiles to select {self.target_display.lower()}s, "
                "then press Verify."
            ),
            fg=self.TEXT_MUTED,
        )

    def _update_subtitle(self) -> None:
        if self.grid_result is None:
            return
        n = self.grid_result.grid_size
        mode_note = " (ABSURD MODE!)" if n >= 20 else ""
        self.subtitle_label.config(
            text=(
                f"Grid: {n} x {n}{mode_note}  |  "
                f"Attempts left: {self.attempts_left}/{self.max_attempts}"
            )
        )

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.grid_result is None:
            return

        n = self.grid_result.grid_size
        cs = self.grid_result.cell_size
        gap = self.grid_result.gap
        step = cs + gap

        col = (event.x - gap) // step
        row = (event.y - gap) // step
        if not (0 <= row < n and 0 <= col < n):
            return

        # Ignore clicks that land strictly inside the gap between tiles
        local_x = (event.x - gap) % step
        local_y = (event.y - gap) % step
        if local_x >= cs or local_y >= cs:
            return

        pos = (row, col)
        if pos in self.selected:
            self._deselect_tile(pos)
        else:
            self._select_tile(pos)

    def _select_tile(
        self, pos: tuple[int, int], outline: str | None = None
    ) -> None:
        if self.grid_result is None or pos in self.selected:
            return
        self.selected.add(pos)

        row, col = pos
        cs = self.grid_result.cell_size
        gap = self.grid_result.gap
        x0 = gap + col * (cs + gap)
        y0 = gap + row * (cs + gap)
        x1 = x0 + cs
        y1 = y0 + cs

        border_w = 2 if cs <= 24 else 4
        items: list[int] = []
        color = self.BLUE if outline is None else outline

        # Selection border: blue for the player, green for the SVM.
        items.append(
            self.canvas.create_rectangle(
                x0 + 1,
                y0 + 1,
                x1 - 1,
                y1 - 1,
                outline=color,
                width=border_w,
            )
        )

        # Checkmark badge on normal grids, or inner highlight on tiny 30x30 tiles
        if cs >= 28:
            r = 10
            cx, cy = x0 + r + 4, y0 + r + 4
            items.append(
                self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r, fill=color, outline="#FFFFFF", width=1
                )
            )
            items.append(
                self.canvas.create_text(
                    cx, cy, text="✓", fill="#FFFFFF", font=("Segoe UI", 10, "bold")
                )
            )
        else:
            items.append(
                self.canvas.create_rectangle(
                    x0 + 3, y0 + 3, x1 - 3, y1 - 3, fill=color, stipple="gray50", outline=""
                )
            )

        self.overlay_items[pos] = items

    def _deselect_tile(self, pos: tuple[int, int]) -> None:
        self.selected.discard(pos)
        for item_id in self.overlay_items.pop(pos, []):
            self.canvas.delete(item_id)

    def _on_verify(self) -> None:
        if self.grid_result is None or self._solving:
            return

        expected = set(self.grid_result.target_positions)
        if self.selected == expected:
            self.status_label.config(
                text=(
                    f"Success! All {self.target_display.lower()}s "
                    "correctly identified."
                ),
                fg=self.SUCCESS_GREEN,
            )
            messagebox.showinfo(
                "Success",
                f"Correct! You found all {len(expected)} "
                f"{self.target_display.lower()}(s) in the "
                f"{self.grid_result.grid_size}x{self.grid_result.grid_size} grid.",
            )
            self.load_new_grid(grid_size=None)
            return

        self.attempts_left -= 1
        self._update_subtitle()
        if self.attempts_left > 0:
            self.status_label.config(
                text=(
                    f"Incorrect selection. {self.attempts_left} attempt(s) "
                    "left on this grid."
                ),
                fg=self.ERROR_RED,
            )
            return

        same_size = self.grid_result.grid_size
        self.load_new_grid(grid_size=same_size)
        self.status_label.config(
            text=(
                f"Out of attempts — generated a new "
                f"{same_size}x{same_size} grid."
            ),
            fg=self.ERROR_RED,
        )

    def _on_reset(self) -> None:
        self.load_new_grid(grid_size=None)
        self.status_label.config(
            text="Reset: generated a new random grid.",
            fg=self.TEXT_MUTED,
        )

    def _on_solve(self) -> None:
        if self.grid_result is None or self._solving:
            return

        self._solving = True
        generation = self._solve_generation
        grid_result = self.grid_result
        target_bird = self.target_bird
        n = grid_result.grid_size

        self.solve_btn.config(state="disabled")
        self.verify_btn.config(state="disabled")
        self.reset_btn.config(state="disabled")
        self.status_label.config(
            text=f"Running EfficientNet-B0 + SVM on {n * n} tiles...",
            fg=self.TEXT_MUTED,
        )

        def worker() -> None:
            try:
                predicted = solve_with_model(grid_result, target_bird)
            except Exception as exc:  # keep the game usable if the model fails
                self._solve_queue.put((generation, "error", exc))
            else:
                self._solve_queue.put((generation, "ok", predicted))

        threading.Thread(target=worker, name="bird-solve", daemon=True).start()
        self.root.after(100, self._poll_solve_result)

    def _poll_solve_result(self) -> None:
        try:
            generation, status, payload = self._solve_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_solve_result)
            return

        self._solving = False
        self.solve_btn.config(state="normal")
        self.verify_btn.config(state="normal")
        self.reset_btn.config(state="normal")

        # Ignore results for a grid that has since been replaced or reset.
        if generation != self._solve_generation:
            return

        if status == "error":
            self.status_label.config(
                text=f"Model error: {payload}", fg=self.ERROR_RED
            )
            messagebox.showerror("Model error", str(payload))
            return

        predicted = payload
        if not predicted:
            self.status_label.config(
                text=(
                    f"The model did not predict any "
                    f"{self.target_display.lower()}s in this grid."
                ),
                fg=self.TEXT_MUTED,
            )
            return

        for pos in list(self.selected):
            self._deselect_tile(pos)
        for pos in predicted:
            self._select_tile(pos, outline=self.SUCCESS_GREEN)
        self.status_label.config(
            text=(
                f"Solve: model highlighted {len(predicted)} tile(s) as "
                f"{self.target_display.lower()}. Press VERIFY to check."
            ),
            fg=self.SUCCESS_GREEN,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bird CAPTCHA game with an EfficientNet-B0 + SVM solver."
    )
    default_dataset = str(DEMO_DATASET) if DEMO_DATASET.is_dir() else None
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help=(
            "Dataset zip or folder "
            "(default: game/demo_dataset, or birds_dataset.zip if supplied)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("test", "train"),
        default="test",
        help="Dataset split to use (default: test).",
    )
    args = parser.parse_args()

    root = tk.Tk()
    BirdCaptchaApp(root, dataset_path=args.dataset, split_folder=args.split)
    root.mainloop()


if __name__ == "__main__":
    main()