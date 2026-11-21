from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class CaptchaGridResult:
    """Result returned by `generate_bird_captcha_grid`."""

    image: Image.Image
    grid_size: int
    cell_size: int
    gap: int
    sparrow_positions: list[tuple[int, int]]  # (row, col) of the target bird
    labels: list[list[str]]  # N x N matrix of bird folder names
    image_paths: list[list[str]]  # N x N matrix of source image paths / zip entries
    source: str = ""  # dataset zip or folder used to build the grid
    is_zip: bool = False  # True when source is a zip archive
    target_bird: str = "sparrow"  # bird the player must select

    @property
    def target_positions(self) -> list[tuple[int, int]]:
        """Positions of the target bird (sparrow_positions is the legacy name)."""
        return self.sparrow_positions


def _resolve_dataset_path(dataset_path: Optional[str | Path]) -> Path:
    """
    Locate the dataset `.zip` (or extracted folder) regardless of what the
    containing folder is named or which working directory Python is run from.
    """
    if dataset_path is not None:
        candidate = Path(dataset_path)
        if candidate.exists():
            return candidate
        if candidate.suffix != ".zip" and candidate.with_suffix(".zip").exists():
            return candidate.with_suffix(".zip")

        # If a relative path wasn't found in CWD, check next to this Python file
        if not candidate.is_absolute():
            rel_candidate = SCRIPT_DIR / candidate
            if rel_candidate.exists():
                return rel_candidate
            if (
                rel_candidate.suffix != ".zip"
                and rel_candidate.with_suffix(".zip").exists()
            ):
                return rel_candidate.with_suffix(".zip")

        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Auto-detect inside the same folder as this Python script
    preferred_zip = SCRIPT_DIR / "birds_dataset.zip"
    if preferred_zip.is_file():
        return preferred_zip

    zips = sorted(SCRIPT_DIR.glob("*.zip"))
    if len(zips) == 1:
        return zips[0]
    if len(zips) > 1:
        for zp in zips:
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    if any(
                        "test" in PurePosixPath(name).parts[:-1]
                        for name in zf.namelist()
                    ):
                        return zp
            except zipfile.BadZipFile:
                continue
        return zips[0]

    for folder_name in ("birds_dataset", "test"):
        folder = SCRIPT_DIR / folder_name
        if folder.is_dir():
            return folder

    raise FileNotFoundError(
        f"No .zip dataset or 'birds_dataset'/'test' folder found in {SCRIPT_DIR}"
    )


def _collect_from_zip(
    zip_path: Path,
    split_folder: str = "test",
) -> dict[str, list[str]]:
    """Scan `<split_folder>/<bird_name>/<image>` inside a `.zip` archive."""
    bird_images: dict[str, list[str]] = {}
    target_split = split_folder.lower()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            if "__MACOSX" in parts or parts[-1].startswith("._"):
                continue
            if PurePosixPath(info.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Support both `test/<bird>/<img.jpg>` and `<root>/test/<bird>/<img.jpg>`
            for idx, part in enumerate(parts):
                if part.lower() == target_split and len(parts) >= idx + 3:
                    bird_name = parts[idx + 1]
                    bird_images.setdefault(bird_name, []).append(info.filename)
                    break

    return bird_images


def _collect_from_dir(
    root_dir: Path,
    split_folder: str = "test",
) -> dict[str, list[str]]:
    """Scan an extracted directory (`<root>/<split>/<bird>/<image>` or `<root>/<bird>/<image>`)."""
    candidate = root_dir / split_folder
    test_dir = candidate if candidate.is_dir() else root_dir

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    bird_images: dict[str, list[str]] = {}
    for subfolder in sorted(test_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        files = [
            str(p)
            for p in sorted(subfolder.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if files:
            bird_images[subfolder.name] = files

    return bird_images


def generate_bird_captcha_grid(
    dataset_path: Optional[str | Path] = None,
    target_bird: str = "sparrow",
    split_folder: str = "test",
    grid_size: Optional[int] = None,
    huge_chance: float = 0.30,
    huge_size: int = 30,
    cell_size: Optional[int] = None,
    gap: Optional[int] = None,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    allow_duplicates: Optional[bool] = None,
    seed: Optional[int] = None,
    output_path: Optional[str | Path] = None,
) -> CaptchaGridResult:
    """
    Generate an N x N CAPTCHA-style grid of random bird images from a `.zip` dataset
    (or extracted folder) in the same folder as this script, guaranteeing at least one
    image from `<split_folder>/<target_bird>`.

    When `grid_size` is None:
      - `huge_chance` (default 30%) probability of generating an absurdly huge `huge_size` (30x30) grid.
      - Otherwise picks a random size from 3x3 up to 10x10.
    """
    rng = random.Random(seed)

    if grid_size is None:
        if rng.random() < huge_chance:
            n = huge_size
        else:
            n = rng.randint(3, 10)
    else:
        if not (3 <= grid_size <= huge_size):
            raise ValueError(
                f"grid_size must be between 3 and {huge_size}, got {grid_size}"
            )
        n = grid_size

    # Auto-scale tile size and gap so both 3x3 and 30x30 fit nicely on screen
    if gap is None:
        gap = 1 if n >= 20 else 3
    if cell_size is None:
        # Target ~540px total canvas width
        cell_size = max(16, min(140, (540 - (n + 1) * gap) // n))

    source = _resolve_dataset_path(dataset_path)
    is_zip = source.is_file() and zipfile.is_zipfile(source)
    bird_images = (
        _collect_from_zip(source, split_folder=split_folder)
        if is_zip
        else _collect_from_dir(source, split_folder=split_folder)
    )

    if not bird_images:
        raise FileNotFoundError(
            f"No images found under '{split_folder}/<bird_name>/' in '{source}'."
        )

    # Match target_bird folder (exact match first, then case-insensitive)
    if target_bird in bird_images:
        target_key = target_bird
    else:
        matches = [k for k in bird_images if k.lower() == target_bird.lower()]
        if not matches:
            raise FileNotFoundError(
                f"No images found for '{target_bird}' under '{split_folder}/' in '{source}'."
            )
        target_key = matches[0]

    total_cells = n * n
    target_pool = [(target_key, p) for p in bird_images[target_key]]
    all_pool = [(bird, p) for bird, paths in bird_images.items() for p in paths]

    # 1. Guarantee at least one image of the target bird
    chosen_target = rng.choice(target_pool)
    remaining_needed = total_cells - 1

    # 2. Sample the remaining (N*N - 1) images from all available birds
    non_chosen_pool = [item for item in all_pool if item != chosen_target]
    use_duplicates = (
        allow_duplicates
        if allow_duplicates is not None
        else len(non_chosen_pool) < remaining_needed
    )

    if use_duplicates:
        other_choices = rng.choices(all_pool, k=remaining_needed)
    else:
        if len(non_chosen_pool) < remaining_needed:
            raise ValueError(
                f"Not enough unique images ({len(all_pool)}) to fill a {n}x{n} grid "
                f"({total_cells} cells). Set allow_duplicates=True to allow repeats."
            )
        other_choices = rng.sample(non_chosen_pool, k=remaining_needed)

    selected = [chosen_target] + other_choices
    rng.shuffle(selected)

    # 3. Stitch tiles into an N x N grid image
    canvas_size = n * cell_size + (n + 1) * gap
    canvas = Image.new("RGB", (canvas_size, canvas_size), color=bg_color)
    resampling = (
        Image.Resampling.BILINEAR if cell_size <= 32 else Image.Resampling.LANCZOS
    )

    target_positions: list[tuple[int, int]] = []
    labels: list[list[str]] = []
    image_paths: list[list[str]] = []
    tile_cache: dict[str, Image.Image] = {}

    zf = zipfile.ZipFile(source, "r") if is_zip else None
    try:
        for row in range(n):
            row_labels: list[str] = []
            row_paths: list[str] = []
            for col in range(n):
                idx = row * n + col
                bird_name, img_ref = selected[idx]

                if img_ref in tile_cache:
                    tile = tile_cache[img_ref]
                else:
                    if zf is not None:
                        with zf.open(img_ref) as f, Image.open(f) as img:
                            tile = ImageOps.fit(
                                img.convert("RGB"),
                                (cell_size, cell_size),
                                method=resampling,
                            )
                    else:
                        with Image.open(img_ref) as img:
                            tile = ImageOps.fit(
                                img.convert("RGB"),
                                (cell_size, cell_size),
                                method=resampling,
                            )
                    tile_cache[img_ref] = tile

                x = gap + col * (cell_size + gap)
                y = gap + row * (cell_size + gap)
                canvas.paste(tile, (x, y))

                if bird_name == target_key:
                    target_positions.append((row, col))
                row_labels.append(bird_name)
                row_paths.append(img_ref)

            labels.append(row_labels)
            image_paths.append(row_paths)
    finally:
        if zf is not None:
            zf.close()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)

    return CaptchaGridResult(
        image=canvas,
        grid_size=n,
        cell_size=cell_size,
        gap=gap,
        sparrow_positions=target_positions,
        labels=labels,
        image_paths=image_paths,
        source=str(source),
        is_zip=is_zip,
        target_bird=target_key,
    )


if __name__ == "__main__":
    result = generate_bird_captcha_grid(
        output_path=SCRIPT_DIR / "captcha_grid.png",
    )
    print(f"Generated {result.grid_size}x{result.grid_size} grid.")
    print(f"Target bird: {result.target_bird}")
    print(f"Target tile coordinates (row, col): {result.target_positions}")