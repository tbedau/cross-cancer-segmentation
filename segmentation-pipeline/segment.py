"""Histopathology segmentation inference for TCGA whole-slide images.

Runs patch-based semantic segmentation using models trained with
segmentation_models_pytorch. Supports cross-entity application of models
trained on specific cancer types.

Usage:
    uv run segment.py --model PRAD --image test_image/example.jpg
    uv run segment.py --model BRCA --image-dir path/to/images/
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import torch
import typer
from numpy.typing import NDArray
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODER = "timm-efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
MODEL_PATCH_SIZE = 512
MODEL_MPP = 1.0  # Microns per pixel at which models were trained
DEFAULT_SLIDE_MPP = 0.461
OVERLAY_ALPHA = 0.3
MAX_IMAGE_PIXELS = 125_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Overlap / stitching geometry (derived from MODEL_PATCH_SIZE and 75% stride)
_OVERLAP_PX = MODEL_PATCH_SIZE - int(MODEL_PATCH_SIZE * 0.75)  # 128
_OVERLAP_MARGIN = _OVERLAP_PX // 2  # 64
_LEADING_EDGE = MODEL_PATCH_SIZE - _OVERLAP_PX  # 384
_CUTOFF = _LEADING_EDGE + _OVERLAP_MARGIN  # 448

console = Console()
app = typer.Typer(
    help="Run histopathology segmentation on TCGA images.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ---------------------------------------------------------------------------
# Color palettes (index 0 → class 1, last entry is informational)
# ---------------------------------------------------------------------------


class ColorScheme(enum.StrEnum):
    """Color scheme for the segmentation overlay."""

    SIMPLIFIED = "simplified"
    DETAILED = "detailed"


_COLORS_SIMPLIFIED: dict[str, list[list[int]]] = {
    "BREAST": [
        [0, 0, 255],  # TUMOR (blue)
        [255, 165, 0],  # TUMOR STROMA (orange)
        [0, 0, 255],  # DCIS (blue)
        [0, 0, 255],  # LCIS (blue)
        [128, 128, 128],  # NECROSIS (gray)
        [128, 128, 128],  # MUCIN (gray)
        [128, 128, 128],  # INFLAM (gray)
        [128, 128, 128],  # FAT (gray)
        [128, 128, 128],  # STROMA (gray)
        [128, 128, 128],  # BLOOD (gray)
        [128, 128, 128],  # SKIN (gray)
        [0, 0, 255],  # BEN EPIT / SKIN ADNEX (blue)
        [255, 255, 255],  # BACK (white)
    ],
    "COLON": [
        [0, 0, 255],  # TUMOR / ADENOM_HG (blue)
        [128, 128, 128],  # MUC / ADENOM_LG (gray)
        [255, 165, 0],  # TU_STROMA (orange)
        [128, 128, 128],  # SUBMUC (gray)
        [128, 128, 128],  # MUSC_PROP / MUSC_MUC (gray)
        [128, 128, 128],  # ADVENT / VESSEL (gray)
        [128, 128, 128],  # LYMPH_NODE / LYMPH_TIS / LYM_AGGR (gray)
        [128, 128, 128],  # ULCUS / NECROSIS (gray)
        [128, 128, 128],  # BLOOD (gray)
        [128, 128, 128],  # MUCIN (gray)
        [255, 255, 255],  # BACK (white)
    ],
    "LUNG": [
        [0, 0, 255],  # TUMOR (blue)
        [255, 165, 0],  # TUMOR STROMA (orange)
        [128, 128, 128],  # NECROSIS (gray)
        [128, 128, 128],  # MUCIN (gray)
        [128, 128, 128],  # BENIGN LUNG (gray)
        [128, 128, 128],  # STROMA / NERVE / FAT / MUSCLE / VESSEL (gray)
        [128, 128, 128],  # BLOOD (gray)
        [128, 128, 128],  # BRONCHUS (gray)
        [128, 128, 128],  # CARTILAGE (gray)
        [128, 128, 128],  # GLAND_BRONCH (gray)
        [128, 128, 128],  # LYMPH_AGGR / LYMPH_NODE (gray)
        [255, 255, 255],  # BACK (white)
    ],
    "KIDNEY": [
        [0, 0, 255],  # TUMOR (blue)
        [128, 128, 128],  # TUMOR_REGRESS (gray)
        [128, 128, 128],  # NECROSIS (gray)
        [128, 128, 128],  # KIDNEY_BENIGN (gray)
        [128, 128, 128],  # UROTHEL (gray)
        [128, 128, 128],  # FAT (gray)
        [128, 128, 128],  # STROMA (gray)
        [128, 128, 128],  # BLOOD (gray)
        [128, 128, 128],  # ADRENAL (gray)
        [255, 255, 255],  # BACK (white)
    ],
    "PROSTATE": [
        [0, 0, 255],  # TUMOR (blue)
        [128, 128, 128],  # N (gray)
        [255, 165, 0],  # N_STR (orange)
        [255, 255, 255],  # BACK (white)
    ],
}

_COLORS_DETAILED: dict[str, list[list[int]]] = {
    "BREAST": [
        [0, 0, 255],  # TUMOR (blue)
        [255, 165, 0],  # TUMOR STROMA (orange)
        [0, 0, 255],  # DCIS (blue)
        [0, 0, 255],  # LCIS (blue)
        [152, 78, 163],  # NECROSIS (purple)
        [247, 129, 191],  # MUCIN (pink)
        [228, 26, 28],  # INFLAM (red)
        [166, 86, 40],  # FAT (brown)
        [166, 86, 40],  # STROMA (brown)
        [128, 128, 128],  # BLOOD (gray)
        [77, 175, 74],  # SKIN (green)
        [77, 175, 74],  # BEN EPIT / SKIN ADNEX (green)
        [255, 255, 255],  # BACK (white)
    ],
    "COLON": [
        [0, 0, 255],  # TUMOR / ADENOM_HG (blue)
        [77, 175, 74],  # MUC / ADENOM_LG (green)
        [255, 165, 0],  # TU_STROMA (orange)
        [77, 175, 74],  # SUBMUC (green)
        [77, 175, 74],  # MUSC_PROP / MUSC_MUC (green)
        [166, 86, 40],  # ADVENT / VESSEL (brown)
        [228, 26, 28],  # LYMPH_NODE / LYMPH_TIS / LYM_AGGR (red)
        [152, 78, 163],  # ULCUS / NECROSIS (purple)
        [128, 128, 128],  # BLOOD (gray)
        [247, 129, 191],  # MUCIN (pink)
        [255, 255, 255],  # BACK (white)
    ],
    "LUNG": [
        [0, 0, 255],  # TUMOR (blue)
        [255, 165, 0],  # TUMOR STROMA (orange)
        [152, 78, 163],  # NECROSIS (purple)
        [247, 129, 191],  # MUCIN (pink)
        [77, 175, 74],  # BENIGN LUNG (green)
        [166, 86, 40],  # STROMA / NERVE / FAT / MUSCLE / VESSEL (brown)
        [128, 128, 128],  # BLOOD (gray)
        [77, 175, 74],  # BRONCHUS (green)
        [77, 175, 74],  # CARTILAGE (green)
        [77, 175, 74],  # GLAND_BRONCH (green)
        [228, 26, 28],  # LYMPH_AGGR / LYMPH_NODE (red)
        [255, 255, 255],  # BACK (white)
    ],
    "KIDNEY": [
        [0, 0, 255],  # TUMOR (blue)
        [128, 128, 128],  # TUMOR_REGRESS (gray)
        [152, 78, 163],  # NECROSIS (purple)
        [77, 175, 74],  # KIDNEY_BENIGN (green)
        [77, 175, 74],  # UROTHEL (green)
        [166, 86, 40],  # FAT (brown)
        [255, 165, 0],  # STROMA (orange)
        [128, 128, 128],  # BLOOD (gray)
        [77, 175, 74],  # ADRENAL (green)
        [255, 255, 255],  # BACK (white)
    ],
    "PROSTATE": [
        [0, 0, 255],  # TUMOR (blue)
        [77, 175, 74],  # N (green)
        [255, 165, 0],  # N_STR (orange)
        [255, 255, 255],  # BACK (white)
    ],
}

COLORS: dict[ColorScheme, dict[str, list[list[int]]]] = {
    ColorScheme.SIMPLIFIED: _COLORS_SIMPLIFIED,
    ColorScheme.DETAILED: _COLORS_DETAILED,
}


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class Model(enum.Enum):
    """Available segmentation models mapped to their directory names."""

    BREAST = "04_brca_031"
    # Requires segmentation-models-pytorch==0.2.1 (change in pyproject.toml + uv sync)
    COLON = "01_coadread_021"
    LUNG = "03_luad_lusc_031"
    KIDNEY = "05_kidney_030_or_031"
    PROSTATE = "02_prad_030"
    DEMO = "demo"

    @property
    def number(self) -> int:
        """Return the numeric model identifier used in output filenames."""
        return {
            Model.BREAST: 1,
            Model.COLON: 2,
            Model.LUNG: 3,
            Model.KIDNEY: 4,
            Model.PROSTATE: 5,
            Model.DEMO: 0,
        }[self]

    def colors(self, scheme: ColorScheme = ColorScheme.SIMPLIFIED) -> list[list[int]]:
        """Return the color palette for this model under the given scheme."""
        key = "LUNG" if self is Model.DEMO else self.name
        return COLORS[scheme][key]


MODEL_NAMES = [m.name for m in Model]


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceConfig:
    """Resolved runtime configuration for a segmentation run."""

    model: Model
    model_dir: Path
    image_paths: list[Path]
    output_dir: Path
    device: str = "cpu"
    color_scheme: ColorScheme = ColorScheme.SIMPLIFIED
    overlay_alpha: float = OVERLAY_ALPHA
    downsample_factor: int | None = None
    mpp_override: float | None = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_dir: Path, device: str) -> torch.nn.Module:
    """Load the single .pth model file from *model_dir*.

    Raises:
        FileNotFoundError: If no .pth file is found.
        RuntimeError: If multiple .pth files exist or loading fails.
    """
    pth_files = sorted(model_dir.glob("*.pth"))
    if len(pth_files) == 0:
        msg = f"No .pth model file found in {model_dir}"
        raise FileNotFoundError(msg)
    if len(pth_files) > 1:
        msg = f"Expected exactly one .pth file in {model_dir}, found {len(pth_files)}"
        raise RuntimeError(msg)

    model_path = pth_files[0]
    console.print(f"Loading model from [cyan]{model_path}[/cyan]")

    # Models are saved as full objects (not state_dict), so weights_only=False
    # is required. Only load model files you trust.
    model: torch.nn.Module = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def get_preprocessing_fn() -> Callable[..., Any]:
    """Return the smp encoder preprocessing function."""
    import segmentation_models_pytorch as smp

    return smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def preprocess_patch(
    patch: Image.Image,
    preprocessing_fn: Callable[..., Any],
) -> NDArray[np.floating[Any]]:
    """Resize *patch* to model input size and apply encoder preprocessing.

    Returns a CHW float32 array ready for ``torch.from_numpy``.
    """
    resized = patch.resize((MODEL_PATCH_SIZE, MODEL_PATCH_SIZE))
    arr = np.array(resized)
    processed: NDArray[np.floating[Any]] = preprocessing_fn(arr)
    return processed.transpose(2, 0, 1).astype("float32")


# ---------------------------------------------------------------------------
# MPP helpers
# ---------------------------------------------------------------------------


def extract_mpp_from_filename(filename: str) -> float:
    """Extract microns-per-pixel from a TCGA filename.

    Looks for the pattern ``_<float>_<tumor|normal>_original.jpg``.
    Returns *DEFAULT_SLIDE_MPP* when not found.
    """
    match = re.search(r"_(\d+\.\d+)_(?:tumor|normal)_original\.jpg$", filename)
    return float(match.group(1)) if match else DEFAULT_SLIDE_MPP


def compute_patch_size(slide_mpp: float) -> int:
    """Compute the patch size in slide pixels for a given slide MPP."""
    return int(MODEL_MPP / slide_mpp * MODEL_PATCH_SIZE)


# ---------------------------------------------------------------------------
# Image padding & patch extraction
# ---------------------------------------------------------------------------


def pad_image(
    image: NDArray[np.uint8],
    patch_size_pixels: int,
) -> tuple[NDArray[np.uint8], int, int]:
    """Pad *image* with white so it can be evenly tiled into patches.

    Returns ``(padded_image, original_height, original_width)``.
    """
    orig_h, orig_w = image.shape[:2]
    new_h = orig_h + patch_size_pixels
    new_w = orig_w + patch_size_pixels
    padded: NDArray[np.uint8] = np.full((new_h, new_w, 3), fill_value=255, dtype=np.uint8)
    padded[:orig_h, :orig_w, :] = image
    return padded, orig_h, orig_w


def extract_patch(
    padded_image: Image.Image,
    col: int,
    row: int,
    patch_size: int,
    stride: int,
) -> Image.Image:
    """Extract a single patch at grid position (*col*, *row*)."""
    x = col * stride
    y = row * stride
    return padded_image.crop((x, y, x + patch_size, y + patch_size))


# ---------------------------------------------------------------------------
# Mask stitching
# ---------------------------------------------------------------------------


def stitch_masks(mask_grid: list[list[NDArray[np.int8]]]) -> NDArray[np.int8]:
    """Assemble a 2-D grid of model-resolution mask patches into one mask.

    Each patch is ``(MODEL_PATCH_SIZE, MODEL_PATCH_SIZE)``.  The stitching
    discards overlap margins (64 px) at the leading edge of each non-first
    row/column and at the trailing edge of every patch, keeping the center
    384 px strip (or 448 px for the first row/column which includes the
    leading edge).
    """
    rows: list[NDArray[np.int8]] = []
    for h, row_masks in enumerate(mask_grid):
        h_start = 0 if h == 0 else _OVERLAP_MARGIN
        strips: list[NDArray[np.int8]] = []
        for w, mask in enumerate(row_masks):
            w_start = 0 if w == 0 else _OVERLAP_MARGIN
            strips.append(mask[h_start:_CUTOFF, w_start:_CUTOFF])
        rows.append(np.concatenate(strips, axis=1))
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Mask post-processing
# ---------------------------------------------------------------------------


def resize_mask_to_original(
    stitched_mask: NDArray[np.int8],
    slide_mpp: float,
    original_width: int,
    original_height: int,
) -> NDArray[np.uint8]:
    """Resize the stitched mask back to the original image dimensions.

    Scales by ``MODEL_MPP / slide_mpp`` (nearest-neighbor) and crops.
    """
    mask_h, mask_w = stitched_mask.shape
    target_w = int(mask_w * MODEL_MPP / slide_mpp)
    target_h = int(mask_h * MODEL_MPP / slide_mpp)

    mask_image = Image.fromarray(stitched_mask)
    resized = mask_image.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    cropped = resized.crop((0, 0, original_width, original_height))
    return np.array(cropped, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def create_color_map(
    mask: NDArray[np.uint8],
    class_colors: list[list[int]],
) -> NDArray[np.uint8]:
    """Convert a class-index mask to an RGB color image.

    Class 0 (background) is left black.  Classes 1..N-1 are mapped to
    ``class_colors[0]`` through ``class_colors[N-2]``.
    """
    r = np.zeros_like(mask, dtype=np.uint8)
    g = np.zeros_like(mask, dtype=np.uint8)
    b = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, len(class_colors)):
        idx = mask == label
        r[idx] = class_colors[label - 1][0]
        g[idx] = class_colors[label - 1][1]
        b[idx] = class_colors[label - 1][2]
    return np.stack([r, g, b], axis=2)


def create_overlay(
    original: Image.Image,
    color_mask: NDArray[np.uint8],
    alpha: float,
) -> Image.Image:
    """Blend *original* with the *color_mask* at the given *alpha*."""
    return Image.blend(
        original.convert("RGBA"),
        Image.fromarray(color_mask).convert("RGBA"),
        alpha=alpha,
    )


def save_outputs(
    mask: NDArray[np.uint8],
    color_mask: NDArray[np.uint8],
    overlay: Image.Image,
    output_dir: Path,
    stem: str,
    downsample_factor: int | None,
) -> None:
    """Save the three output files for a single image.

    Files created:
        ``<stem>_mask.png``     — Grayscale class-index mask
        ``<stem>_color.png``    — Color-mapped segmentation
        ``<stem>_overlay.jpg``  — Blended overlay on original
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grayscale mask (always full resolution)
    Image.fromarray(mask).convert("L").save(output_dir / f"{stem}_mask.png")

    # Color map and overlay (optionally downsampled)
    color_img = Image.fromarray(color_mask)
    if downsample_factor is not None and downsample_factor > 1:
        ds_size = (
            color_img.width // downsample_factor,
            color_img.height // downsample_factor,
        )
        color_img = color_img.resize(ds_size, Image.Resampling.LANCZOS)
        overlay = overlay.resize(ds_size, Image.Resampling.LANCZOS)

    color_img.save(output_dir / f"{stem}_color.png")
    overlay.convert("RGB").save(output_dir / f"{stem}_overlay.jpg")


# ---------------------------------------------------------------------------
# Per-image inference pipeline
# ---------------------------------------------------------------------------


def segment_image(
    image_path: Path,
    model: torch.nn.Module,
    config: InferenceConfig,
    preprocessing_fn: Callable[..., Any],
    progress: Progress,
) -> None:
    """Run the full segmentation pipeline on a single image."""
    image = Image.open(image_path)
    image_np: NDArray[np.uint8] = np.array(image)

    # Determine slide MPP
    slide_mpp = config.mpp_override or extract_mpp_from_filename(image_path.name)
    console.print(f"Processing [bold]{image_path.name}[/bold] (MPP={slide_mpp:.3f})")

    # Pad image
    patch_size_pixels = compute_patch_size(slide_mpp)
    padded, orig_h, orig_w = pad_image(image_np, patch_size_pixels)
    padded_pil = Image.fromarray(padded)
    padded_h, padded_w = padded.shape[:2]

    # Compute patch grid
    stride = int(patch_size_pixels * 0.75)
    n_cols = (padded_w - patch_size_pixels) // stride + 1
    n_rows = (padded_h - patch_size_pixels) // stride + 1
    total_patches = n_rows * n_cols

    # Predict all patches
    mask_grid: list[list[NDArray[np.int8]]] = []
    task = progress.add_task(f"[cyan]{image_path.name}[/cyan]", total=total_patches)

    for row in range(n_rows):
        row_masks: list[NDArray[np.int8]] = []
        for col in range(n_cols):
            patch = extract_patch(padded_pil, col, row, patch_size_pixels, stride)
            tensor = preprocess_patch(patch, preprocessing_fn)
            x = torch.from_numpy(tensor).unsqueeze(0).to(config.device)
            with torch.inference_mode():
                predictions = model.predict(x)  # type: ignore[operator]  # SMP model method
            mask: NDArray[np.int8] = np.argmax(predictions.squeeze().cpu().numpy(), axis=0).astype(
                np.int8
            )
            row_masks.append(mask)
            progress.advance(task)
        mask_grid.append(row_masks)

    progress.remove_task(task)

    # Stitch and resize
    stitched = stitch_masks(mask_grid)
    final_mask = resize_mask_to_original(stitched, slide_mpp, orig_w, orig_h)

    # Generate color outputs
    colors = config.model.colors(config.color_scheme)
    color_mask = create_color_map(final_mask, colors)
    overlay = create_overlay(image, color_mask, config.overlay_alpha)

    # Build output stem: strip "_original" suffix, append model number
    base_stem = image_path.stem.removesuffix("_original")
    output_stem = f"{base_stem}_model{config.model.number}"

    # Save
    save_outputs(
        mask=final_mask,
        color_mask=color_mask,
        overlay=overlay,
        output_dir=config.output_dir,
        stem=output_stem,
        downsample_factor=config.downsample_factor,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _configure_deterministic(device: str) -> None:
    """Enable deterministic operations for reproducible results."""
    if device.startswith("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_inference(config: InferenceConfig) -> None:
    """Load the model and process all images in *config*."""
    _configure_deterministic(config.device)
    model = load_model(config.model_dir, config.device)
    preprocessing_fn = get_preprocessing_fn()

    console.print(
        f"Segmenting [bold]{len(config.image_paths)}[/bold] image(s) "
        f"with [green]{config.model.name}[/green] model"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        images_task = progress.add_task("[bold]Images[/bold]", total=len(config.image_paths))

        for image_path in config.image_paths:
            try:
                segment_image(image_path, model, config, preprocessing_fn, progress)
            except Exception as e:
                console.print(f"[red]Failed to process {image_path.name}: {e}[/red]")
            progress.advance(images_task)


def _detect_device() -> str:
    """Auto-detect the best available torch device."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_model(value: str | None) -> Model:
    """Parse model name string to Model enum."""
    if value is None:
        raise typer.BadParameter(f"Missing required option. Choose from: {', '.join(MODEL_NAMES)}")
    try:
        return Model[value.upper()]
    except KeyError:
        raise typer.BadParameter(f"'{value}' is not one of {', '.join(MODEL_NAMES)}") from None


@app.command()
def main(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Segmentation model to use.", callback=_parse_model),
    ],
    image: Annotated[
        Path | None,
        typer.Option("--image", "-i", help="Path to a single JPEG image."),
    ] = None,
    image_dir: Annotated[
        Path | None,
        typer.Option("--image-dir", "-d", help="Directory containing JPEG images."),
    ] = None,
    model_dir: Annotated[
        Path | None,
        typer.Option("--model-dir", help="Model directory (default: models/<model>)."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Output directory (default: ./output)."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Torch device: cpu, cuda:0, mps (default: auto-detect)."),
    ] = None,
    color_scheme: Annotated[
        ColorScheme,
        typer.Option("--color-scheme", help="Color palette: simplified or detailed."),
    ] = ColorScheme.SIMPLIFIED,
    overlay_alpha: Annotated[
        float,
        typer.Option("--overlay-alpha", help="Blend alpha for overlay (0.0-1.0)."),
    ] = OVERLAY_ALPHA,
    downsample: Annotated[
        int | None,
        typer.Option("--downsample", help="Downsample factor for color/overlay outputs."),
    ] = None,
    mpp: Annotated[
        float | None,
        typer.Option("--mpp", help="Override slide MPP instead of parsing from filename."),
    ] = None,
) -> None:
    """Run histopathology segmentation on TCGA images."""
    # The callback already converted `model` from str to Model at runtime
    resolved_model = cast(Model, model)

    # Validate mutually exclusive options
    if image is None and image_dir is None:
        console.print("[red]Error: Either --image or --image-dir is required.[/red]")
        raise typer.Exit(1)
    if image is not None and image_dir is not None:
        console.print("[red]Error: --image and --image-dir are mutually exclusive.[/red]")
        raise typer.Exit(1)

    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    resolved_model_dir = (
        Path(model_dir) if model_dir else script_dir / "models" / resolved_model.value
    )
    resolved_output_dir = Path(output_dir) if output_dir else script_dir / "output"

    # Collect image paths
    if image:
        image_paths = [Path(image).resolve()]
    else:
        assert image_dir is not None
        image_paths = sorted(Path(image_dir).glob("*.jpg"))
        if not image_paths:
            console.print(f"[red]Error: No .jpg files found in {image_dir}[/red]")
            raise typer.Exit(1)

    resolved_device = device or _detect_device()
    console.print(f"Using device: [cyan]{resolved_device}[/cyan]")

    config = InferenceConfig(
        model=resolved_model,
        model_dir=resolved_model_dir,
        image_paths=image_paths,
        output_dir=resolved_output_dir,
        device=resolved_device,
        color_scheme=color_scheme,
        overlay_alpha=overlay_alpha,
        downsample_factor=downsample,
        mpp_override=mpp,
    )

    run_inference(config)


if __name__ == "__main__":
    app()
