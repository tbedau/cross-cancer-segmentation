"""Dice coefficient analysis comparing model predictions against ground truth.

Reads ``_mask.png`` files produced by ``segment.py``, applies a cohort-specific
class remapping (collapsing model classes to Tumor / Tumor_Stroma / Background),
and computes per-ROI Dice coefficients.  The output CSV matches the schema
expected by the R analysis in ``scoring-analysis/figures-for-pub.qmd``.

Usage::

    uv run dice_analysis.py \\
        --gt-dir  /path/to/ground_truth_class_maps \\
        --pred-dir /path/to/segment_py_output \\
        --model COLON --cohort COADREAD \\
        --output-dir /path/to/results

Ground truth PNGs must follow the naming convention produced by the QuPath
export script: ``{slide}_class_map_ground_truth_roi_{N}.png``.

Prediction masks must follow segment.py's naming: ``{stem}_mask.png`` where
*stem* contains ``roi_{N}`` and the slide name.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from cohort_config import COHORT_SPECS, Cohort, ModelKey

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Image.MAX_IMAGE_PIXELS = None  # Some ROIs exceed the default safety limit

console = Console()
app = typer.Typer(
    help="Compute Dice coefficients between ground truth and model predictions.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# CSV column names (must match what the R code expects after janitor::clean_names)
CSV_COLUMNS = [
    "Slide",
    "ROI",
    "Overall_DICE",
    "Non_Background_Percentage",
    "DICE_Tumor",
    "GT_Percent_Tumor",
    "Pred_Percent_Tumor",
    "DICE_Tumor_Stroma",
    "GT_Percent_Tumor_Stroma",
    "Pred_Percent_Tumor_Stroma",
]


# ---------------------------------------------------------------------------
# Dice calculation
# ---------------------------------------------------------------------------


def dice_coefficient(
    y_true: NDArray[np.uint8],
    y_pred: NDArray[np.uint8],
    class_label: int,
) -> float:
    """Compute the Dice coefficient for a single class.

    Returns 1.0 when the class is absent from both masks.
    """
    gt = (y_true == class_label).astype(np.int32)
    pred = (y_pred == class_label).astype(np.int32)

    gt_sum = int(np.sum(gt))
    pred_sum = int(np.sum(pred))

    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0

    intersection = int(np.sum(gt * pred))
    return (2.0 * intersection) / (gt_sum + pred_sum)


# ---------------------------------------------------------------------------
# File matching
# ---------------------------------------------------------------------------


def _find_gt_files(gt_dir: Path) -> dict[tuple[str, int], Path]:
    """Build a lookup of (slide_name, roi_number) → ground truth path."""
    pattern = re.compile(r"^(.+)_class_map_ground_truth_roi_(\d+)\.png$")
    result: dict[tuple[str, int], Path] = {}
    for p in sorted(gt_dir.glob("*.png")):
        m = pattern.match(p.name)
        if m:
            result[(m.group(1), int(m.group(2)))] = p
    return result


def _find_pred_files(pred_dir: Path) -> dict[tuple[str, int], Path]:
    """Build a lookup of (slide_name_fragment, roi_number) → prediction mask path.

    segment.py names masks ``{base}_roi_{N}_model{M}_mask.png`` where *base*
    is the slide name plus MPP.  We extract the ROI number and a slide
    fragment (everything before the MPP pattern ``_<float>_``) for matching.
    """
    roi_pattern = re.compile(r"roi_(\d+)")
    result: dict[tuple[str, int], Path] = {}
    for p in sorted(pred_dir.glob("*_mask.png")):
        roi_match = roi_pattern.search(p.name)
        if roi_match:
            roi_num = int(roi_match.group(1))
            # The full stem minus the trailing _model{M}_mask suffix
            stem = p.name
            result[(stem, roi_num)] = p
    return result


def match_files(
    gt_dir: Path,
    pred_dir: Path,
) -> list[tuple[str, int, Path, Path]]:
    """Match ground truth and prediction files by slide name and ROI number.

    Returns a list of ``(slide_name, roi_number, gt_path, pred_path)`` tuples.
    """
    gt_lookup = _find_gt_files(gt_dir)
    pred_files = list(pred_dir.glob("*_mask.png"))

    roi_pattern = re.compile(r"roi_(\d+)")
    matched: list[tuple[str, int, Path, Path]] = []

    for (slide_name, roi_num), gt_path in gt_lookup.items():
        for pred_path in pred_files:
            pred_roi = roi_pattern.search(pred_path.name)
            if pred_roi and int(pred_roi.group(1)) == roi_num and slide_name in pred_path.name:
                matched.append((slide_name, roi_num, gt_path, pred_path))
                break

    return sorted(matched, key=lambda t: (t[0], t[1]))


# ---------------------------------------------------------------------------
# Per-ROI analysis
# ---------------------------------------------------------------------------


def _apply_remap(mask: NDArray[np.uint8], class_map: dict[int, int]) -> NDArray[np.uint8]:
    """Remap class indices in *mask* according to *class_map*.

    Unmapped values default to 0 (background).
    """
    max_val = int(mask.max())
    lut = np.zeros(max_val + 1, dtype=np.uint8)
    for src, dst in class_map.items():
        if src <= max_val:
            lut[src] = dst
    return lut[mask]


def _simplify_gt(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Collapse ground truth classes 3-8 to background (0).

    The QuPath export may produce up to 9 classes (0-8).  For the v2 Dice
    analysis we only keep class 1 (Tumor) and class 2 (Tumor_Stroma).
    """
    return np.where(mask <= 2, mask, np.uint8(0))


def analyze_roi(
    gt_path: Path,
    pred_path: Path,
    class_map: dict[int, int],
) -> dict[str, float]:
    """Compute Dice metrics for a single ROI pair.

    Returns a dict with keys matching :data:`CSV_COLUMNS` (minus Slide/ROI).
    """
    gt_raw = np.array(Image.open(gt_path))
    pred_raw = np.array(Image.open(pred_path))

    # Ensure shapes match (resize prediction to GT dimensions if needed)
    if gt_raw.shape != pred_raw.shape:
        pred_img = Image.open(pred_path).resize(
            (gt_raw.shape[1], gt_raw.shape[0]),
            Image.Resampling.NEAREST,
        )
        pred_raw = np.array(pred_img)

    # Remap both to simplified {0, 1, 2}
    gt = _simplify_gt(gt_raw)
    pred = _apply_remap(pred_raw, class_map)

    total_pixels = gt.size

    # Non-background percentage (in ground truth)
    non_bg = float(np.sum(gt > 0) / total_pixels * 100)

    # Per-class metrics
    dice_tumor = dice_coefficient(gt, pred, 1)
    dice_stroma = dice_coefficient(gt, pred, 2)

    gt_pct_tumor = float(np.sum(gt == 1) / total_pixels * 100)
    pred_pct_tumor = float(np.sum(pred == 1) / total_pixels * 100)
    gt_pct_stroma = float(np.sum(gt == 2) / total_pixels * 100)
    pred_pct_stroma = float(np.sum(pred == 2) / total_pixels * 100)

    # Overall Dice: mean over non-background classes present in GT
    class_scores: list[float] = []
    if gt_pct_tumor > 0:
        class_scores.append(dice_tumor)
    if gt_pct_stroma > 0:
        class_scores.append(dice_stroma)
    overall_dice = sum(class_scores) / len(class_scores) if class_scores else 0.0

    return {
        "Overall_DICE": overall_dice,
        "Non_Background_Percentage": non_bg,
        "DICE_Tumor": dice_tumor,
        "GT_Percent_Tumor": gt_pct_tumor,
        "Pred_Percent_Tumor": pred_pct_tumor,
        "DICE_Tumor_Stroma": dice_stroma,
        "GT_Percent_Tumor_Stroma": gt_pct_stroma,
        "Pred_Percent_Tumor_Stroma": pred_pct_stroma,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    gt_dir: Annotated[
        Path,
        typer.Option("--gt-dir", help="Directory with ground truth class map PNGs."),
    ],
    pred_dir: Annotated[
        Path,
        typer.Option("--pred-dir", help="Directory with segment.py *_mask.png outputs."),
    ],
    model: Annotated[
        ModelKey,
        typer.Option("--model", "-m", help="Which model produced the predictions."),
    ],
    cohort: Annotated[
        Cohort,
        typer.Option("--cohort", "-c", help="Evaluation cohort (determines class remapping)."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for result CSVs."),
    ] = Path("dice_results"),
) -> None:
    """Compute Dice coefficients between ground truth and model predictions."""
    # Resolve class remapping
    spec = COHORT_SPECS[cohort]
    class_map = spec.model_maps[model]

    console.print(
        f"Dice analysis: model=[bold]{model.value}[/bold], "
        f"cohort=[bold]{cohort.value}[/bold] ({spec.dir_name})"
    )

    # Match files
    matched = match_files(gt_dir, pred_dir)
    if not matched:
        console.print("[red]No matched file pairs found.[/red] Check directory paths and naming.")
        raise typer.Exit(code=1)

    console.print(f"Found [bold]{len(matched)}[/bold] matched ROI pairs.")

    # Analyze each pair
    rows: list[dict[str, str | float]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing ROIs...", total=len(matched))
        for slide_name, roi_num, gt_path, pred_path in matched:
            metrics = analyze_roi(gt_path, pred_path, class_map)
            rows.append({"Slide": slide_name, "ROI": roi_num, **metrics})
            progress.advance(task)

    # Write CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "detailed_dice_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"Results saved to [bold]{csv_path}[/bold]")

    # Print summary
    if rows:
        mean_dice = sum(r["Overall_DICE"] for r in rows) / len(rows)  # type: ignore[arg-type]
        console.print(f"Mean Overall Dice: [bold]{mean_dice:.4f}[/bold] (n={len(rows)})")


if __name__ == "__main__":
    app()
