# Segmentation Inference

Patch-based semantic segmentation of histopathology tissue regions from TCGA whole-slide images using models trained with [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch).

## Quick Start

```bash
cd segmentation-pipeline
uv sync
uv run create_demo_model.py
uv run segment.py \
    --model DEMO \
    --image test_image/TCGA-2J-AABR-01Z-00-DX1.31C68170-831C-4A83-8CDF-9A07ED57A1AC_0.253_tumor_original.jpg
```

The trained model weights are not included in this repository.
`create_demo_model.py` generates a model with the correct architecture but
random (untrained) weights so the pipeline can be tested end-to-end.
Predictions will be random noise. The included test image is a 1000x800 px
center crop from a full-size TCGA whole-slide image export at its original
resolution (0.253 MPP).

## Available Models

Models are named after the organ they were trained on but can be applied
cross-entity (e.g. the LUNG model on head & neck SCC).
The trained weights are not provided in this repository.

| Model      | Classes |
|------------|---------|
| `BREAST`   | 13      |
| `COLON`    | 11      |
| `LUNG`     | 12      |
| `KIDNEY`   | 10      |
| `PROSTATE` | 4       |
| `DEMO`     | 12      |

## Usage

```
uv run segment.py --model NAME --image PATH [options]
uv run segment.py --model NAME --image-dir DIR [options]
```

### Options

| Flag              | Default              | Description                                       |
|-------------------|----------------------|---------------------------------------------------|
| `--model`, `-m`   | *(required)*         | `BREAST`, `COLON`, `LUNG`, `KIDNEY`, `PROSTATE`, or `DEMO` |
| `--image`, `-i`   | -                    | Single image path (mutually exclusive with --image-dir) |
| `--image-dir`, `-d` | -                  | Directory of `.jpg` images                        |
| `--model-dir`     | `models/<auto>`      | Override model directory                          |
| `--output-dir`, `-o` | `output/`          | Where to write results                            |
| `--device`        | auto                 | `cpu`, `cuda:0`, `mps`                            |
| `--color-scheme`  | `simplified`         | Color palette: `simplified` or `detailed`         |
| `--overlay-alpha` | `0.3`                | Mask blend alpha                                  |
| `--downsample`    | *(none)*             | Downsample factor for color/overlay outputs       |
| `--mpp`           | *(from filename)*    | Override microns-per-pixel                        |

### Outputs

For each input image `<stem>.jpg`, three files are written to `--output-dir`:

| File                  | Description                               |
|-----------------------|-------------------------------------------|
| `<stem>_mask.png`     | Grayscale PNG, pixel values = class index |
| `<stem>_color.png`    | Color-mapped segmentation                 |
| `<stem>_overlay.jpg`  | Original image with color mask blended    |

## Dice Coefficient Analysis

`dice_analysis.py` compares model predictions against ground truth annotations
by computing per-ROI [Dice coefficients](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).
Both masks are first collapsed to three classes — Background (0), Tumor (1),
and Tumor_Stroma (2) — so that models with different class vocabularies can be
compared on a common scale. The class remapping rules for each (cohort, model)
pair are defined in `cohort_config.py`.

### Data layout

The pipeline expects ground truth class maps exported from QuPath (see
`qupath-annotation/export_ground_truth_class_maps.groovy`) and prediction masks
produced by `segment.py`, organized by cohort:

```
<data_dir>/
├── 01_LUNG_COHORT/
│   ├── 01_ROI_GT_CLASS_MAPS/          Ground truth PNGs from QuPath
│   ├── 02_ROI_IMAGES/                 ROI image JPEGs from QuPath
│   ├── 03_ROI_MODEL_INFERENCE_OUTPUT/ Prediction masks from segment.py
│   │   ├── BRCA_MODEL/
│   │   ├── CRC_MODEL/
│   │   ├── LUNG_MODEL/
│   │   ├── KIDNEY_MODEL/
│   │   └── PROSTATE_MODEL/
│   └── 04_ANALYSIS_RESULTS/           Dice result CSVs
│       └── ...
├── 02_PRAD_COHORT/
├── 03_BRCA_COHORT/
└── 04_COADREAD_COHORT/
```

### File naming conventions

Ground truth and prediction files are matched by **slide name** and
**ROI number**. Both must follow specific naming patterns:

| File type   | Pattern | Example |
|-------------|---------|---------|
| Ground truth | `{slide}_class_map_ground_truth_roi_{N}.png` | `TCGA-A6-2671-…_class_map_ground_truth_roi_1.png` |
| Prediction   | `{slide}_…_roi_{N}_…_mask.png` | `TCGA-A6-2671-…_roi_1_0.500_tumor_model2_mask.png` |

The slide name must appear as a substring in the prediction filename, and
`roi_{N}` must be present in both.

### Running a single analysis

```bash
uv run dice_analysis.py \
    --gt-dir  /path/to/01_ROI_GT_CLASS_MAPS \
    --pred-dir /path/to/03_ROI_MODEL_INFERENCE_OUTPUT/CRC_MODEL \
    --model COLON --cohort COADREAD \
    --output-dir /path/to/04_ANALYSIS_RESULTS/CRC_MODEL
```

| Flag | Description |
|------|-------------|
| `--gt-dir` | Directory containing ground truth class map PNGs |
| `--pred-dir` | Directory containing `segment.py` `*_mask.png` outputs |
| `--model`, `-m` | Model that produced the predictions (`BREAST`, `COLON`, `LUNG`, `KIDNEY`, `PROSTATE`) |
| `--cohort`, `-c` | Evaluation cohort (`BRCA`, `COADREAD`, `LUNG`, `PRAD`) |
| `--output-dir`, `-o` | Where to write the result CSV (default: `dice_results/`) |

### Running the full pipeline

`run_dice_pipeline.sh` automates inference and Dice analysis for all 20
cohort × model combinations:

```bash
./run_dice_pipeline.sh /path/to/data_dir
```

For each (cohort, model) pair it runs `segment.py` on the ROI images, then
`dice_analysis.py` to compute Dice scores.

### Output CSV

Each run produces `detailed_dice_results.csv` with one row per ROI:

| Column | Description |
|--------|-------------|
| `Slide` | Slide name |
| `ROI` | ROI number |
| `Overall_DICE` | Mean Dice across non-background classes present in GT |
| `Non_Background_Percentage` | % of GT pixels that are non-background |
| `DICE_Tumor` | Dice coefficient for class 1 (Tumor) |
| `GT_Percent_Tumor` | % of GT pixels classified as Tumor |
| `Pred_Percent_Tumor` | % of prediction pixels classified as Tumor |
| `DICE_Tumor_Stroma` | Dice coefficient for class 2 (Tumor_Stroma) |
| `GT_Percent_Tumor_Stroma` | % of GT pixels classified as Tumor_Stroma |
| `Pred_Percent_Tumor_Stroma` | % of prediction pixels classified as Tumor_Stroma |

This schema matches what the R analysis in `scoring-analysis/` expects.

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Segmentation Algorithm

1. Load image, extract MPP (microns per pixel) from TCGA filename
2. Pad image with white to ensure complete patch coverage
3. Extract overlapping patches (75% stride = 25% overlap at slide resolution)
4. Resize each patch to 512 x 512 px, run through the segmentation model
5. Stitch predictions by taking the center region of each patch, discarding overlap margins
6. Resize stitched mask back to original image dimensions (nearest-neighbor)
7. Generate grayscale mask, color-mapped segmentation, and overlay
