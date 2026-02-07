#!/bin/bash
#
# Full Dice analysis pipeline for all cohorts and models.
#
# Expects the following data directory layout (created by QuPath export):
#
#   <BASE_DIR>/
#   ├── 01_LUNG_COHORT/
#   │   ├── 01_ROI_GT_CLASS_MAPS/      (ground truth PNGs from QuPath)
#   │   └── 02_ROI_IMAGES/             (ROI image JPEGs from QuPath)
#   ├── 02_PRAD_COHORT/
#   ├── 03_BRCA_COHORT/
#   └── 04_COADREAD_COHORT/
#
# For each (cohort, model) pair the script:
#   1. Runs segment.py inference on the ROI images
#   2. Runs dice_analysis.py to compute Dice scores
#
# Usage:
#   chmod +x run_dice_pipeline.sh
#   ./run_dice_pipeline.sh /path/to/data

set -euo pipefail

BASE_DIR="${1:?Usage: $0 <data_dir>}"

# Cohort names match the Cohort enum in cohort_config.py
COHORTS=("LUNG"     "PROSTATE" "BREAST"       "COADREAD")
# Directory names match the R code's expected pattern ({NN}_{NAME}_COHORT)
COHORT_DIRS=("01_LUNG_COHORT" "02_PRAD_COHORT" "03_BRCA_COHORT" "04_COADREAD_COHORT")

# Model names match the ModelKey enum in cohort_config.py
MODELS=("BREAST"     "COLON"     "LUNG"       "KIDNEY"       "PROSTATE")
# Result directory names match the R code's expected pattern ({NAME}_MODEL)
MODEL_RESULT_DIRS=("BRCA_MODEL" "CRC_MODEL" "LUNG_MODEL" "KIDNEY_MODEL" "PROSTATE_MODEL")

echo "========================================="
echo "Dice Analysis Pipeline"
echo "Data directory: $BASE_DIR"
echo "Cohorts: ${COHORTS[*]}"
echo "Models:  ${MODELS[*]}"
echo "========================================="

for i in "${!COHORTS[@]}"; do
  cohort="${COHORTS[$i]}"
  cohort_dir="${COHORT_DIRS[$i]}"

  roi_dir="$BASE_DIR/$cohort_dir/02_ROI_IMAGES"
  gt_dir="$BASE_DIR/$cohort_dir/01_ROI_GT_CLASS_MAPS"

  if [ ! -d "$gt_dir" ]; then
    echo "WARNING: Ground truth directory not found: $gt_dir — skipping $cohort"
    continue
  fi

  for j in "${!MODELS[@]}"; do
    model="${MODELS[$j]}"
    model_result_dir="${MODEL_RESULT_DIRS[$j]}"

    pred_dir="$BASE_DIR/$cohort_dir/03_ROI_MODEL_INFERENCE_OUTPUT/$model_result_dir"
    out_dir="$BASE_DIR/$cohort_dir/04_ANALYSIS_RESULTS/$model_result_dir"

    echo ""
    echo "----- $cohort / $model -----"

    # Step 1: Run inference (only if ROI images directory exists)
    if [ -d "$roi_dir" ]; then
      echo "  [1/2] Running inference..."
      uv run segment.py --model "$model" --image-dir "$roi_dir" --output-dir "$pred_dir"
    else
      echo "  [1/2] Skipping inference (no ROI images at $roi_dir)"
    fi

    # Step 2: Dice analysis (only if predictions exist)
    if [ -d "$pred_dir" ]; then
      echo "  [2/2] Computing Dice scores..."
      uv run dice_analysis.py \
        --gt-dir "$gt_dir" \
        --pred-dir "$pred_dir" \
        --model "$model" \
        --cohort "$cohort" \
        --output-dir "$out_dir"
    else
      echo "  [2/2] Skipping Dice analysis (no predictions at $pred_dir)"
    fi
  done
done

echo ""
echo "========================================="
echo "Pipeline complete."
echo "========================================="
