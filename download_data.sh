#!/usr/bin/env bash
# Download research data from Zenodo.
#
# Usage:
#   ./download_data.sh           Download evaluation data (default, required for analysis notebooks)
#   ./download_data.sh --rois    Download tissue ROIs
#   ./download_data.sh --masks   Download segmentation masks
#   ./download_data.sh --all     Download all datasets
#
# Each dataset is extracted into its own directory at the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Zenodo record IDs and filenames ---
EVAL_RECORD_ID="18518811"
EVAL_FILENAME="cross-cancer-segmentation-eval-data.zip"
EVAL_TARGET_DIR="${SCRIPT_DIR}/data"

ROIS_RECORD_ID="18668580"
ROIS_TARGET_DIR="${SCRIPT_DIR}/rois"
ROIS_COHORTS=(
    TCGA-BLCA TCGA-BRCA TCGA-CESC TCGA-CHOL TCGA-COADREAD
    TCGA-ESCA TCGA-HNSC TCGA-KICH TCGA-KIRC TCGA-KIRP
    TCGA-LIHC TCGA-LUAD TCGA-LUSC TCGA-MESO TCGA-OV
    TCGA-PAAD TCGA-PRAD TCGA-SKCM TCGA-STAD TCGA-THCA
    TCGA-UCEC
)

MASKS_RECORD_ID="18669667"
MASKS_TARGET_DIR="${SCRIPT_DIR}/masks"
MASKS_COHORTS=(
    TCGA-BLCA TCGA-BRCA TCGA-CESC TCGA-CHOL TCGA-COADREAD
    TCGA-ESCA TCGA-HNSC TCGA-KICH TCGA-KIRC TCGA-KIRP
    TCGA-LIHC TCGA-LUAD TCGA-LUSC TCGA-MESO TCGA-OV
    TCGA-PAAD TCGA-PRAD TCGA-SKCM TCGA-STAD TCGA-THCA
    TCGA-UCEC
)

download_and_extract() {
    local record_id="$1"
    local filename="$2"
    local target_dir="$3"
    local label="$4"

    if [ -d "$target_dir" ]; then
        echo "Skipping ${label}: $(basename "$target_dir")/ already exists. Remove it first to re-download."
        return 1
    fi

    local url="https://zenodo.org/records/${record_id}/files/${filename}"
    local zip_path="${SCRIPT_DIR}/${filename}"

    echo "Downloading ${label} from Zenodo (record ${record_id})..."
    curl -L --fail --progress-bar "$url" -o "$zip_path"

    echo "Extracting..."
    unzip -q "$zip_path" -d "$SCRIPT_DIR"
    rm "$zip_path"

    echo "Done. ${label} extracted to ${target_dir}/"
}

download_tar_cohorts() {
    local record_id="$1"
    local target_dir="$2"
    local label="$3"
    local -n cohorts=$4

    if [ -d "$target_dir" ]; then
        echo "Skipping ${label}: $(basename "$target_dir")/ already exists. Remove it first to re-download."
        return 1
    fi

    echo "Downloading ${label} from Zenodo (record ${record_id})..."
    echo ""

    mkdir -p "$target_dir"

    local total=${#cohorts[@]}
    local index=0

    for cohort in "${cohorts[@]}"; do
        index=$((index + 1))
        local filename="${cohort}.tar"
        local url="https://zenodo.org/records/${record_id}/files/${filename}"
        local tar_path="${SCRIPT_DIR}/${filename}"

        echo "[${index}/${total}] Downloading ${filename}..."
        curl -L --fail --progress-bar "$url" -o "$tar_path"

        echo "Extracting..."
        tar xf "$tar_path" -C "$target_dir"
        rm "$tar_path"
    done

    echo "Done. ${label} extracted to ${target_dir}/"
}

# Parse arguments
DOWNLOAD_EVAL=false
DOWNLOAD_ROIS=false
DOWNLOAD_MASKS=false

if [ $# -eq 0 ]; then
    DOWNLOAD_EVAL=true
else
    for arg in "$@"; do
        case "$arg" in
            --all)
                DOWNLOAD_EVAL=true
                DOWNLOAD_ROIS=true
                DOWNLOAD_MASKS=true
                ;;
            --rois)
                DOWNLOAD_ROIS=true
                ;;
            --masks)
                DOWNLOAD_MASKS=true
                ;;
            *)
                echo "Unknown option: $arg"
                echo "Usage: $0 [--rois] [--masks] [--all]"
                exit 1
                ;;
        esac
    done
fi

if $DOWNLOAD_EVAL; then
    download_and_extract "$EVAL_RECORD_ID" "$EVAL_FILENAME" "$EVAL_TARGET_DIR" "Evaluation data"
fi

if $DOWNLOAD_ROIS; then
    download_tar_cohorts "$ROIS_RECORD_ID" "$ROIS_TARGET_DIR" "Tumor ROIs" ROIS_COHORTS
fi

if $DOWNLOAD_MASKS; then
    download_tar_cohorts "$MASKS_RECORD_ID" "$MASKS_TARGET_DIR" "Segmentation masks" MASKS_COHORTS
fi
