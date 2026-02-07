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

ROIS_RECORD_ID="XXXXXXX"  # TODO: Update after Zenodo upload
ROIS_FILENAME="cross-cancer-segmentation-rois.zip"  # TODO: Update after Zenodo upload
ROIS_TARGET_DIR="${SCRIPT_DIR}/rois"

MASKS_RECORD_ID="XXXXXXX"  # TODO: Update after Zenodo upload
MASKS_FILENAME="cross-cancer-segmentation-masks.zip"  # TODO: Update after Zenodo upload
MASKS_TARGET_DIR="${SCRIPT_DIR}/masks"

download_and_extract() {
    local record_id="$1"
    local filename="$2"
    local target_dir="$3"
    local label="$4"

    if [ "$record_id" = "XXXXXXX" ]; then
        echo "Skipping ${label}: Zenodo record ID not yet configured."
        return 1
    fi

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
    download_and_extract "$ROIS_RECORD_ID" "$ROIS_FILENAME" "$ROIS_TARGET_DIR" "Tissue ROIs"
fi

if $DOWNLOAD_MASKS; then
    download_and_extract "$MASKS_RECORD_ID" "$MASKS_FILENAME" "$MASKS_TARGET_DIR" "Segmentation masks"
fi
