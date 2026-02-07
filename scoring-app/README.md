# Scoring App

FastAPI web application for manual side-by-side scoring of segmentation model outputs on histopathology images. Five model predictions are displayed simultaneously using zoomable image viewers (OpenSeadragon), and each can be rated on a 0–10 scale.

## Prerequisites

- Docker and Docker Compose
- Segmentation output images from [`segmentation-pipeline/`](../segmentation-pipeline/) placed in a single directory

The image directory should contain JPEG files following the naming convention produced by the inference pipeline:

```
{slide_name}_{mpp}_{tissue_type}_model{1-5}.jpg
```

## Setup

1. Copy the example environment file and set the path to your image directory:

    ```bash
    cp .env.example .env
    ```

    Edit `.env` and set `IMG_DIRECTORY` to the absolute path containing the model output images.

2. Start the application:

    ```bash
    docker compose up -d
    ```

3. Initialize the database by scanning the image directory:

    ```bash
    docker compose exec app python db_tools.py init
    ```

    This parses all image filenames, groups them by TCGA case and tissue type, creates project/sample records, and generates thumbnails.

4. Open http://localhost:8000 in a web browser.

## Usage

### Scoring workflow

1. Select a TCGA project from the overview.
2. Click a sample to open the scoring view.
3. Each model prediction is shown in a zoomable viewer. Rate each on a 0–10 scale using the buttons below the image, or use keyboard shortcuts.
4. Optionally add free-text comments per model.
5. Navigate between samples with the Previous/Next buttons or arrow keys.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| Arrow Left / Right | Previous / next sample |
| `T` / `N` | Switch to tumor / normal view |
| `0`–`9`, `ß` (=10) | Score all models at once (or focused model only) |
| `X` | Clear score |
| Arrow Up / Down | Increment / decrement score (when a model card is focused) |

### Exporting data

Scoring data can be exported as CSV from the project overview (per project or all projects). The CSV contains one row per model per tissue type per sample, with columns for scores and comments.

## Database

Sample and scoring data is stored in an SQLite database at `data/samples.db`. The database is persisted via a Docker volume mount. To reset:

```bash
docker compose exec app python db_tools.py clean
docker compose exec app python db_tools.py init
```

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Run the app locally without Docker:

```bash
uv sync
IMG_DIRECTORY=/path/to/images uv run fastapi dev app/main.py
```
