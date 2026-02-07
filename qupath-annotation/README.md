# Helper Scripts

QuPath Groovy scripts used to create tissue region annotations on TCGA whole-slide images and export them as JPEG images for downstream model inference.

These scripts are intended to be run inside [QuPath v0.5.0](https://qupath.github.io/) via **Automate > Script Editor**.

## Scripts

### `add_annotation_objects.groovy`

Creates two rectangular annotation objects on the currently open image:

- **Tumor** (12,500 x 10,000 px) — centered on the image
- **Normal** (12,500 x 10,000 px) — placed to the right of the tumor annotation with a 1,000 px gap

The annotations are saved to the QuPath project automatically. They are meant to be **manually adjusted** (resized/repositioned) to accurately cover the desired tissue regions before running the export script.

### `export_annotations_as_images.groovy`

Exports the image regions covered by "Tumor" and "Normal" annotations as JPEG files with 5x downsampling. Output files are written to an `image_export/` directory within the QuPath project folder.

Output filename format:

```
{imageName}_{mpp}_{className}_original.jpg
```

- `imageName` — slide name without file extension
- `mpp` — microns per pixel (resolution), rounded to 3 decimal places
- `className` — `tumor` or `normal`

### `export_ground_truth_class_maps.groovy`

Exports ground truth annotations as labeled PNG class maps for Dice coefficient analysis. Rectangular bounding boxes with class `EXPORT_ROI` define the export regions. Each annotation within a bounding box is mapped to a numeric class index and written as a single-channel PNG.

The current label mapping is **CRC-specific** (TUMOR=1, TU_STROMA=2, MUC=3, etc.) and must be adapted for other cohorts by editing the `LabeledImageServer.Builder` section.

Output filename format:

```
{imageName}_class_map_ground_truth_roi_{N}.png
```

Output is written to a `ground_truth_class_maps/` directory within the QuPath project folder.

## Usage

1. Open a QuPath project containing TCGA whole-slide images.
2. Select an image in the project.
3. Run `add_annotation_objects.groovy` to create initial annotation rectangles.
4. Manually adjust the annotations to cover the desired tissue regions.
5. Run `export_annotations_as_images.groovy` to export the annotated regions.
6. Exported images will be in the `image_export/` directory of your QuPath project.

To batch-process multiple images, use **Run > Run for project** in the Script Editor.
