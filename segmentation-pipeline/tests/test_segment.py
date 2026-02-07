"""Tests for segment.py — preprocessing and postprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from segment import (
    _CUTOFF,
    _OVERLAP_MARGIN,
    DEFAULT_SLIDE_MPP,
    MODEL_MPP,
    MODEL_PATCH_SIZE,
    Model,
    _parse_model,
    compute_patch_size,
    create_color_map,
    extract_mpp_from_filename,
    pad_image,
    resize_mask_to_original,
    stitch_masks,
)

# -----------------------------------------------------------------------
# extract_mpp_from_filename
# -----------------------------------------------------------------------


class TestExtractMppFromFilename:
    """Tests for MPP extraction from TCGA filenames."""

    def test_standard_tumor_filename(self) -> None:
        filename = "TCGA-2J-AABR-01Z-00-DX1.31C68170_0.253_tumor_original.jpg"
        assert extract_mpp_from_filename(filename) == pytest.approx(0.253)

    def test_standard_normal_filename(self) -> None:
        filename = "TCGA-2J-AABR-01Z-00-DX1.31C68170_0.461_normal_original.jpg"
        assert extract_mpp_from_filename(filename) == pytest.approx(0.461)

    def test_no_mpp_returns_default(self) -> None:
        assert extract_mpp_from_filename("some_random_image.jpg") == DEFAULT_SLIDE_MPP

    def test_mpp_at_end_of_filename(self) -> None:
        filename = "slide_0.500_tumor_original.jpg"
        assert extract_mpp_from_filename(filename) == pytest.approx(0.5)


# -----------------------------------------------------------------------
# compute_patch_size
# -----------------------------------------------------------------------


class TestComputePatchSize:
    """Tests for patch size computation from slide MPP."""

    def test_mpp_equals_model_mpp(self) -> None:
        assert compute_patch_size(MODEL_MPP) == MODEL_PATCH_SIZE

    def test_mpp_half_of_model(self) -> None:
        assert compute_patch_size(MODEL_MPP / 2) == MODEL_PATCH_SIZE * 2

    def test_known_value(self) -> None:
        expected = int(MODEL_MPP / 0.461 * MODEL_PATCH_SIZE)
        assert compute_patch_size(0.461) == expected


# -----------------------------------------------------------------------
# pad_image
# -----------------------------------------------------------------------


class TestPadImage:
    """Tests for white-padding of images."""

    def test_padding_dimensions(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, orig_h, orig_w = pad_image(img, 64)
        assert padded.shape == (164, 264, 3)
        assert orig_h == 100
        assert orig_w == 200

    def test_padding_is_white(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        padded, _, _ = pad_image(img, 5)
        assert padded[10:, 10:, :].min() == 255

    def test_original_data_preserved(self) -> None:
        img = np.full((10, 10, 3), 42, dtype=np.uint8)
        padded, _, _ = pad_image(img, 5)
        np.testing.assert_array_equal(padded[:10, :10, :], img)


# -----------------------------------------------------------------------
# stitch_masks
# -----------------------------------------------------------------------


class TestStitchMasks:
    """Tests for mask stitching with overlap margins."""

    def test_single_patch(self) -> None:
        patch = np.ones((MODEL_PATCH_SIZE, MODEL_PATCH_SIZE), dtype=np.int8)
        result = stitch_masks([[patch]])
        assert result.shape == (_CUTOFF, _CUTOFF)

    def test_two_patches_horizontal(self) -> None:
        p1 = np.full((MODEL_PATCH_SIZE, MODEL_PATCH_SIZE), 1, dtype=np.int8)
        p2 = np.full((MODEL_PATCH_SIZE, MODEL_PATCH_SIZE), 2, dtype=np.int8)
        result = stitch_masks([[p1, p2]])
        expected_w = _CUTOFF + (_CUTOFF - _OVERLAP_MARGIN)
        assert result.shape == (_CUTOFF, expected_w)
        assert result[0, 0] == 1
        assert result[0, -1] == 2

    def test_2x2_grid(self) -> None:
        patches = [
            [
                np.full((MODEL_PATCH_SIZE, MODEL_PATCH_SIZE), i * 2 + j, dtype=np.int8)
                for j in range(2)
            ]
            for i in range(2)
        ]
        result = stitch_masks(patches)
        expected_h = _CUTOFF + (_CUTOFF - _OVERLAP_MARGIN)
        expected_w = _CUTOFF + (_CUTOFF - _OVERLAP_MARGIN)
        assert result.shape == (expected_h, expected_w)


# -----------------------------------------------------------------------
# create_color_map
# -----------------------------------------------------------------------


class TestCreateColorMap:
    """Tests for class-to-RGB color mapping."""

    def test_background_is_black(self) -> None:
        mask = np.zeros((2, 2), dtype=np.uint8)
        colors = [[255, 0, 0], [0, 255, 0]]
        result = create_color_map(mask, colors)
        np.testing.assert_array_equal(result, np.zeros((2, 2, 3), dtype=np.uint8))

    def test_class_1_gets_first_color(self) -> None:
        mask = np.ones((2, 2), dtype=np.uint8)
        colors = [[255, 0, 0], [0, 255, 0]]
        result = create_color_map(mask, colors)
        assert list(result[0, 0]) == [255, 0, 0]

    def test_mixed_classes(self) -> None:
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        # Need 3 entries: loop runs range(1, len(colors)), so class 2 needs colors[1]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        result = create_color_map(mask, colors)
        assert list(result[0, 1]) == [255, 0, 0]  # class 1 → colors[0]
        assert list(result[1, 0]) == [0, 255, 0]  # class 2 → colors[1]
        assert list(result[0, 0]) == [0, 0, 0]  # background


# -----------------------------------------------------------------------
# resize_mask_to_original
# -----------------------------------------------------------------------


class TestResizeMaskToOriginal:
    """Tests for mask resizing back to original dimensions."""

    def test_identity_resize(self) -> None:
        mask = np.ones((100, 100), dtype=np.int8)
        result = resize_mask_to_original(mask, MODEL_MPP, 80, 80)
        assert result.shape == (80, 80)
        assert result.dtype == np.uint8

    def test_upscale(self) -> None:
        mask = np.ones((10, 10), dtype=np.int8)
        result = resize_mask_to_original(mask, 0.5, 15, 15)
        assert result.shape == (15, 15)


# -----------------------------------------------------------------------
# _parse_model
# -----------------------------------------------------------------------


class TestParseModel:
    """Tests for model name string parsing."""

    def test_valid_names(self) -> None:
        for name in ["BREAST", "COLON", "LUNG", "KIDNEY", "PROSTATE", "DEMO"]:
            assert _parse_model(name) == Model[name]

    def test_case_insensitive(self) -> None:
        assert _parse_model("breast") == Model.BREAST
        assert _parse_model("Lung") == Model.LUNG

    def test_none_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter):
            _parse_model(None)

    def test_invalid_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter):
            _parse_model("NONEXISTENT")
