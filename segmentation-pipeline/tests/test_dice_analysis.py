"""Tests for dice_analysis.py — core Dice coefficient computation and file matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dice_analysis import (
    _apply_remap,
    _find_gt_files,
    _find_pred_files,
    _simplify_gt,
    analyze_roi,
    dice_coefficient,
    match_files,
)

# -----------------------------------------------------------------------
# dice_coefficient
# -----------------------------------------------------------------------


class TestDiceCoefficient:
    """Tests for the Dice similarity coefficient calculation."""

    def test_perfect_match(self) -> None:
        mask = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        assert dice_coefficient(mask, mask, class_label=1) == 1.0

    def test_no_overlap(self) -> None:
        gt = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        pred = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        assert dice_coefficient(gt, pred, class_label=1) == 0.0

    def test_absent_from_both(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.uint8)
        assert dice_coefficient(gt, pred, class_label=1) == 1.0

    def test_absent_from_gt_only(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.ones((4, 4), dtype=np.uint8)
        assert dice_coefficient(gt, pred, class_label=1) == 0.0

    def test_absent_from_pred_only(self) -> None:
        gt = np.ones((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.uint8)
        assert dice_coefficient(gt, pred, class_label=1) == 0.0

    def test_partial_overlap(self) -> None:
        gt = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        pred = np.array([[0, 1, 1, 0]], dtype=np.uint8)
        # intersection=1, gt_sum=2, pred_sum=2 → Dice = 2*1/4 = 0.5
        assert dice_coefficient(gt, pred, class_label=1) == pytest.approx(0.5)

    def test_multiclass_isolation(self) -> None:
        gt = np.array([[1, 2, 2], [0, 0, 0]], dtype=np.uint8)
        pred = np.array([[1, 2, 0], [0, 0, 2]], dtype=np.uint8)
        # For class 2: gt=2px, pred=2px, intersection=1 → Dice = 2*1/4 = 0.5
        assert dice_coefficient(gt, pred, class_label=2) == pytest.approx(0.5)


# -----------------------------------------------------------------------
# _apply_remap
# -----------------------------------------------------------------------


class TestApplyRemap:
    """Tests for the LUT-based class remapping."""

    def test_identity_map(self) -> None:
        mask = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        result = _apply_remap(mask, {0: 0, 1: 1, 2: 2})
        np.testing.assert_array_equal(result, mask)

    def test_collapse_classes(self) -> None:
        mask = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
        class_map = {0: 0, 1: 1, 2: 2, 3: 1, 4: 1}
        expected = np.array([[0, 1, 2, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(_apply_remap(mask, class_map), expected)

    def test_unmapped_values_become_zero(self) -> None:
        mask = np.array([[5, 6, 7]], dtype=np.uint8)
        class_map = {0: 0, 1: 1}
        expected = np.array([[0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(_apply_remap(mask, class_map), expected)

    def test_realistic_map_targets_012(self) -> None:
        from cohort_config import COHORT_SPECS, Cohort, ModelKey

        class_map = COHORT_SPECS[Cohort.BREAST].model_maps[ModelKey.BREAST]
        mask = np.arange(14, dtype=np.uint8).reshape(2, 7)
        result = _apply_remap(mask, class_map)
        assert set(np.unique(result)).issubset({0, 1, 2})


# -----------------------------------------------------------------------
# _simplify_gt
# -----------------------------------------------------------------------


class TestSimplifyGt:
    """Tests for ground truth simplification (collapse classes 3-8 to 0)."""

    def test_keeps_012(self) -> None:
        mask = np.array([[0, 1, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(_simplify_gt(mask), mask)

    def test_collapses_higher_classes(self) -> None:
        mask = np.array([[3, 4, 5, 6, 7, 8]], dtype=np.uint8)
        expected = np.zeros((1, 6), dtype=np.uint8)
        np.testing.assert_array_equal(_simplify_gt(mask), expected)

    def test_mixed(self) -> None:
        mask = np.arange(9, dtype=np.uint8).reshape(1, 9)
        expected = np.array([[0, 1, 2, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(_simplify_gt(mask), expected)


# -----------------------------------------------------------------------
# _find_gt_files / _find_pred_files
# -----------------------------------------------------------------------


class TestFindGtFiles:
    """Tests for ground truth file discovery."""

    def test_finds_matching_files(self, tmp_path: Path) -> None:
        for name in [
            "TCGA-AA-AAAA_class_map_ground_truth_roi_1.png",
            "TCGA-AA-AAAA_class_map_ground_truth_roi_2.png",
        ]:
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / name)
        # Non-matching file
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "other.png")

        result = _find_gt_files(tmp_path)
        assert len(result) == 2
        assert ("TCGA-AA-AAAA", 1) in result
        assert ("TCGA-AA-AAAA", 2) in result

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert _find_gt_files(tmp_path) == {}


class TestFindPredFiles:
    """Tests for prediction mask file discovery."""

    def test_finds_mask_files(self, tmp_path: Path) -> None:
        name = "TCGA-AA-AAAA_0.461_tumor_roi_1_model1_mask.png"
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / name)

        result = _find_pred_files(tmp_path)
        assert len(result) == 1
        key = next(iter(result.keys()))
        assert key[1] == 1

    def test_ignores_non_mask_files(self, tmp_path: Path) -> None:
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "something_color.png")
        assert _find_pred_files(tmp_path) == {}


# -----------------------------------------------------------------------
# match_files
# -----------------------------------------------------------------------


class TestMatchFiles:
    """Tests for GT-to-prediction file matching."""

    def test_matches_by_slide_and_roi(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        slide = "TCGA-AA-AAAA"
        img = np.zeros((4, 4), dtype=np.uint8)
        Image.fromarray(img).save(gt_dir / f"{slide}_class_map_ground_truth_roi_1.png")
        Image.fromarray(img).save(pred_dir / f"{slide}_0.461_tumor_roi_1_model1_mask.png")

        matched = match_files(gt_dir, pred_dir)
        assert len(matched) == 1
        assert matched[0][0] == slide
        assert matched[0][1] == 1

    def test_no_match_different_roi(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        slide = "TCGA-AA-AAAA"
        img = np.zeros((4, 4), dtype=np.uint8)
        Image.fromarray(img).save(gt_dir / f"{slide}_class_map_ground_truth_roi_1.png")
        Image.fromarray(img).save(pred_dir / f"{slide}_0.461_tumor_roi_2_model1_mask.png")

        assert match_files(gt_dir, pred_dir) == []


# -----------------------------------------------------------------------
# analyze_roi
# -----------------------------------------------------------------------


class TestAnalyzeRoi:
    """Integration test for the full ROI analysis pipeline."""

    def test_perfect_prediction(self, tmp_path: Path) -> None:
        gt_mask = np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        pred_mask = gt_mask.copy()

        gt_path = tmp_path / "gt.png"
        pred_path = tmp_path / "pred.png"
        Image.fromarray(gt_mask).save(gt_path)
        Image.fromarray(pred_mask).save(pred_path)

        result = analyze_roi(gt_path, pred_path, {0: 0, 1: 1, 2: 2})
        assert result["DICE_Tumor"] == pytest.approx(1.0)
        assert result["DICE_Tumor_Stroma"] == pytest.approx(1.0)
        assert result["Overall_DICE"] == pytest.approx(1.0)

    def test_all_background(self, tmp_path: Path) -> None:
        mask = np.zeros((4, 4), dtype=np.uint8)
        gt_path = tmp_path / "gt.png"
        pred_path = tmp_path / "pred.png"
        Image.fromarray(mask).save(gt_path)
        Image.fromarray(mask).save(pred_path)

        result = analyze_roi(gt_path, pred_path, {0: 0})
        assert result["Non_Background_Percentage"] == pytest.approx(0.0)
        # No non-bg classes in GT → overall_dice = 0.0
        assert result["Overall_DICE"] == pytest.approx(0.0)

    def test_remap_applied_to_prediction(self, tmp_path: Path) -> None:
        gt_mask = np.ones((4, 4), dtype=np.uint8)
        pred_mask = np.full((4, 4), 3, dtype=np.uint8)

        gt_path = tmp_path / "gt.png"
        pred_path = tmp_path / "pred.png"
        Image.fromarray(gt_mask).save(gt_path)
        Image.fromarray(pred_mask).save(pred_path)

        # 3 → 1 (tumor)
        result = analyze_roi(gt_path, pred_path, {0: 0, 1: 1, 2: 2, 3: 1})
        assert result["DICE_Tumor"] == pytest.approx(1.0)
