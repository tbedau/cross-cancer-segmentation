"""Tests for db_tools.py — TCGA filename parsing."""

from __future__ import annotations

from db_tools import parse_filename


class TestParseFilename:
    """Tests for the TCGA filename regex parser."""

    def test_full_filename_with_uuid(self) -> None:
        filename = (
            "TCGA-SKCM-AB-1234-01A-01.aabbccdd-1122-3344-5566-778899aabbcc_0.461_tumor_model1.jpg"
        )
        result = parse_filename(filename)
        assert result is not None
        tcga_project, tcga_case_id, _, _, slide_mpp, tissue_type = result
        assert tcga_project == "TCGA-SKCM"
        assert tcga_case_id == "TCGA-AB-1234"
        assert slide_mpp == "0.461"
        assert tissue_type == "tumor"

    def test_normal_tissue(self) -> None:
        filename = (
            "TCGA-BRCA-AA-AAAA-01A-01.aabbccdd-1122-3344-5566-778899aabbcc_0.500_normal_model3.jpg"
        )
        result = parse_filename(filename)
        assert result is not None
        assert result[5] == "normal"

    def test_all_model_numbers(self) -> None:
        for i in range(1, 6):
            filename = (
                f"TCGA-LUAD-BB-BBBB-01A-01"
                f".aabbccdd-1122-3344-5566-778899aabbcc_0.461_tumor_model{i}.jpg"
            )
            assert parse_filename(filename) is not None

    def test_invalid_model_number(self) -> None:
        for i in [0, 6, 9]:
            filename = (
                f"TCGA-LUAD-BB-BBBB-01A-01"
                f".aabbccdd-1122-3344-5566-778899aabbcc_0.461_tumor_model{i}.jpg"
            )
            assert parse_filename(filename) is None

    def test_non_tcga_filename(self) -> None:
        assert parse_filename("random_image.jpg") is None

    def test_minimal_filename_without_uuid(self) -> None:
        filename = "TCGA-PRAD-CC-CCCC_0.253_tumor_model2.jpg"
        result = parse_filename(filename)
        assert result is not None
        assert result[0] == "TCGA-PRAD"
        assert result[1] == "TCGA-CC-CCCC"
        assert result[4] == "0.253"
