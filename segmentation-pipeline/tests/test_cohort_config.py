"""Tests for cohort_config.py — configuration integrity checks."""

from __future__ import annotations

from cohort_config import COHORT_SPECS, Cohort, ModelKey, get_remap_lut


class TestCohortCompleteness:
    """Verify that every (Cohort, ModelKey) pair is defined."""

    def test_all_cohorts_have_specs(self) -> None:
        for cohort in Cohort:
            assert cohort in COHORT_SPECS, f"Missing spec for {cohort}"

    def test_all_models_in_every_cohort(self) -> None:
        for cohort, spec in COHORT_SPECS.items():
            for model in ModelKey:
                assert model in spec.model_maps, (
                    f"Missing model map for {model} in cohort {cohort}"
                )

    def test_all_mappings_target_012(self) -> None:
        for cohort, spec in COHORT_SPECS.items():
            for model, class_map in spec.model_maps.items():
                target_values = set(class_map.values())
                assert target_values.issubset({0, 1, 2}), (
                    f"Invalid target values {target_values} for {cohort}/{model}"
                )

    def test_all_mappings_include_zero_to_zero(self) -> None:
        for cohort, spec in COHORT_SPECS.items():
            for model, class_map in spec.model_maps.items():
                assert class_map.get(0) == 0, f"Background not mapped to 0 for {cohort}/{model}"


class TestGetRemapLut:
    """Tests for the get_remap_lut lookup function."""

    def test_returns_correct_map(self) -> None:
        for cohort in Cohort:
            for model in ModelKey:
                expected = COHORT_SPECS[cohort].model_maps[model]
                assert get_remap_lut(cohort, model) is expected


class TestEnumValues:
    """Verify enum string values match expectations."""

    def test_cohort_values(self) -> None:
        assert Cohort.BREAST == "BRCA"
        assert Cohort.COADREAD == "COADREAD"
        assert Cohort.LUNG == "LUNG"
        assert Cohort.PROSTATE == "PRAD"

    def test_model_key_values(self) -> None:
        assert ModelKey.BREAST == "BREAST"
        assert ModelKey.COLON == "COLON"
        assert ModelKey.LUNG == "LUNG"
        assert ModelKey.KIDNEY == "KIDNEY"
        assert ModelKey.PROSTATE == "PROSTATE"
