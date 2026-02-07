"""Tests for app/models.py — JSON serialization round-trips on Sample."""

from __future__ import annotations

from app.models import Sample, TissueType


class TestSampleScoringTumor:
    """Tests for scoring_tumor JSON get/set round-trip."""

    def test_set_and_get(self, sample: Sample) -> None:
        data = {"model1": 3, "model2": None, "model3": 5}
        sample.set_scoring_tumor(data)
        assert sample.get_scoring_tumor() == data

    def test_get_empty(self, sample: Sample) -> None:
        assert sample.scoring_tumor is None
        assert sample.get_scoring_tumor() == {}

    def test_overwrite(self, sample: Sample) -> None:
        sample.set_scoring_tumor({"model1": 1})
        sample.set_scoring_tumor({"model1": 5, "model2": 3})
        assert sample.get_scoring_tumor() == {"model1": 5, "model2": 3}


class TestSampleScoringNormal:
    """Tests for scoring_normal JSON get/set round-trip."""

    def test_set_and_get(self, sample: Sample) -> None:
        data = {"model1": None, "model2": 4}
        sample.set_scoring_normal(data)
        assert sample.get_scoring_normal() == data

    def test_get_empty(self, sample: Sample) -> None:
        assert sample.get_scoring_normal() == {}


class TestSampleCommentsTumor:
    """Tests for comments_tumor JSON get/set round-trip."""

    def test_set_and_get(self, sample: Sample) -> None:
        data = {"model1": "good", "model2": None}
        sample.set_comments_tumor(data)
        assert sample.get_comments_tumor() == data

    def test_get_empty(self, sample: Sample) -> None:
        assert sample.get_comments_tumor() == {}


class TestSampleCommentsNormal:
    """Tests for comments_normal JSON get/set round-trip."""

    def test_set_and_get(self, sample: Sample) -> None:
        data = {"model3": "artifact in corner"}
        sample.set_comments_normal(data)
        assert sample.get_comments_normal() == data

    def test_get_empty(self, sample: Sample) -> None:
        assert sample.get_comments_normal() == {}


class TestTissueType:
    """Tests for the TissueType enum."""

    def test_values(self) -> None:
        assert TissueType.tumor == "tumor"
        assert TissueType.normal == "normal"

    def test_membership(self) -> None:
        assert len(TissueType) == 2
