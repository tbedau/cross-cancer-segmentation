"""Shared fixtures for scoring-app tests."""

from __future__ import annotations

import pytest

from app.models import Sample


@pytest.fixture
def sample() -> Sample:
    """Return a minimal Sample instance for testing JSON round-trips."""
    return Sample(
        tcga_project_id=1,
        tcga_case_id="TCGA-AB-1234",
        slide_mpp=0.461,
    )
