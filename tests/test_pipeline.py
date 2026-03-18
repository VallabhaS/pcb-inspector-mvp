"""Tests for the end-to-end pipeline."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import run_inspection
from app.models.risk_scorer import MetadataInput
from app.schemas import AnalysisResponse
import config


def _make_test_image() -> Image.Image:
    return Image.new("RGB", (256, 256), (20, 80, 30))


def test_pipeline_returns_valid_response():
    """Full pipeline should return a valid AnalysisResponse."""
    result = run_inspection(
        image=_make_test_image(),
        filename="test_image.png",
    )
    assert isinstance(result, AnalysisResponse)
    assert result.result_id.startswith("insp_")
    assert result.filename == "test_image.png"
    assert 0 <= result.severity <= 100
    assert 0 <= result.failure_risk <= 100
    assert len(result.explanation) > 10


def test_pipeline_with_metadata():
    """Pipeline should accept and use metadata."""
    meta = MetadataInput(
        component_age_years=10,
        operating_temp_celsius=85,
        layer_count=8,
        environment="automotive",
    )
    result = run_inspection(
        image=_make_test_image(),
        filename="test_with_meta.png",
        metadata=meta,
    )
    assert result.metadata_used is not None
    assert result.metadata_used["environment"] == "automotive"


def test_pipeline_saves_output():
    """Pipeline should save results to the outputs directory."""
    result = run_inspection(
        image=_make_test_image(),
        filename="test_save.png",
    )
    report_path = config.OUTPUTS_DIR / f"{result.result_id}_report.json"
    assert report_path.exists(), f"Report not saved at {report_path}"


if __name__ == "__main__":
    test_pipeline_returns_valid_response()
    test_pipeline_with_metadata()
    test_pipeline_saves_output()
    print("All pipeline tests passed.")
