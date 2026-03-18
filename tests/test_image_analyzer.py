"""Tests for the image analysis module."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.image_analyzer import ImageAnalyzer, overlay_heatmap
import config


def _make_test_image(color=(20, 80, 30), size=(256, 256)) -> Image.Image:
    """Create a simple solid test image."""
    return Image.new("RGB", size, color)


def test_analyzer_returns_valid_result():
    """Analyzer should return all expected fields with valid ranges."""
    analyzer = ImageAnalyzer()
    img = _make_test_image()
    result = analyzer.analyze(img)

    assert result.defect_category in config.DEFECT_CATEGORIES, (
        f"Unknown category: {result.defect_category}"
    )
    assert 0 <= result.confidence <= 1, f"Confidence out of range: {result.confidence}"
    assert 0 <= result.anomaly_score <= 1, f"Anomaly score out of range: {result.anomaly_score}"
    assert 0 <= result.severity <= 100, f"Severity out of range: {result.severity}"
    assert result.heatmap is not None, "Heatmap should not be None"
    assert result.heatmap.shape[2] == 3, "Heatmap should be RGB"


def test_heatmap_overlay():
    """overlay_heatmap should produce a valid blended image."""
    img = _make_test_image(size=(100, 100))
    heatmap = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = overlay_heatmap(img, heatmap, alpha=0.5)

    assert isinstance(result, Image.Image)
    assert result.size == (100, 100)


def test_different_images_produce_different_heatmaps():
    """Two very different images should produce different Grad-CAM heatmaps."""
    analyzer = ImageAnalyzer()
    dark = _make_test_image(color=(0, 0, 0))
    bright = _make_test_image(color=(255, 255, 255))

    r1 = analyzer.analyze(dark)
    r2 = analyzer.analyze(bright)

    # Heatmaps should differ even if anomaly scores both saturate
    assert not np.array_equal(r1.heatmap, r2.heatmap), (
        "Completely different images produced identical heatmaps"
    )


if __name__ == "__main__":
    test_analyzer_returns_valid_result()
    test_heatmap_overlay()
    test_different_images_produce_different_heatmaps()
    print("All image analyzer tests passed.")
