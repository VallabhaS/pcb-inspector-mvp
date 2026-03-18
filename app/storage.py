"""
Result storage module.

Saves analysis outputs (JSON reports + heatmap images) to the outputs/ directory.
Simple file-based storage — no database needed for the MVP.
"""

import json
from pathlib import Path

from PIL import Image

import config


def _ensure_output_dir() -> Path:
    """Create the outputs directory if it doesn't exist."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.OUTPUTS_DIR


def save_result(result_id: str, data, suffix: str = "report") -> str:
    """
    Save an analysis result to disk.

    Args:
        result_id: Unique identifier (e.g. "insp_20240101_120000_abc123")
        data: Either a dict (saved as JSON) or a PIL Image (saved as PNG)
        suffix: File suffix, e.g. "report" or "heatmap"

    Returns:
        The relative path to the saved file.
    """
    out_dir = _ensure_output_dir()

    if isinstance(data, Image.Image):
        filepath = out_dir / f"{result_id}_{suffix}.png"
        data.save(filepath)
    elif isinstance(data, dict):
        filepath = out_dir / f"{result_id}_{suffix}.json"
        filepath.write_text(json.dumps(data, indent=2, default=str))
    else:
        raise TypeError(f"Cannot save data of type {type(data)}")

    return str(filepath)


def load_result(result_id: str) -> dict | None:
    """Load a JSON report by result ID. Returns None if not found."""
    filepath = config.OUTPUTS_DIR / f"{result_id}_report.json"
    if not filepath.exists():
        return None
    return json.loads(filepath.read_text())


def list_results(limit: int = 50) -> list[dict]:
    """List recent analysis results, newest first."""
    out_dir = config.OUTPUTS_DIR
    if not out_dir.exists():
        return []

    reports = sorted(out_dir.glob("*_report.json"), reverse=True)[:limit]
    results = []
    for path in reports:
        try:
            data = json.loads(path.read_text())
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results
