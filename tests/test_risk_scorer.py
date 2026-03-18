"""Tests for the risk scoring module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.risk_scorer import MetadataInput, compute_failure_risk, compute_metadata_score


def test_normal_defect_low_risk():
    """A normal defect with low severity should give a low risk score."""
    result = compute_failure_risk(severity=10, defect_category="normal")
    assert result["failure_risk"] < 20, f"Expected low risk, got {result['failure_risk']}"


def test_severe_defect_high_risk():
    """A severe solder bridge should give a high risk score."""
    result = compute_failure_risk(severity=90, defect_category="solder_bridge")
    assert result["failure_risk"] > 50, f"Expected high risk, got {result['failure_risk']}"


def test_metadata_increases_risk():
    """Harsh metadata should increase the risk score."""
    base = compute_failure_risk(severity=50, defect_category="crack")
    harsh_meta = MetadataInput(
        component_age_years=15,
        operating_temp_celsius=120,
        layer_count=12,
        is_lead_free=True,
        environment="aerospace",
    )
    with_meta = compute_failure_risk(severity=50, defect_category="crack", metadata=harsh_meta)
    assert with_meta["failure_risk"] >= base["failure_risk"], (
        f"Harsh metadata should not decrease risk: {with_meta['failure_risk']} vs {base['failure_risk']}"
    )


def test_risk_breakdown_sums_correctly():
    """Component contributions should roughly sum to the total risk."""
    result = compute_failure_risk(severity=60, defect_category="corrosion")
    breakdown = result["breakdown"]
    total_from_parts = sum(breakdown.values())
    # Allow ±2 for rounding
    assert abs(total_from_parts - result["failure_risk"]) <= 2, (
        f"Breakdown sum {total_from_parts} != risk {result['failure_risk']}"
    )


def test_metadata_score_bounds():
    """Metadata score should always be between 0 and 1."""
    for age in [0, 5, 20, 50]:
        for temp in [-40, 25, 100, 150]:
            score = compute_metadata_score(MetadataInput(
                component_age_years=age, operating_temp_celsius=temp
            ))
            assert 0 <= score <= 1, f"Score {score} out of bounds for age={age}, temp={temp}"


if __name__ == "__main__":
    test_normal_defect_low_risk()
    test_severe_defect_high_risk()
    test_metadata_increases_risk()
    test_risk_breakdown_sums_correctly()
    test_metadata_score_bounds()
    print("All risk scorer tests passed.")
