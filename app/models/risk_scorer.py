"""
Failure-risk scoring module.

Computes a 0-100 failure-risk score from three inputs:
  1. Image severity (from the image analyzer)
  2. Defect-type risk weight (from config)
  3. Optional structured metadata (component age, environment, etc.)

The formula is a transparent weighted sum — fully interpretable, no black box.
"""

from dataclasses import dataclass

import config


@dataclass
class MetadataInput:
    """Optional structured metadata about the inspected component."""
    component_age_years: float = 0.0     # how old the component is
    operating_temp_celsius: float = 25.0  # typical operating temperature
    layer_count: int = 2                  # PCB layer count
    is_lead_free: bool = True             # lead-free solder?
    environment: str = "indoor"           # indoor | outdoor | automotive | aerospace

    def to_dict(self) -> dict:
        return {
            "component_age_years": self.component_age_years,
            "operating_temp_celsius": self.operating_temp_celsius,
            "layer_count": self.layer_count,
            "is_lead_free": self.is_lead_free,
            "environment": self.environment,
        }


# Environment risk multipliers
_ENV_RISK = {
    "indoor":     0.2,
    "outdoor":    0.5,
    "automotive": 0.7,
    "aerospace":  0.9,
}


def compute_metadata_score(meta: MetadataInput) -> float:
    """
    Convert metadata into a single 0-1 risk factor.

    Each factor is normalized to [0, 1] and averaged.
    """
    age_factor = min(1.0, meta.component_age_years / 20.0)
    temp_factor = min(1.0, max(0.0, (meta.operating_temp_celsius - 20) / 80.0))
    layer_factor = min(1.0, meta.layer_count / 16.0)
    solder_factor = 0.3 if meta.is_lead_free else 0.1
    env_factor = _ENV_RISK.get(meta.environment, 0.3)

    return (age_factor + temp_factor + layer_factor + solder_factor + env_factor) / 5.0


def compute_failure_risk(
    severity: int,
    defect_category: str,
    metadata: MetadataInput | None = None,
) -> dict:
    """
    Compute the final failure-risk score.

    Returns:
        dict with keys: failure_risk (0-100), breakdown (component scores)
    """
    # Normalize severity to 0-1
    image_score = severity / 100.0

    # Defect type weight
    defect_weight = config.DEFECT_RISK_WEIGHTS.get(defect_category, 0.5)

    # Metadata score
    if metadata is not None:
        meta_score = compute_metadata_score(metadata)
    else:
        meta_score = 0.3  # neutral default when no metadata provided

    # Weighted combination
    raw_risk = (
        config.RISK_WEIGHT_IMAGE * image_score
        + config.RISK_WEIGHT_DEFECT * defect_weight
        + config.RISK_WEIGHT_META * meta_score
    )

    failure_risk = int(min(100, max(0, round(raw_risk * 100))))

    return {
        "failure_risk": failure_risk,
        "breakdown": {
            "image_severity_contribution": round(config.RISK_WEIGHT_IMAGE * image_score * 100, 1),
            "defect_type_contribution": round(config.RISK_WEIGHT_DEFECT * defect_weight * 100, 1),
            "metadata_contribution": round(config.RISK_WEIGHT_META * meta_score * 100, 1),
        },
    }
