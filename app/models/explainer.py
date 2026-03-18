"""
Explanation generator module.

Produces human-readable explanations of defect findings and risk scores.
Uses template-based generation — no LLM required, fully deterministic.
"""

# ── Defect descriptions ──────────────────────────────────────────────────────

_DEFECT_INFO = {
    "scratch": {
        "description": "Surface scratch detected on the board",
        "likely_cause": "mechanical handling damage during manufacturing or assembly",
        "reliability_impact": "May expose underlying copper traces, leading to oxidation and eventual open circuits",
    },
    "contamination": {
        "description": "Foreign material or residue detected",
        "likely_cause": "flux residue, dust ingress, or chemical contamination during processing",
        "reliability_impact": "Can cause electrical leakage, dendritic growth, or corrosion over time",
    },
    "misalignment": {
        "description": "Component or layer misalignment detected",
        "likely_cause": "pick-and-place machine calibration error or stencil misregistration",
        "reliability_impact": "May cause poor solder joints, intermittent connections, or short circuits",
    },
    "solder_bridge": {
        "description": "Unintended solder connection between adjacent pads",
        "likely_cause": "excess solder paste, incorrect stencil aperture, or reflow profile issue",
        "reliability_impact": "Creates short circuits that can cause immediate functional failure",
    },
    "open_circuit": {
        "description": "Missing or broken electrical connection detected",
        "likely_cause": "insufficient solder, tombstoning, or trace damage",
        "reliability_impact": "Direct functional failure — the affected circuit path is non-operational",
    },
    "corrosion": {
        "description": "Oxidation or corrosion on metal surfaces",
        "likely_cause": "moisture exposure, galvanic reaction, or inadequate conformal coating",
        "reliability_impact": "Progressive degradation — increases resistance and eventually causes open circuits",
    },
    "crack": {
        "description": "Fracture in solder joint, trace, or substrate",
        "likely_cause": "thermal cycling stress, mechanical shock, or board flex",
        "reliability_impact": "Intermittent failures under thermal or mechanical stress, worsens over time",
    },
    "normal": {
        "description": "No significant defects detected",
        "likely_cause": "N/A",
        "reliability_impact": "Component appears within normal manufacturing tolerances",
    },
}


# ── Severity descriptions ────────────────────────────────────────────────────

def _severity_label(severity: int) -> str:
    if severity < 20:
        return "minimal"
    elif severity < 40:
        return "low"
    elif severity < 60:
        return "moderate"
    elif severity < 80:
        return "high"
    else:
        return "critical"


def _risk_label(risk: int) -> str:
    if risk < 20:
        return "very low"
    elif risk < 40:
        return "low"
    elif risk < 60:
        return "moderate"
    elif risk < 80:
        return "elevated"
    else:
        return "high"


# ── Public API ───────────────────────────────────────────────────────────────

def generate_explanation(
    defect_category: str,
    confidence: float,
    severity: int,
    failure_risk: int,
) -> str:
    """
    Generate a plain-English explanation of the inspection findings.

    Returns a multi-line string suitable for display in a report.
    """
    info = _DEFECT_INFO.get(defect_category, _DEFECT_INFO["normal"])
    sev_label = _severity_label(severity)
    risk_label = _risk_label(failure_risk)

    if defect_category == "normal":
        return (
            f"Finding: {info['description']}.\n"
            f"Confidence: {confidence:.0%}\n\n"
            f"The inspected area shows no anomalies exceeding detection thresholds. "
            f"Severity is {sev_label} ({severity}/100) and failure risk is "
            f"{risk_label} ({failure_risk}/100).\n\n"
            f"Recommendation: No action required. Standard quality."
        )

    return (
        f"Finding: {info['description']}.\n"
        f"Confidence: {confidence:.0%}\n\n"
        f"Likely cause: {info['likely_cause']}.\n\n"
        f"Reliability impact: {info['reliability_impact']}.\n\n"
        f"Severity: {sev_label} ({severity}/100)\n"
        f"Failure risk: {risk_label} ({failure_risk}/100)\n\n"
        f"Recommendation: "
        + _recommendation(defect_category, severity, failure_risk)
    )


def _recommendation(defect_category: str, severity: int, failure_risk: int) -> str:
    """Generate an actionable recommendation."""
    if failure_risk >= 70:
        return (
            "Immediate attention required. This defect poses a significant "
            "reliability risk. Recommend rework or rejection of this unit."
        )
    elif failure_risk >= 40:
        return (
            "Further inspection recommended. Consider additional testing "
            f"(e.g., X-ray for {defect_category}) before clearing this unit."
        )
    elif severity >= 30:
        return (
            "Monitor this defect. While the overall risk is manageable, "
            "document the finding for trend analysis."
        )
    else:
        return "Acceptable within tolerance. Log for records."
