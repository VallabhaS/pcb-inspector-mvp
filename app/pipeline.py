"""
Pipeline orchestrator — ties together image analysis, risk scoring,
explanation generation, and result storage into a single call.
"""

import uuid
from datetime import datetime

from PIL import Image

from app.models.image_analyzer import ImageAnalyzer, overlay_heatmap
from app.models.risk_scorer import MetadataInput, compute_failure_risk
from app.models.explainer import generate_explanation
from app.schemas import AnalysisResponse
from app.storage import save_result


# Singleton analyzer (loads model once, reuses across requests)
_analyzer: ImageAnalyzer | None = None


def get_analyzer() -> ImageAnalyzer:
    """Lazy-load the image analyzer so model weights are only loaded once."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ImageAnalyzer()
    return _analyzer


def run_inspection(
    image: Image.Image,
    filename: str,
    metadata: MetadataInput | None = None,
) -> AnalysisResponse:
    """
    Run the full inspection pipeline on a single image.

    Steps:
      1. Analyze the image (defect detection + heatmap)
      2. Score failure risk (image severity + defect type + metadata)
      3. Generate human-readable explanation
      4. Save results to disk
      5. Return structured response
    """
    # 1. Image analysis
    analyzer = get_analyzer()
    analysis = analyzer.analyze(image)

    # 2. Risk scoring
    risk_result = compute_failure_risk(
        severity=analysis.severity,
        defect_category=analysis.defect_category,
        metadata=metadata,
    )

    # 3. Explanation
    explanation = generate_explanation(
        defect_category=analysis.defect_category,
        confidence=analysis.confidence,
        severity=analysis.severity,
        failure_risk=risk_result["failure_risk"],
    )

    # 4. Build result ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    result_id = f"insp_{timestamp}_{short_id}"

    # 5. Generate heatmap overlay and save
    heatmap_path = None
    if analysis.heatmap is not None:
        overlay = overlay_heatmap(image, analysis.heatmap)
        heatmap_path = save_result(result_id, overlay, suffix="heatmap")

    # 6. Build response
    response = AnalysisResponse(
        result_id=result_id,
        filename=filename,
        defect_category=analysis.defect_category,
        confidence=analysis.confidence,
        anomaly_score=analysis.anomaly_score,
        severity=analysis.severity,
        failure_risk=risk_result["failure_risk"],
        risk_breakdown=risk_result["breakdown"],
        explanation=explanation,
        heatmap_path=heatmap_path,
        metadata_used=metadata.to_dict() if metadata else None,
    )

    # 7. Save JSON report
    save_result(result_id, response.model_dump(), suffix="report")

    return response
