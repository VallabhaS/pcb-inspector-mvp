"""
Pipeline orchestrator — the hybrid analysis engine.

Runs two independent analyses:
  1. CNN (ResNet18)  → spatial anomaly detection + Grad-CAM heatmap
  2. Vision API (Claude) → semantic defect understanding

Then cross-validates their findings, fuses the results, computes risk,
and generates an explanation that draws on both sources.

If the Vision API is unavailable (no key, no credits), the system
gracefully degrades to CNN-only mode — the app always works.

This is what separates this system from a generic AI wrapper:
  - The CNN provides something no API can: pixel-level localization
  - The Vision API provides something no CNN can: semantic understanding
  - Cross-validation catches when either model is wrong
  - The risk scorer is deterministic and auditable
"""

import logging
import uuid
from datetime import datetime

from PIL import Image

from app.models.image_analyzer import ImageAnalyzer, overlay_heatmap
from app.models.vision_analyzer import VisionAnalyzer, VisionAnalysisResult, cross_validate
from app.models.risk_scorer import MetadataInput, compute_failure_risk
from app.models.explainer import generate_explanation
from app.schemas import AnalysisResponse
from app.storage import save_result

logger = logging.getLogger(__name__)

# Singletons (load once, reuse across requests)
_cnn_analyzer: ImageAnalyzer | None = None
_vision_analyzer: VisionAnalyzer | None = None
_vision_available: bool | None = None  # None = not checked yet


def get_cnn_analyzer() -> ImageAnalyzer:
    """Lazy-load the CNN analyzer so model weights are only loaded once."""
    global _cnn_analyzer
    if _cnn_analyzer is None:
        _cnn_analyzer = ImageAnalyzer()
    return _cnn_analyzer


def get_vision_analyzer() -> VisionAnalyzer | None:
    """Lazy-load the Vision API analyzer. Returns None if unavailable."""
    global _vision_analyzer, _vision_available
    if _vision_available is False:
        return None
    if _vision_analyzer is None:
        try:
            _vision_analyzer = VisionAnalyzer()
            _vision_available = True
        except (ValueError, Exception) as e:
            logger.warning(f"Vision API unavailable: {e}. Running in CNN-only mode.")
            _vision_available = False
            return None
    return _vision_analyzer


def run_inspection(
    image: Image.Image,
    filename: str,
    metadata: MetadataInput | None = None,
) -> AnalysisResponse:
    """
    Run the full hybrid inspection pipeline on a single image.

    Steps:
      1. CNN analysis  → anomaly score + heatmap (spatial)
      2. Vision API    → semantic defect understanding (if available)
      3. Cross-validate both analyses (or use CNN-only)
      4. Fused risk scoring
      5. Generate explanation
      6. Save everything
    """
    # ── Step 1: CNN spatial analysis (always runs) ───────────────────────
    cnn = get_cnn_analyzer()
    cnn_result = cnn.analyze(image)

    # ── Step 2: Vision API semantic analysis (best-effort) ───────────────
    vision_result = None
    vision_analyzer = get_vision_analyzer()
    if vision_analyzer is not None:
        try:
            vision_result = vision_analyzer.analyze(image)
            logger.info("Vision API analysis completed successfully.")
        except Exception as e:
            logger.warning(f"Vision API call failed: {e}. Falling back to CNN-only.")
            vision_result = None

    # ── Step 3: Determine final classification ───────────────────────────
    if vision_result is not None and vision_result.confidence > 0:
        # Hybrid mode: cross-validate CNN + Vision
        fusion = cross_validate(
            cnn_category=cnn_result.defect_category,
            cnn_anomaly_score=cnn_result.anomaly_score,
            vision_result=vision_result,
        )
        final_category = fusion["final_category"]
        final_confidence = fusion["fused_confidence"]

        # Blend severities (vision model understands context better)
        vision_severity_map = {"low": 20, "medium": 45, "high": 70, "critical": 90}
        vision_sev_num = vision_severity_map.get(fusion["vision_severity"], 40)
        blended_severity = int(0.35 * cnn_result.severity + 0.65 * vision_sev_num)
        blended_severity = min(100, max(0, blended_severity))
        mode = "hybrid"
    else:
        # CNN-only mode
        fusion = None
        final_category = cnn_result.defect_category
        final_confidence = cnn_result.confidence
        blended_severity = cnn_result.severity
        mode = "cnn-only"

    # ── Step 4: Risk scoring ─────────────────────────────────────────────
    risk_result = compute_failure_risk(
        severity=blended_severity,
        defect_category=final_category,
        metadata=metadata,
    )

    # ── Step 5: Build explanation ────────────────────────────────────────
    base_explanation = generate_explanation(
        defect_category=final_category,
        confidence=final_confidence,
        severity=blended_severity,
        failure_risk=risk_result["failure_risk"],
    )

    if fusion is not None:
        explanation = _build_hybrid_explanation(
            base_explanation=base_explanation,
            fusion=fusion,
            cnn_anomaly=cnn_result.anomaly_score,
        )
    else:
        explanation = base_explanation + (
            "\n\nNote: Analysis ran in CNN-only mode (Vision API unavailable). "
            "Results are based on spatial anomaly detection only. "
            "Add API credits for full hybrid analysis."
        )

    # ── Step 6: Build result ID ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    result_id = f"insp_{timestamp}_{short_id}"

    # ── Step 7: Heatmap ──────────────────────────────────────────────────
    heatmap_path = None
    if cnn_result.heatmap is not None:
        overlay = overlay_heatmap(image, cnn_result.heatmap)
        heatmap_path = save_result(result_id, overlay, suffix="heatmap")

    # ── Step 8: Build response ───────────────────────────────────────────
    analysis_detail = {"mode": mode}
    if fusion is not None:
        analysis_detail.update({
            "agreement": fusion["agreement"],
            "cnn_category": fusion["cnn_category"],
            "vision_category": fusion["vision_category"],
            "board_description": fusion["board_description"],
            "all_defects_found": fusion["all_defects"],
            "vision_reasoning": fusion["vision_reasoning"],
        })

    response = AnalysisResponse(
        result_id=result_id,
        filename=filename,
        defect_category=final_category,
        confidence=final_confidence,
        anomaly_score=cnn_result.anomaly_score,
        severity=blended_severity,
        failure_risk=risk_result["failure_risk"],
        risk_breakdown=risk_result["breakdown"],
        explanation=explanation,
        heatmap_path=heatmap_path,
        metadata_used=metadata.to_dict() if metadata else None,
        analysis_detail=analysis_detail,
    )

    # ── Step 9: Save ─────────────────────────────────────────────────────
    save_result(result_id, response.model_dump(), suffix="report")

    return response


def _build_hybrid_explanation(
    base_explanation: str,
    fusion: dict,
    cnn_anomaly: float,
) -> str:
    """Combine the template explanation with vision model insights."""
    parts = [base_explanation]

    if fusion["board_description"]:
        parts.append(f"\nBoard context: {fusion['board_description']}")

    agreement = fusion["agreement"]
    if agreement == "strong":
        parts.append(
            "\nCross-validation: Both CNN and vision model agree on this finding "
            "(high confidence)."
        )
    elif agreement == "partial":
        parts.append(
            f"\nCross-validation: Models partially agree. "
            f"CNN detected: {fusion['cnn_category']}. "
            f"Vision model detected: {fusion['vision_category']}. "
            f"Using vision model's classification (better semantic understanding)."
        )
    else:
        parts.append(
            f"\nCross-validation: Models DISAGREE. "
            f"CNN detected: {fusion['cnn_category']} (anomaly score: {cnn_anomaly:.2f}). "
            f"Vision model detected: {fusion['vision_category']}. "
            f"Recommend manual review."
        )

    if fusion["vision_reasoning"]:
        parts.append(f"\nDetailed analysis: {fusion['vision_reasoning']}")

    all_defects = fusion["all_defects"]
    if len(all_defects) > 1:
        additional = [d["type"] for d in all_defects[1:]]
        parts.append(
            f"\nAdditional findings: {', '.join(additional)} "
            f"(secondary defects detected by vision model)"
        )

    return "\n".join(parts)
