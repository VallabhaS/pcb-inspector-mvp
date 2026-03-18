"""
Vision API analyzer using Google Gemini's multimodal capabilities.

This module sends inspection images to Gemini with domain-specific
PCB/semiconductor prompting and parses structured defect analysis back.
It serves as the "semantic brain" of the pipeline — understanding WHAT a
defect is and WHY it matters — while the CNN module (image_analyzer.py)
handles spatial localization (WHERE it is).

This is NOT a generic "ask the AI" wrapper. The prompt engineering encodes
PCB inspection domain knowledge, enforces structured output, and the results
are cross-validated against the CNN's independent analysis.
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class VisionAnalysisResult:
    """Structured output from the vision model."""
    defects_found: list[dict] = field(default_factory=list)  # [{type, description, location, severity}]
    board_description: str = ""        # what the model sees (board type, components, etc.)
    overall_condition: str = ""        # good / fair / poor / critical
    confidence: float = 0.0            # 0-1 how confident the model is
    reasoning: str = ""                # chain of thought from the model
    raw_response: str = ""             # full model response for debugging


# ── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert PCB and semiconductor inspection engineer with 20 years
of experience in quality control for electronics manufacturing. You specialize in visual
inspection of printed circuit boards, chip dies, wafer surfaces, and solder joints.

Your task: analyze the provided inspection image and return a structured JSON assessment.

You must be HONEST. If the image is not a PCB/chip/wafer, say so. If you cannot identify
a defect with confidence, say so. Never fabricate findings.

Defect types you know:
- scratch: surface scratches on board or traces
- contamination: foreign material, flux residue, dust
- misalignment: component or layer offset from intended position
- solder_bridge: unintended solder connection between adjacent pads
- open_circuit: missing or broken connection
- corrosion: oxidation or chemical degradation of metal surfaces
- crack: fracture in solder joint, trace, or substrate
- delamination: separation of board layers
- tombstoning: component standing up on one end
- void: cavity in solder joint (often invisible without X-ray)
- normal: no defects detected

Respond with ONLY valid JSON in this exact format:
{
  "is_inspection_image": true,
  "board_description": "Brief description of what you see — board type, visible components, surface finish",
  "defects": [
    {
      "type": "defect_type_from_list_above",
      "description": "Specific description of this defect",
      "location": "Where on the image (e.g., upper-left quadrant, near IC U3, center of board)",
      "severity": "low|medium|high|critical",
      "confidence": 0.85
    }
  ],
  "overall_condition": "good|fair|poor|critical",
  "confidence": 0.8,
  "reasoning": "Step-by-step explanation of your analysis process and findings"
}

If no defects are found, return an empty defects array.
If the image is not an inspection image, set is_inspection_image to false and explain in reasoning."""


# ── Analyzer class ───────────────────────────────────────────────────────────

class VisionAnalyzer:
    """Gemini-powered semantic defect analyzer for inspection images."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env or as an environment variable."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def analyze(self, image: Image.Image) -> VisionAnalysisResult:
        """
        Send an inspection image to Gemini for semantic defect analysis.

        Returns structured findings that complement the CNN's spatial analysis.
        """
        # Ensure image is RGB
        image_rgb = image.convert("RGB")

        # Call Gemini with the inspection image
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                "Analyze this inspection image for defects. "
                "Follow the system instructions exactly.",
                image_rgb,
            ],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=1500,
                temperature=0.2,
            ),
        )

        raw_text = response.text
        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> VisionAnalysisResult:
        """Parse Gemini's JSON response into a structured result."""
        try:
            # Handle cases where the model wraps JSON in markdown code blocks
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]  # remove first line
                cleaned = cleaned.rsplit("```", 1)[0]  # remove last fence
                cleaned = cleaned.strip()

            data = json.loads(cleaned)

            defects = []
            for d in data.get("defects", []):
                defects.append({
                    "type": d.get("type", "unknown"),
                    "description": d.get("description", ""),
                    "location": d.get("location", ""),
                    "severity": d.get("severity", "medium"),
                    "confidence": float(d.get("confidence", 0.5)),
                })

            return VisionAnalysisResult(
                defects_found=defects,
                board_description=data.get("board_description", ""),
                overall_condition=data.get("overall_condition", "unknown"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                raw_response=raw_text,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # If parsing fails, return a degraded result with the raw text
            return VisionAnalysisResult(
                confidence=0.0,
                reasoning=f"Failed to parse structured response: {e}",
                raw_response=raw_text,
            )


# ── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate(
    cnn_category: str,
    cnn_anomaly_score: float,
    vision_result: VisionAnalysisResult,
) -> dict:
    """
    Cross-validate CNN and Vision API findings.

    Returns a fused assessment with agreement level and final classification.
    Both models analyzing independently and then comparing = more trustworthy
    than either alone.
    """
    # Extract the primary defect from vision results
    if vision_result.defects_found:
        top_vision_defect = vision_result.defects_found[0]["type"]
        vision_severity = vision_result.defects_found[0]["severity"]
    else:
        top_vision_defect = "normal"
        vision_severity = "low"

    # Determine agreement
    cnn_says_defective = cnn_category != "normal"
    vision_says_defective = top_vision_defect != "normal"
    types_match = cnn_category == top_vision_defect

    if types_match:
        agreement = "strong"
        final_category = cnn_category
        confidence_boost = 0.15
    elif cnn_says_defective == vision_says_defective:
        # Both agree something is wrong (or both say normal), but differ on type
        # Trust the vision model for classification (it understands context)
        agreement = "partial"
        final_category = top_vision_defect if vision_says_defective else "normal"
        confidence_boost = 0.05
    else:
        # One says defective, the other says normal — flag for review
        agreement = "disagreement"
        # If CNN anomaly score is high, lean toward defective
        if cnn_anomaly_score > 0.6:
            final_category = top_vision_defect if vision_says_defective else cnn_category
        else:
            final_category = top_vision_defect
        confidence_boost = -0.1  # reduce confidence when models disagree

    # Fused confidence
    base_confidence = vision_result.confidence
    fused_confidence = min(0.99, max(0.1, base_confidence + confidence_boost))

    return {
        "final_category": final_category,
        "fused_confidence": round(fused_confidence, 3),
        "agreement": agreement,
        "cnn_category": cnn_category,
        "vision_category": top_vision_defect,
        "vision_severity": vision_severity,
        "vision_reasoning": vision_result.reasoning,
        "board_description": vision_result.board_description,
        "all_defects": vision_result.defects_found,
    }
