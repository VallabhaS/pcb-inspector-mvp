"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, ConfigDict, Field


class MetadataRequest(BaseModel):
    """Optional metadata about the component being inspected."""
    component_age_years: float = Field(0.0, ge=0, description="Component age in years")
    operating_temp_celsius: float = Field(25.0, description="Typical operating temperature")
    layer_count: int = Field(2, ge=1, le=64, description="PCB layer count")
    is_lead_free: bool = Field(True, description="Whether lead-free solder was used")
    environment: str = Field("indoor", description="Operating environment: indoor|outdoor|automotive|aerospace")


class AnalysisResponse(BaseModel):
    """Full inspection result returned by the API."""
    # Identifiers
    result_id: str = Field(..., description="Unique ID for this analysis result")
    filename: str = Field(..., description="Original image filename")

    # Defect analysis
    defect_category: str = Field(..., description="Predicted defect type")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence in the prediction")
    anomaly_score: float = Field(..., ge=0, le=1, description="How anomalous the image appears")

    # Scores
    severity: int = Field(..., ge=0, le=100, description="Severity score")
    failure_risk: int = Field(..., ge=0, le=100, description="Failure risk score")
    risk_breakdown: dict = Field(..., description="Component contributions to risk score")

    # Explanation
    explanation: str = Field(..., description="Human-readable finding summary")

    # Output paths
    heatmap_path: str | None = Field(None, description="Path to heatmap overlay image")
    metadata_used: dict | None = Field(None, description="Metadata that was factored into risk")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "result_id": "insp_20240101_120000_abc123",
            "filename": "board_top_001.png",
            "defect_category": "solder_bridge",
            "confidence": 0.82,
            "anomaly_score": 0.67,
            "severity": 67,
            "failure_risk": 58,
            "risk_breakdown": {
                "image_severity_contribution": 26.8,
                "defect_type_contribution": 28.0,
                "metadata_contribution": 7.5,
            },
            "explanation": "Finding: Unintended solder connection...",
            "heatmap_path": "outputs/insp_20240101_120000_abc123_heatmap.png",
            "metadata_used": None,
        }
    })
