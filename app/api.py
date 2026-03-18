"""
FastAPI backend for PCB Inspector MVP.

Endpoints:
  POST /analyze       — upload an image (+ optional metadata) → full analysis
  GET  /results       — list past analysis results
  GET  /results/{id}  — retrieve a specific result
  GET  /health        — health check
"""

import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Ensure project root is on the path so `config` and `app` resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from app.models.risk_scorer import MetadataInput
from app.pipeline import run_inspection
from app.storage import list_results, load_result

app = FastAPI(
    title="PCB Inspector MVP",
    description="Analyze PCB/chip/wafer inspection images for defects and failure risk.",
    version="0.1.0",
)

# Serve saved heatmap images
config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(..., description="Inspection image (PNG, JPG, BMP)"),
    component_age_years: float = Form(0.0),
    operating_temp_celsius: float = Form(25.0),
    layer_count: int = Form(2),
    is_lead_free: bool = Form(True),
    environment: str = Form("indoor"),
):
    """
    Upload an inspection image and receive a full defect analysis.

    Optionally provide component metadata to improve the risk score.
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected an image file, got {file.content_type}"},
        )

    # Read image
    contents = await file.read()
    from io import BytesIO
    image = Image.open(BytesIO(contents))

    # Build metadata
    metadata = MetadataInput(
        component_age_years=component_age_years,
        operating_temp_celsius=operating_temp_celsius,
        layer_count=layer_count,
        is_lead_free=is_lead_free,
        environment=environment,
    )

    # Run pipeline
    result = run_inspection(
        image=image,
        filename=file.filename or "unknown.png",
        metadata=metadata,
    )

    return result.model_dump()


@app.get("/results")
def get_results(limit: int = 50):
    """List recent inspection results."""
    return list_results(limit=limit)


@app.get("/results/{result_id}")
def get_result(result_id: str):
    """Retrieve a specific inspection result by ID."""
    result = load_result(result_id)
    if result is None:
        return JSONResponse(status_code=404, content={"error": "Result not found"})
    return result


@app.get("/heatmap/{result_id}")
def get_heatmap(result_id: str):
    """Retrieve the heatmap image for a specific result."""
    heatmap_path = config.OUTPUTS_DIR / f"{result_id}_heatmap.png"
    if not heatmap_path.exists():
        return JSONResponse(status_code=404, content={"error": "Heatmap not found"})
    return FileResponse(str(heatmap_path), media_type="image/png")
