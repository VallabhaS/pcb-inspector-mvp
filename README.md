# PCB Inspector MVP

A local ML application that analyzes PCB, chip, and wafer inspection images to detect visible defects, assign severity scores, and predict failure risk.

## What It Does

Upload an inspection image → get back:
- **Defect category** (scratch, contamination, solder bridge, crack, etc.)
- **Suspicious region heatmap** (Grad-CAM visualization)
- **Severity score** (0–100)
- **Failure risk score** (0–100)
- **Plain-English explanation** of the finding and recommended action

## Architecture

```
Streamlit UI  ←→  FastAPI Backend
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   Image Analyzer  Risk Scorer  Explainer
   (ResNet18 +     (weighted    (template-
    Grad-CAM)       formula)     based)
         │
         ▼
    Result Storage (JSON + PNG)
```

**Image Analyzer** — Uses a pretrained ResNet18 (ImageNet weights) as a feature extractor. Computes anomaly scores from activation statistics and generates Grad-CAM heatmaps. This is zero-shot anomaly detection, not a fine-tuned PCB classifier. It provides a credible baseline that can be upgraded with PCB-specific training data.

**Risk Scorer** — Deterministic weighted formula combining image severity, defect-type risk weight, and optional component metadata (age, temperature, environment, layer count). Fully interpretable — no black box.

**Explainer** — Template-based generation mapping defect type + severity + risk into human-readable reports with actionable recommendations.

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repo
git clone https://github.com/VallabhaS/pcb-inspector-mvp.git
cd pcb-inspector-mvp

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample images
python data/generate_samples.py
```

### Run

```bash
# Option 1: Launch both services
bash run.sh

# Option 2: Launch manually
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload &
streamlit run ui/streamlit_app.py --server.port 8501
```

Then open http://localhost:8501 in your browser.

### Run Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
pcb-inspector-mvp/
├── config.py                  # All tuneable constants
├── requirements.txt           # Python dependencies
├── run.sh                     # One-command launcher
├── app/
│   ├── api.py                 # FastAPI endpoints
│   ├── pipeline.py            # Orchestrates analysis → scoring → explanation
│   ├── schemas.py             # Pydantic request/response models
│   ├── storage.py             # File-based result persistence
│   └── models/
│       ├── image_analyzer.py  # ResNet18 feature extraction + Grad-CAM
│       ├── risk_scorer.py     # Weighted risk formula
│       └── explainer.py       # Template-based explanations
├── ui/
│   └── streamlit_app.py       # Streamlit frontend
├── tests/                     # Unit + integration tests
├── data/
│   ├── samples/               # Synthetic sample images
│   └── generate_samples.py    # Sample image generator
└── outputs/                   # Saved analysis results
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload image + optional metadata → full analysis |
| `GET` | `/results` | List recent inspection results |
| `GET` | `/results/{id}` | Get a specific result |
| `GET` | `/heatmap/{id}` | Get the heatmap image for a result |
| `GET` | `/health` | Health check |

### Example: Analyze via curl

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "file=@data/samples/sample_scratch.png" \
  -F "component_age_years=5" \
  -F "environment=automotive"
```

## Example Output

```json
{
  "result_id": "insp_20240315_143022_a1b2c3d4",
  "filename": "sample_scratch.png",
  "defect_category": "scratch",
  "confidence": 0.72,
  "anomaly_score": 0.64,
  "severity": 64,
  "failure_risk": 45,
  "risk_breakdown": {
    "image_severity_contribution": 25.6,
    "defect_type_contribution": 10.5,
    "metadata_contribution": 8.8
  },
  "explanation": "Finding: Surface scratch detected on the board.\nConfidence: 72%\n\nLikely cause: mechanical handling damage during manufacturing or assembly.\n\nReliability impact: May expose underlying copper traces, leading to oxidation and eventual open circuits.\n\nSeverity: moderate (64/100)\nFailure risk: moderate (45/100)\n\nRecommendation: Further inspection recommended...",
  "heatmap_path": "outputs/insp_20240315_143022_a1b2c3d4_heatmap.png"
}
```

## Honesty Note

This MVP uses a pretrained ImageNet model (ResNet18) for feature extraction — it was **not** trained on PCB-specific data. The defect classification uses heuristics on activation statistics, not a validated PCB classifier. This provides a working baseline architecture that can be upgraded by:

1. Fine-tuning on real PCB datasets (DeepPCB, MVTec AD, HRIPCB)
2. Replacing the heuristic classifier with a trained classification head
3. Training an autoencoder for more precise anomaly detection

The risk scorer and explainer are deterministic and interpretable — what you see is exactly what the system computed.

## Roadmap

- [ ] Fine-tune on DeepPCB or MVTec AD dataset
- [ ] Replace heuristic classifier with trained classification head
- [ ] Add batch processing for multiple images
- [ ] Add anomaly autoencoder for better defect sensitivity
- [ ] Add PDF report export
- [ ] Add comparison mode (before/after, golden reference vs. sample)
- [ ] Containerize with Docker
- [ ] Add user authentication for multi-user deployments

## Datasets for Future Training

| Dataset | Description | Link |
|---------|-------------|------|
| DeepPCB | 1,500 image pairs with 6 defect types | [github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB) |
| MVTec AD | Industrial anomaly detection benchmark | [mvtec.com/company/research](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| HRIPCB | High-resolution PCB defect dataset | Available via academic request |

## License

MIT
