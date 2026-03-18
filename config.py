"""
Central configuration for pcb-inspector-mvp.

All tuneable constants live here so they're easy to find and change.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ── Image analysis ───────────────────────────────────────────────────────────
IMAGE_SIZE = 224                    # ResNet18 input size
ANOMALY_THRESHOLD = 0.45            # above this → likely defective
HEATMAP_ALPHA = 0.5                 # overlay transparency for Grad-CAM

# ── Defect categories ────────────────────────────────────────────────────────
DEFECT_CATEGORIES = [
    "scratch",
    "contamination",
    "misalignment",
    "solder_bridge",
    "open_circuit",
    "corrosion",
    "crack",
    "normal",
]

# Risk weight per defect type (higher = more dangerous)
DEFECT_RISK_WEIGHTS = {
    "scratch":       0.3,
    "contamination": 0.4,
    "misalignment":  0.6,
    "solder_bridge": 0.8,
    "open_circuit":  0.9,
    "corrosion":     0.7,
    "crack":         0.85,
    "normal":        0.0,
}

# ── Risk scoring ─────────────────────────────────────────────────────────────
# failure_risk = w_image * severity + w_defect * defect_weight + w_meta * meta_score
RISK_WEIGHT_IMAGE = 0.40
RISK_WEIGHT_DEFECT = 0.35
RISK_WEIGHT_META = 0.25

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000
