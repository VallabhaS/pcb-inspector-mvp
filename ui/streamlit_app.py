"""
Streamlit frontend for PCB Inspector MVP.

Hybrid analysis UI — shows CNN heatmap, vision model reasoning,
cross-validation status, scores, and actionable explanation.
"""

import sys
from pathlib import Path
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

API_URL = f"http://{config.API_HOST}:{config.API_PORT}"

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PCB Inspector",
    page_icon="🔬",
    layout="wide",
)

st.title("PCB Inspector")
st.caption(
    "Hybrid CNN + Vision AI inspection system. "
    "Upload a PCB / chip / wafer image for defect analysis."
)

# ── Sidebar: metadata inputs ────────────────────────────────────────────────

st.sidebar.header("Component Metadata")
st.sidebar.caption("Optional — improves risk scoring accuracy")

component_age = st.sidebar.slider("Component age (years)", 0.0, 30.0, 0.0, 0.5)
operating_temp = st.sidebar.slider("Operating temperature (°C)", -40.0, 150.0, 25.0, 5.0)
layer_count = st.sidebar.number_input("PCB layer count", 1, 64, 2)
is_lead_free = st.sidebar.checkbox("Lead-free solder", value=True)
environment = st.sidebar.selectbox(
    "Operating environment",
    ["indoor", "outdoor", "automotive", "aerospace"],
)

st.sidebar.divider()
st.sidebar.caption(
    "**How it works:** Two independent models analyze your image — "
    "a CNN for spatial anomaly detection and Claude's vision AI for "
    "semantic understanding. Results are cross-validated for reliability."
)

# ── Main: image upload ──────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload inspection image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_heat = st.columns([1, 1])
    with col_img:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing with CNN + Vision AI — this takes a few seconds..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {
                    "component_age_years": component_age,
                    "operating_temp_celsius": operating_temp,
                    "layer_count": layer_count,
                    "is_lead_free": is_lead_free,
                    "environment": environment,
                }
                response = requests.post(f"{API_URL}/analyze", files=files, data=data, timeout=120)
                response.raise_for_status()
                result = response.json()

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the API. "
                    f"Make sure the backend is running on {API_URL}."
                )
                st.stop()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # ── Heatmap ──────────────────────────────────────────────────────
        with col_heat:
            st.subheader("CNN Anomaly Heatmap")
            if result.get("heatmap_path"):
                try:
                    heatmap_resp = requests.get(
                        f"{API_URL}/heatmap/{result['result_id']}", timeout=10
                    )
                    if heatmap_resp.status_code == 200:
                        heatmap_img = Image.open(BytesIO(heatmap_resp.content))
                        st.image(heatmap_img, use_container_width=True)
                        st.caption("Warm colors = regions the CNN flagged as anomalous")
                except Exception:
                    st.info("Heatmap could not be loaded.")

        st.divider()

        # ── Analysis mode badge ──────────────────────────────────────────
        detail = result.get("analysis_detail") or {}
        mode = detail.get("mode", "unknown")
        if mode == "hybrid":
            st.success("**Analysis mode: Hybrid** — CNN spatial analysis + Claude Vision AI semantic analysis")
        else:
            st.warning(
                "**Analysis mode: CNN-only** — Vision API unavailable. "
                "Add credits at console.anthropic.com for full hybrid analysis."
            )

        # ── Score cards ──────────────────────────────────────────────────
        score_cols = st.columns(4)
        with score_cols[0]:
            cat = result["defect_category"]
            color = "🟢" if cat == "normal" else "🔴"
            st.metric("Defect", f"{color} {cat}")
        with score_cols[1]:
            st.metric("Severity", f"{result['severity']}/100")
        with score_cols[2]:
            st.metric("Failure Risk", f"{result['failure_risk']}/100")
        with score_cols[3]:
            st.metric("Confidence", f"{result['confidence']:.0%}")

        # ── Cross-validation status ──────────────────────────────────────
        agreement = detail.get("agreement", "unknown")

        if agreement == "strong":
            st.success(
                f"**Cross-validation: STRONG agreement** — Both CNN and Vision AI "
                f"independently identified **{cat}**. High confidence in this finding."
            )
        elif agreement == "partial":
            st.warning(
                f"**Cross-validation: Partial agreement** — "
                f"CNN detected *{detail.get('cnn_category', '?')}*, "
                f"Vision AI detected *{detail.get('vision_category', '?')}*. "
                f"Using Vision AI's classification (better semantic understanding)."
            )
        elif agreement == "disagreement":
            st.error(
                f"**Cross-validation: DISAGREEMENT** — "
                f"CNN detected *{detail.get('cnn_category', '?')}*, "
                f"Vision AI detected *{detail.get('vision_category', '?')}*. "
                f"Manual review recommended."
            )

        # ── Board description ────────────────────────────────────────────
        if detail.get("board_description"):
            st.info(f"**Board context:** {detail['board_description']}")

        # ── All defects found ────────────────────────────────────────────
        all_defects = detail.get("all_defects_found", [])
        if all_defects:
            st.subheader("Defects Found")
            for i, defect in enumerate(all_defects, 1):
                severity_colors = {
                    "low": "🟡", "medium": "🟠", "high": "🔴", "critical": "⛔"
                }
                icon = severity_colors.get(defect.get("severity", ""), "⚪")
                with st.expander(
                    f"{icon} Defect #{i}: {defect['type']} — "
                    f"{defect.get('severity', 'unknown')} severity "
                    f"({defect.get('confidence', 0):.0%} confidence)"
                ):
                    st.write(f"**Description:** {defect.get('description', 'N/A')}")
                    st.write(f"**Location:** {defect.get('location', 'N/A')}")

        # ── Risk breakdown ───────────────────────────────────────────────
        st.subheader("Risk Breakdown")
        breakdown = result["risk_breakdown"]
        risk_cols = st.columns(3)
        with risk_cols[0]:
            val = breakdown["image_severity_contribution"]
            st.metric("Image Severity", f"{val:.1f}%")
        with risk_cols[1]:
            val = breakdown["defect_type_contribution"]
            st.metric("Defect Type Risk", f"{val:.1f}%")
        with risk_cols[2]:
            val = breakdown["metadata_contribution"]
            st.metric("Metadata Factors", f"{val:.1f}%")

        # ── Vision AI reasoning ──────────────────────────────────────────
        if detail.get("vision_reasoning"):
            st.subheader("Vision AI Reasoning")
            st.write(detail["vision_reasoning"])

        # ── Full explanation ─────────────────────────────────────────────
        with st.expander("Full Explanation"):
            st.text(result["explanation"])

        # ── Raw JSON ─────────────────────────────────────────────────────
        with st.expander("Raw JSON Response"):
            st.json(result)

# ── History section ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Inspection History")

try:
    history_resp = requests.get(f"{API_URL}/results?limit=10", timeout=5)
    if history_resp.status_code == 200:
        history = history_resp.json()
        if not history:
            st.info("No past inspections yet. Upload an image to get started.")
        else:
            for item in history:
                detail = item.get("analysis_detail") or {}
                agreement = detail.get("agreement", "")
                badge = {"strong": "✅", "partial": "⚠️", "disagreement": "❌"}.get(agreement, "")
                with st.expander(
                    f"{badge} {item['filename']} — {item['defect_category']} "
                    f"(severity {item['severity']}, risk {item['failure_risk']})"
                ):
                    st.json(item)
    else:
        st.info("Could not load history.")
except requests.ConnectionError:
    st.info("API not connected — history unavailable.")
