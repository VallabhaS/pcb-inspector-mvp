"""
Streamlit frontend for PCB Inspector MVP.

Provides:
  - Image upload
  - Optional metadata input
  - Analysis trigger
  - Results display with heatmap, scores, and explanation
  - History of past inspections
"""

import sys
from pathlib import Path

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

st.title("PCB Inspector MVP")
st.caption(
    "Upload a PCB / chip / wafer inspection image to detect defects, "
    "assess severity, and estimate failure risk."
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

# ── Main: image upload ──────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload inspection image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
)

col_img, col_result = st.columns([1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col_img:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # Send to FastAPI backend
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {
                    "component_age_years": component_age,
                    "operating_temp_celsius": operating_temp,
                    "layer_count": layer_count,
                    "is_lead_free": is_lead_free,
                    "environment": environment,
                }
                response = requests.post(f"{API_URL}/analyze", files=files, data=data, timeout=60)
                response.raise_for_status()
                result = response.json()

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the API. "
                    "Make sure the FastAPI backend is running on "
                    f"{API_URL} (run `bash run.sh` or start it manually)."
                )
                st.stop()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # ── Display results ──────────────────────────────────────────────

        with col_result:
            st.subheader("Analysis Results")

            # Score cards
            score_cols = st.columns(3)
            with score_cols[0]:
                st.metric("Severity", f"{result['severity']}/100")
            with score_cols[1]:
                st.metric("Failure Risk", f"{result['failure_risk']}/100")
            with score_cols[2]:
                st.metric("Confidence", f"{result['confidence']:.0%}")

            # Defect category
            cat = result["defect_category"]
            if cat == "normal":
                st.success(f"Defect Category: **{cat}**")
            else:
                st.warning(f"Defect Category: **{cat}**")

            # Heatmap
            if result.get("heatmap_path"):
                st.subheader("Suspicious Region Heatmap")
                try:
                    heatmap_resp = requests.get(
                        f"{API_URL}/heatmap/{result['result_id']}", timeout=10
                    )
                    if heatmap_resp.status_code == 200:
                        from io import BytesIO
                        heatmap_img = Image.open(BytesIO(heatmap_resp.content))
                        st.image(heatmap_img, use_container_width=True)
                except Exception:
                    st.info("Heatmap generated but could not be loaded for display.")

            # Explanation
            st.subheader("Explanation")
            st.text(result["explanation"])

            # Risk breakdown
            st.subheader("Risk Breakdown")
            breakdown = result["risk_breakdown"]
            st.bar_chart(
                {
                    "Image severity": breakdown["image_severity_contribution"],
                    "Defect type": breakdown["defect_type_contribution"],
                    "Metadata": breakdown["metadata_contribution"],
                }
            )

            # Raw JSON (collapsed)
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
                with st.expander(
                    f"{item['filename']} — {item['defect_category']} "
                    f"(severity {item['severity']}, risk {item['failure_risk']})"
                ):
                    st.json(item)
    else:
        st.info("Could not load history.")
except requests.ConnectionError:
    st.info("API not connected — history unavailable.")
