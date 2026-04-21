"""Streamlit demo for the AI art detector."""

from __future__ import annotations

import os

import streamlit as st

from ai_art_detector.config import load_experiment_config
from ai_art_detector.inference.predictor import InvalidImageError, load_predictor


@st.cache_resource(show_spinner=False)
def _load_predictor_cached(
    config_path: str,
    checkpoint_path: str | None,
    metrics_path: str | None,
    onnx_path: str | None,
    threshold: float | None,
    device: str,
):
    config = load_experiment_config(config_path)
    return load_predictor(
        config=config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        onnx_path=onnx_path,
        threshold=threshold,
        device=device,
    )


def main() -> None:
    st.set_page_config(page_title="AI Art Detector", page_icon="🖼️", layout="wide")
    st.markdown(
        """
        <style>
        .hero {padding: 1.25rem 1.5rem; border-radius: 18px; background: linear-gradient(135deg, #f5efe6 0%, #dde8f6 100%);}
        .result-card {padding: 1rem 1.25rem; border-radius: 14px; border: 1px solid rgba(0,0,0,0.08);}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <h1 style="margin-bottom:0.25rem;">AI Art Detector</h1>
          <p style="margin:0;">Upload an image to estimate whether it looks more like AI-generated artwork or human-made artwork. Treat the output as model evidence, not provenance truth.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model Settings")
        config_path = st.text_input("Config Path", value=os.getenv("AIAD_CONFIG_PATH", "configs/experiment.yaml"))
        checkpoint_path = st.text_input("Checkpoint Path", value=os.getenv("AIAD_MODEL_PATH", ""))
        metrics_path = st.text_input("Metrics Path", value=os.getenv("AIAD_METRICS_PATH", ""))
        onnx_path = st.text_input("ONNX Path", value=os.getenv("AIAD_ONNX_PATH", ""))
        device = st.text_input("Device", value=os.getenv("AIAD_DEVICE", "auto"))
        threshold_override = st.text_input("Threshold Override", value=os.getenv("AIAD_THRESHOLD", ""))

    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if upload is None:
        st.info("Set a checkpoint or ONNX path in the sidebar, then upload an image.")
        return

    try:
        predictor = _load_predictor_cached(
            config_path=config_path,
            checkpoint_path=checkpoint_path or None,
            metrics_path=metrics_path or None,
            onnx_path=onnx_path or None,
            threshold=float(threshold_override) if threshold_override else None,
            device=device,
        )
        result = predictor.predict_bytes(upload.getvalue())
    except (ValueError, ModuleNotFoundError) as exc:
        st.error(str(exc))
        return
    except InvalidImageError as exc:
        st.error(str(exc))
        return

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.image(upload.getvalue(), caption=upload.name, use_container_width=True)
    with right:
        label_color = "#c96c20" if result.predicted_label == "ai" else "#256f3a"
        st.markdown(
            f"""
            <div class="result-card">
              <p style="margin:0;color:#6b7280;font-size:0.9rem;">Predicted Label</p>
              <h2 style="margin:0.15rem 0 0.5rem;color:{label_color};">{result.predicted_label.upper()}</h2>
              <p style="margin:0;"><strong>Confidence:</strong> {result.confidence:.3f}</p>
              <p style="margin:0.25rem 0 0;"><strong>P(AI):</strong> {result.probability_ai:.3f}</p>
              <p style="margin:0.25rem 0 0;"><strong>Threshold:</strong> {result.threshold:.3f}</p>
              <p style="margin:0.25rem 0 0;"><strong>Backend:</strong> {result.backend}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(result.probability_ai, 0.0), 1.0), text=f"P(AI) = {result.probability_ai:.3f}")
        st.progress(min(max(result.probabilities["human"], 0.0), 1.0), text=f"P(Human) = {result.probabilities['human']:.3f}")
        st.markdown(
            """
            **Interpretation**

            The model outputs a probability-like score for the AI class and then applies a decision threshold.
            A high score is not proof of provenance; it is evidence relative to the training data distribution.
            """
        )


if __name__ == "__main__":
    main()
