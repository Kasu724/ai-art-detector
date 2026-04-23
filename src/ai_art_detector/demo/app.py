"""Streamlit demo for the AI art detector."""

from __future__ import annotations

from html import escape
import os

import streamlit as st

from ai_art_detector.config import load_experiment_config
from ai_art_detector.inference.predictor import InvalidImageError, load_predictor
from ai_art_detector.utils.env import load_project_env

load_project_env()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #151b22;
          --panel: #1d2630;
          --panel-soft: #202b36;
          --field: #172029;
          --text: #e4e9ee;
          --muted: #a9b4bf;
          --border: #35424f;
          --ai: #e06d5f;
          --human: #74c69d;
          --review: #d1a24a;
        }

        html, body, [class*="css"] {
          font-family: "Segoe UI", "Aptos", "Helvetica Neue", Arial, sans-serif;
        }

        body,
        .stApp {
          background: var(--bg);
          color: var(--text);
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
          background: var(--bg);
        }

        .stMarkdown,
        .stText,
        label,
        p,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {
          color: var(--text);
        }

        .block-container {
          max-width: 1040px;
          padding-top: 2.2rem;
          padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
          background: #19212a;
          border-right: 1px solid var(--border);
        }

        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
          color: var(--text);
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stFileUploader"] small,
        [data-testid="InputInstructions"] {
          color: var(--muted);
        }

        div[data-baseweb="input"] {
          background: var(--field);
          border-color: var(--border);
        }

        div[data-baseweb="input"] input {
          color: var(--text);
          -webkit-text-fill-color: var(--text);
        }

        div[data-baseweb="input"] input::placeholder {
          color: var(--muted);
          opacity: 1;
        }

        div[data-testid="stFileUploaderDropzone"] {
          background: var(--panel);
          border-color: var(--border);
        }

        div[data-testid="stFileUploaderDropzone"] * {
          color: var(--text);
        }

        button {
          color: var(--text);
          background: var(--panel-soft);
          border-color: var(--border);
        }

        .page-header {
          padding-bottom: 1rem;
          margin-bottom: 1.25rem;
          border-bottom: 1px solid var(--border);
        }

        .page-header h1 {
          margin: 0 0 0.35rem;
          font-size: 2.1rem;
          line-height: 1.1;
          letter-spacing: -0.02em;
          font-weight: 700;
        }

        .page-header p {
          max-width: 760px;
          margin: 0;
          color: var(--muted);
          line-height: 1.55;
          font-size: 0.98rem;
        }

        .section-label {
          margin: 1.25rem 0 0.55rem;
          color: var(--muted);
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
        }

        .empty-state,
        .result-card,
        .note-card {
          border: 1px solid var(--border);
          border-radius: 12px;
          background: var(--panel);
        }

        .empty-state {
          padding: 1.15rem 1.25rem;
          color: var(--muted);
          line-height: 1.55;
        }

        .result-card {
          padding: 1.1rem;
        }

        .verdict-row {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          align-items: flex-start;
          padding-bottom: 0.9rem;
          border-bottom: 1px solid var(--border);
        }

        .verdict-row h2 {
          margin: 0.2rem 0 0;
          font-size: 1.65rem;
          line-height: 1.12;
          font-weight: 700;
        }

        .small-label {
          color: var(--muted);
          font-size: 0.76rem;
          font-weight: 700;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .badge {
          display: inline-block;
          padding: 0.32rem 0.55rem;
          border-radius: 999px;
          border: 1px solid currentColor;
          font-size: 0.76rem;
          font-weight: 700;
          white-space: nowrap;
        }

        .stats {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.75rem;
          margin-top: 1rem;
        }

        .stat {
          padding: 0.75rem;
          border: 1px solid var(--border);
          border-radius: 10px;
          background: var(--panel-soft);
        }

        .stat-value {
          margin-top: 0.2rem;
          font-size: 1.25rem;
          font-weight: 700;
          color: var(--text);
        }

        .meter {
          margin-top: 1rem;
        }

        .meter-row {
          display: flex;
          justify-content: space-between;
          color: var(--muted);
          font-size: 0.88rem;
          font-weight: 600;
        }

        .track {
          height: 10px;
          margin-top: 0.35rem;
          overflow: hidden;
          border-radius: 999px;
          background: #2d3946;
        }

        .fill {
          height: 100%;
          border-radius: 999px;
        }

        .note-card {
          margin-top: 0.85rem;
          padding: 0.9rem 1rem;
          color: var(--muted);
          line-height: 1.5;
          font-size: 0.92rem;
        }

        .note-card strong {
          color: var(--text);
        }

        img {
          border-radius: 10px;
          border: 1px solid var(--border);
          background: var(--panel);
        }

        @media (max-width: 800px) {
          .stats {
            grid-template-columns: 1fr;
          }

          .verdict-row {
            display: block;
          }

          .badge {
            margin-top: 0.65rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _width(value: float) -> float:
    return max(0.0, min(100.0, value * 100.0))


def _moderation_meta(probability_ai: float, threshold: float) -> dict[str, str]:
    if probability_ai >= threshold:
        return {
            "badge": "Flag",
            "title": "Likely AI-generated",
            "color": "#e06d5f",
            "copy": "This image is above the active AI threshold. Treat this as a review flag, not proof.",
        }
    if probability_ai >= max(threshold - 0.12, 0.0):
        return {
            "badge": "Review",
            "title": "Borderline",
            "color": "#d1a24a",
            "copy": "This image is close to the threshold. It should be reviewed manually.",
        }
    return {
        "badge": "Pass",
        "title": "Likely human-made",
        "color": "#74c69d",
        "copy": "This image is below the active AI threshold. That does not prove authorship.",
    }


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
    st.set_page_config(page_title="AI Art Detector", page_icon="AI", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class="page-header">
          <h1>AI Art Detector</h1>
          <p>
            Upload an artwork image to estimate whether it looks AI-generated or human-made.
            The model output should be used as a moderation signal, not a final source-of-truth.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        st.caption("Loaded from `.env`. Override here if needed.")
        config_path = st.text_input("Config path", value=os.getenv("AIAD_CONFIG_PATH", "configs/experiment.yaml"))
        checkpoint_path = st.text_input("Checkpoint path", value=os.getenv("AIAD_MODEL_PATH", ""))
        metrics_path = st.text_input("Metrics path", value=os.getenv("AIAD_METRICS_PATH", ""))
        onnx_path = st.text_input("ONNX path", value=os.getenv("AIAD_ONNX_PATH", ""))
        threshold_override = st.text_input("Threshold", value=os.getenv("AIAD_THRESHOLD", ""))
        device = st.text_input("Device", value=os.getenv("AIAD_DEVICE", "auto"))

    st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    if upload is None:
        st.markdown(
            """
            <div class="empty-state">
              Choose a PNG, JPG, JPEG, or WebP image. Results will appear here after inference.
            </div>
            """,
            unsafe_allow_html=True,
        )
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

    meta = _moderation_meta(result.probability_ai, result.threshold)
    ai_width = _width(result.probability_ai)
    human_width = _width(result.probabilities["human"])

    left, right = st.columns([1.05, 1.0], gap="large")
    with left:
        st.markdown('<div class="section-label">Image</div>', unsafe_allow_html=True)
        st.image(upload.getvalue(), caption=escape(upload.name), use_container_width=True)

    with right:
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-card">
              <div class="verdict-row">
                <div>
                  <div class="small-label">Decision</div>
                  <h2 style="color:{meta['color']};">{meta['title']}</h2>
                </div>
                <span class="badge" style="color:{meta['color']};">{meta['badge']}</span>
              </div>

              <div class="stats">
                <div class="stat">
                  <div class="small-label">AI score</div>
                  <div class="stat-value">{_percent(result.probability_ai)}</div>
                </div>
                <div class="stat">
                  <div class="small-label">Confidence</div>
                  <div class="stat-value">{_percent(result.confidence)}</div>
                </div>
                <div class="stat">
                  <div class="small-label">Threshold</div>
                  <div class="stat-value">{result.threshold:.2f}</div>
                </div>
              </div>

              <div class="meter">
                <div class="meter-row"><span>AI-generated</span><span>{_percent(result.probability_ai)}</span></div>
                <div class="track"><div class="fill" style="width:{ai_width:.1f}%; background:#e06d5f;"></div></div>
              </div>

              <div class="meter">
                <div class="meter-row"><span>Human-made</span><span>{_percent(result.probabilities["human"])}</span></div>
                <div class="track"><div class="fill" style="width:{human_width:.1f}%; background:#74c69d;"></div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="note-card">
              <strong>Interpretation:</strong> {meta['copy']}
            </div>
            <div class="note-card">
              <strong>Model:</strong> {escape(result.model_name)} via {escape(result.backend)}.
              Calibration is {"enabled" if result.calibrated else "disabled"}.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
