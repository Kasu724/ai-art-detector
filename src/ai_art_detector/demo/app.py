"""Streamlit demo for the AI art detector."""

from __future__ import annotations

from html import escape
import os

import streamlit as st

from ai_art_detector.config import load_experiment_config
from ai_art_detector.inference.predictor import InvalidImageError, load_predictor
from ai_art_detector.utils.env import load_project_env

load_project_env()


DARK_THEME = {
    "scheme": "dark",
    "bg": "#131a22",
    "surface": "#18222c",
    "panel": "#202b36",
    "panel_soft": "#26323e",
    "field": "#151f29",
    "text": "#e6edf3",
    "muted": "#a7b4c1",
    "border": "#344352",
    "sidebar": "#161f28",
    "track": "#2e3b48",
    "ai": "#e06d5f",
    "human": "#74c69d",
    "review": "#d1a24a",
    "shadow": "rgba(6, 10, 15, 0.28)",
}


def _theme_css_variables() -> str:
    return "\n".join(f"          --{key.replace('_', '-')}: {value};" for key, value in DARK_THEME.items())


def _inject_styles() -> None:
    css_vars = _theme_css_variables()
    st.markdown(
        f"""
        <style>
        :root {{
{css_vars}
          color-scheme: var(--scheme);
        }}

        html, body, [class*="css"] {{
          font-family: "Segoe UI", "Aptos", "Helvetica Neue", Arial, sans-serif;
        }}

        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewBlockContainer"] {{
          background: var(--bg) !important;
          color: var(--text) !important;
        }}

        [data-testid="stHeader"] {{
          background: transparent !important;
          color: var(--text) !important;
          pointer-events: none !important;
        }}

        [data-testid="stToolbar"] {{
          background: transparent !important;
          pointer-events: none !important;
        }}

        [data-testid="stToolbar"] > div > div:not(:first-child) {{
          display: none !important;
        }}

        [data-testid="stExpandSidebarButton"] {{
          display: inline-flex !important;
          visibility: visible !important;
          opacity: 1 !important;
          align-items: center;
          justify-content: center;
          width: 2.1rem !important;
          min-width: 2.1rem !important;
          height: 2.1rem !important;
          padding: 0 !important;
          color: var(--text) !important;
          background: var(--panel-soft) !important;
          border: 1px solid var(--border) !important;
          border-radius: 10px !important;
          box-shadow: 0 10px 24px var(--shadow);
          pointer-events: auto !important;
        }}

        [data-testid="stExpandSidebarButton"]:hover,
        [data-testid="stSidebarCollapseButton"] button:hover {{
          background: var(--panel) !important;
          border-color: var(--muted) !important;
        }}

        [data-testid="stExpandSidebarButton"] *,
        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stSidebarCollapseButton"] svg {{
          color: var(--text) !important;
          fill: var(--text) !important;
        }}

        [data-testid="stDecoration"],
        #MainMenu {{
          display: none !important;
          height: 0 !important;
          min-height: 0 !important;
          visibility: hidden !important;
        }}

        .stMarkdown,
        .stText,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] *,
        label,
        p,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        span {{
          color: var(--text);
        }}

        code,
        pre,
        [data-testid="stMarkdownContainer"] code,
        [data-testid="stCaptionContainer"] code {{
          color: var(--text) !important;
          background: var(--panel-soft) !important;
          border: 1px solid var(--border);
          border-radius: 6px;
        }}

        code,
        [data-testid="stMarkdownContainer"] code,
        [data-testid="stCaptionContainer"] code {{
          padding: 0.08rem 0.28rem;
        }}

        .block-container {{
          max-width: 1080px;
          padding-top: 0.85rem;
          padding-bottom: 3rem;
        }}

        section[data-testid="stSidebar"] {{
          background: var(--sidebar) !important;
          border-right: 1px solid var(--border);
          overflow: hidden !important;
        }}

        section[data-testid="stSidebar"] * {{
          color: var(--text);
        }}

        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
          background: var(--sidebar) !important;
          overflow-y: auto !important;
          overflow-x: hidden !important;
          scrollbar-gutter: auto;
          padding-top: 0 !important;
        }}

        section[data-testid="stSidebar"] > div {{
          max-height: 100vh;
        }}

        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
          height: 0 !important;
          min-height: 0 !important;
          margin: 0 !important;
          padding: 0 !important;
          position: relative !important;
          overflow: visible !important;
          z-index: 5;
        }}

        section[data-testid="stSidebar"] [data-testid="stLogoSpacer"] {{
          display: none !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{
          position: absolute !important;
          top: 0.75rem !important;
          right: 0.15rem !important;
          z-index: 10;
        }}

        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {{
          display: inline-flex !important;
          align-items: center !important;
          justify-content: center !important;
          width: 2.1rem !important;
          min-width: 2.1rem !important;
          height: 2.1rem !important;
          padding: 0 !important;
          color: var(--text) !important;
          background: var(--panel-soft) !important;
          border: 1px solid var(--border) !important;
          border-radius: 10px !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
          padding-top: 0.9rem !important;
          padding-bottom: 1.5rem !important;
          margin-top: 0 !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
          padding-top: 0 !important;
          margin-top: 0 !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
          gap: 0.55rem;
        }}

        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stRadio {{
          margin-bottom: 0.15rem;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
          margin-top: 0;
          margin-bottom: 0.45rem;
        }}

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        [data-testid="stFileUploader"] small,
        [data-testid="InputInstructions"],
        [data-testid="InputInstructions"] * {{
          color: var(--muted) !important;
        }}

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="radio"],
        .stTextInput > div > div,
        .stNumberInput > div > div,
        .stSelectbox > div > div {{
          background: var(--field) !important;
          border-color: var(--border) !important;
        }}

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="radio"] span,
        .stTextInput input,
        .stNumberInput input,
        textarea {{
          color: var(--text) !important;
          -webkit-text-fill-color: var(--text) !important;
          background: transparent !important;
        }}

        div[data-baseweb="input"] input::placeholder,
        .stTextInput input::placeholder,
        textarea::placeholder {{
          color: var(--muted) !important;
          opacity: 1;
        }}

        div[role="radiogroup"],
        div[role="radiogroup"] label,
        div[role="radio"] {{
          background: transparent !important;
          color: var(--text) !important;
        }}

        div[data-testid="stFileUploaderDropzone"] {{
          background: var(--panel) !important;
          border-color: var(--border) !important;
          color: var(--text) !important;
        }}

        div[data-testid="stFileUploaderDropzone"] * {{
          color: var(--text) !important;
        }}

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] div {{
          border-color: var(--border) !important;
        }}

        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] [data-testid="InputInstructions"],
        div[data-testid="stFileUploaderDropzone"] [data-testid="InputInstructions"] * {{
          color: var(--muted) !important;
        }}

        div[data-testid="stFileUploaderDropzone"] button {{
          background: var(--field) !important;
          border-color: var(--border) !important;
          color: var(--text) !important;
        }}

        button {{
          color: var(--text) !important;
          background: var(--panel-soft) !important;
          border-color: var(--border) !important;
        }}

        button:hover {{
          border-color: var(--muted) !important;
          color: var(--text) !important;
          background: var(--panel) !important;
        }}

        [data-testid="stAlert"],
        [data-testid="stNotification"],
        [data-testid="stException"] {{
          background: var(--panel) !important;
          color: var(--text) !important;
          border-color: var(--border) !important;
        }}

        [data-testid="stAlert"] *,
        [data-testid="stNotification"] *,
        [data-testid="stException"] * {{
          color: var(--text) !important;
        }}

        .page-header {{
          display: flex;
          justify-content: space-between;
          gap: 1.25rem;
          align-items: flex-end;
          padding: 1.25rem;
          margin-bottom: 1.25rem;
          border: 1px solid var(--border);
          border-radius: 16px;
          background: var(--surface);
          box-shadow: 0 18px 42px var(--shadow);
        }}

        .page-header h1 {{
          margin: 0 0 0.4rem;
          font-size: clamp(2rem, 4vw, 3rem);
          line-height: 1.05;
          letter-spacing: -0.035em;
          font-weight: 750;
          color: var(--text);
        }}

        .page-header p {{
          max-width: 720px;
          margin: 0;
          color: var(--muted);
          line-height: 1.55;
          font-size: 0.98rem;
        }}

        .header-chip {{
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          padding: 0.42rem 0.65rem;
          border: 1px solid var(--border);
          border-radius: 999px;
          background: var(--panel);
          color: var(--muted);
          font-size: 0.78rem;
          font-weight: 700;
          white-space: nowrap;
        }}

        .section-label {{
          margin: 1.15rem 0 0.55rem;
          color: var(--muted);
          font-size: 0.77rem;
          font-weight: 750;
          letter-spacing: 0.07em;
          text-transform: uppercase;
        }}

        .empty-state,
        .result-card,
        .note-card {{
          border: 1px solid var(--border);
          border-radius: 14px;
          background: var(--surface);
          box-shadow: 0 14px 34px var(--shadow);
        }}

        .empty-state {{
          padding: 1.15rem 1.25rem;
          color: var(--muted);
          line-height: 1.55;
        }}

        .result-card {{
          padding: 1.1rem;
        }}

        .verdict-row {{
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          align-items: flex-start;
          padding-bottom: 0.95rem;
          border-bottom: 1px solid var(--border);
        }}

        .verdict-row h2 {{
          margin: 0.18rem 0 0;
          font-size: 1.62rem;
          line-height: 1.12;
          font-weight: 750;
        }}

        .small-label {{
          color: var(--muted);
          font-size: 0.74rem;
          font-weight: 750;
          letter-spacing: 0.06em;
          text-transform: uppercase;
        }}

        .badge {{
          display: inline-block;
          padding: 0.34rem 0.6rem;
          border-radius: 999px;
          border: 1px solid currentColor;
          background: color-mix(in srgb, currentColor 12%, transparent);
          font-size: 0.75rem;
          font-weight: 750;
          white-space: nowrap;
        }}

        .stats {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.75rem;
          margin-top: 1rem;
        }}

        .stat {{
          padding: 0.78rem;
          border: 1px solid var(--border);
          border-radius: 12px;
          background: var(--panel);
        }}

        .stat-value {{
          margin-top: 0.2rem;
          font-size: 1.25rem;
          font-weight: 750;
          color: var(--text);
        }}

        .meter {{
          margin-top: 1rem;
        }}

        .meter-row {{
          display: flex;
          justify-content: space-between;
          color: var(--muted);
          font-size: 0.88rem;
          font-weight: 650;
        }}

        .meter-row span {{
          color: var(--muted);
        }}

        .track {{
          height: 10px;
          margin-top: 0.35rem;
          overflow: hidden;
          border-radius: 999px;
          background: var(--track);
        }}

        .fill {{
          height: 100%;
          border-radius: 999px;
        }}

        .note-card {{
          margin-top: 0.85rem;
          padding: 0.9rem 1rem;
          color: var(--muted);
          line-height: 1.5;
          font-size: 0.92rem;
        }}

        .note-card,
        .note-card span {{
          color: var(--muted);
        }}

        .note-card strong {{
          color: var(--text);
        }}

        img {{
          border-radius: 14px;
          border: 1px solid var(--border);
          background: var(--panel);
        }}

        @media (max-width: 850px) {{
          .page-header {{
            display: block;
          }}

          .header-chip {{
            margin-top: 0.9rem;
          }}

          .stats {{
            grid-template-columns: 1fr;
          }}

          .verdict-row {{
            display: block;
          }}

          .badge {{
            margin-top: 0.65rem;
          }}
        }}
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
            "color": "var(--ai)",
            "copy": "This image is above the active AI threshold. Treat this as a review flag, not proof.",
        }
    if probability_ai >= max(threshold - 0.12, 0.0):
        return {
            "badge": "Review",
            "title": "Borderline",
            "color": "var(--review)",
            "copy": "This image is close to the threshold. It should be reviewed manually.",
        }
    return {
        "badge": "Pass",
        "title": "Likely human-made",
        "color": "var(--human)",
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

    with st.sidebar:
        st.header("Settings")
        st.caption("Loaded from `.env`. Override here if needed.")
        config_path = st.text_input("Config path", value=os.getenv("AIAD_CONFIG_PATH", "configs/experiment.yaml"))
        checkpoint_path = st.text_input("Checkpoint path", value=os.getenv("AIAD_MODEL_PATH", ""))
        metrics_path = st.text_input("Metrics path", value=os.getenv("AIAD_METRICS_PATH", ""))
        onnx_path = st.text_input("ONNX path", value=os.getenv("AIAD_ONNX_PATH", ""))
        threshold_override = st.text_input("Threshold", value=os.getenv("AIAD_THRESHOLD", ""))
        device = st.text_input("Device", value=os.getenv("AIAD_DEVICE", "auto"))

    _inject_styles()

    st.markdown(
        """
        <div class="page-header">
          <div>
            <h1>AI Art Detector</h1>
            <p>
              Upload artwork and get an AI-likelihood score with a moderation-oriented decision.
              Use the result as one review signal, not as proof of provenance.
            </p>
          </div>
          <div class="header-chip">v4 recall model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                <div class="track"><div class="fill" style="width:{ai_width:.1f}%; background:var(--ai);"></div></div>
              </div>

              <div class="meter">
                <div class="meter-row"><span>Human-made</span><span>{_percent(result.probabilities["human"])}</span></div>
                <div class="track"><div class="fill" style="width:{human_width:.1f}%; background:var(--human);"></div></div>
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
