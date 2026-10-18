"""
app.py — Streamlit Application for Toxic Comment Classification
================================================================
Member 5: Deployment + UI

Integrates all project modules:
  • Member 1: Data preprocessing (toxic_preprocessing.py)
  • Member 2: DistilBERT model + thresholds
  • Member 3: Fairness audit results
  • Member 4: Calibration, SHAP explanations
  • Member 5: This application

Run:  streamlit run app.py
"""

import json, os, sys, time, warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="ToxiGuard — Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────

LABELS = ["toxic", "obscene", "threat", "insult", "identity_hate", "sexual_explicit"]

LABEL_ICONS = {
    "toxic": "☣️", "obscene": "🤬", "threat": "⚠️",
    "insult": "💢", "identity_hate": "🚫", "sexual_explicit": "🔞",
}

LABEL_COLORS = {
    "toxic": "#ef4444", "obscene": "#f97316", "threat": "#dc2626",
    "insult": "#eab308", "identity_hate": "#a855f7", "sexual_explicit": "#ec4899",
}

IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender",
    "heterosexual", "homosexual_gay_or_lesbian",
    "bisexual", "other_sexual_orientation",
    "christian", "jewish", "muslim", "hindu",
    "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino",
    "other_race_or_ethnicity", "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness", "other_disability",
]

def get_current_thresholds() -> Dict[str, float]:
    _base_dir = Path(__file__).resolve().parent
    _thr_candidates = [Path("thresholds.json"), _base_dir / "thresholds.json"]
    for _tp in _thr_candidates:
        if _tp.exists():
            try:
                with open(_tp) as f:
                    return json.load(f)
            except Exception:
                pass
    return {
        "toxic": 0.050, "obscene": 0.900, "threat": 0.863,
        "insult": 0.789, "identity_hate": 0.821, "sexual_explicit": 0.920,
    }

THRESHOLDS = get_current_thresholds()


# ── Custom CSS ───────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Result cards */
    .result-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 78, 59, 0.2));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }
    .result-toxic {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(127, 29, 29, 0.2));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }
    .result-uncertain {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.15), rgba(113, 63, 18, 0.2));
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }

    /* Probability bar */
    .prob-bar-container {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    .prob-bar {
        height: 8px;
        border-radius: 4px;
        transition: width 0.6s ease;
        margin-top: 6px;
    }
    .prob-label {
        font-size: 0.9rem;
        color: #e2e8f0;
        font-weight: 500;
    }
    .prob-value {
        font-size: 0.85rem;
        color: #94a3b8;
        float: right;
    }

    /* Hero section */
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 8px;
        font-weight: 400;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px 4px;
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-safe {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-warning {
        background: rgba(234, 179, 8, 0.2);
        color: #fde68a;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    /* Info panel */
    .info-panel {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
    }

    /* Review queue item */
    .review-item {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(234, 179, 8, 0.2);
        border-left: 4px solid #eab308;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Text input styling */
    .stTextArea textarea {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 16px !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #c7d2fe !important;
    }

    /* Plotly chart background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6b7280; }
    </style>
    """, unsafe_allow_html=True)


# ── Cached model loading ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load the model once and cache it across sessions."""
    from predictor import ToxicityPredictor
    return ToxicityPredictor()


@st.cache_data(show_spinner=False)
def load_test_predictions():
    """Load pre-computed test predictions for dashboards."""
    search_paths = [
        "test_predictions.parquet",
        "member2_outputs/test_predictions.parquet",
        "processed_data/processed_data/test.parquet",
    ]
    for p in search_paths:
        if Path(p).exists():
            return pd.read_parquet(p)
    return None


@st.cache_data(show_spinner=False)
def load_fairness_data():
    """Load pre-computed fairness audit CSVs."""
    search = [
        "member3_outputs/group_rates_toxic.csv",
        "group_rates_toxic.csv",
    ]
    for p in search:
        if Path(p).exists():
            return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_calibration_results():
    """Load calibration comparison table."""
    search = [
        "member4_outputs/calibration_before_after.csv",
        "calibration_before_after.csv",
    ]
    for p in search:
        if Path(p).exists():
            return pd.read_csv(p)
    return None


# ── Helper: plotly theme ─────────────────────────────────────────────

def dark_plotly_layout(fig, title="", height=400):
    """Apply consistent dark theme to plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#e2e8f0", family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.3)",
        font=dict(color="#94a3b8", family="Inter"),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.5)",
            bordercolor="rgba(148, 163, 184, 0.1)",
            font=dict(color="#e2e8f0"),
        ),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.08)", zerolinecolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.08)", zerolinecolor="rgba(148,163,184,0.1)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════

def page_predict():
    """Main prediction interface."""
    st.markdown('<p class="hero-title">🛡️ ToxiGuard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">AI-Powered Toxic Comment Detection · DistilBERT Multi-Label Classifier</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Model loading
    with st.spinner("🔧 Loading model..."):
        predictor = load_predictor()

    if predictor.is_demo:
        st.warning(
            "⚡ **Demo Mode**: No fine-tuned model found. Predictions will use an untrained DistilBERT head. "
            "Place your `model/` directory (with `config.json` and `model.safetensors`) in the project root to use the real model."
        )

    # Input area
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area(
            "Enter a comment to analyse",
            height=140,
            placeholder="Type or paste a comment here...\n\nExamples:\n• 'This article is very informative, thank you!'\n• 'You are an absolute idiot and should be banned'",
            key="comment_input",
        )
    current_thresholds = get_current_thresholds()
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-panel">
            <p style="color: #c7d2fe; font-weight: 600; margin-bottom: 8px;">📋 Label Thresholds</p>
        """ + "".join([
            f'<p style="color: #94a3b8; font-size: 0.8rem; margin: 2px 0;">'
            f'{LABEL_ICONS[l]} {l}: {current_thresholds.get(l, 0.5):.3f}</p>'
            for l in LABELS
        ]) + "</div>", unsafe_allow_html=True)

    analyze_btn = st.button("🔍 Analyse Comment", use_container_width=True, type="primary")

    if analyze_btn and text_input.strip():
        start_time = time.time()
        result = predictor.predict(text_input)
        elapsed = time.time() - start_time

        # Store in session for review queue
        if "predictions_history" not in st.session_state:
            st.session_state.predictions_history = []
        st.session_state.predictions_history.append({
            **result,
            "timestamp": datetime.now().isoformat(),
            "inference_ms": round(elapsed * 1000, 1),
        })

        st.markdown("---")

        # ── Overall verdict ──
        triggered = result["labels_triggered"]
        is_uncertain = result["is_uncertain"]

        if not triggered and not is_uncertain:
            st.markdown("""
            <div class="result-safe">
                <h2 style="color: #6ee7b7; margin:0;">✅ Comment appears safe</h2>
                <p style="color: #a7f3d0; margin-top:8px;">No toxicity labels were triggered above the decision thresholds.</p>
            </div>
            """, unsafe_allow_html=True)
        elif is_uncertain and not triggered:
            st.markdown("""
            <div class="result-uncertain">
                <h2 style="color: #fde68a; margin:0;">🔶 Uncertain — Human Review Recommended</h2>
                <p style="color: #fef3c7; margin-top:8px;">Some probabilities are close to their thresholds. A human reviewer should verify this case.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            labels_html = " ".join([
                f'<span class="badge badge-danger">{LABEL_ICONS[l]} {l}</span>'
                for l in triggered
            ])
            uncertain_html = ""
            if is_uncertain:
                uncertain_html = '<p style="color: #fde68a; margin-top:8px;">⚠️ Some labels are uncertain — human review recommended.</p>'
            st.markdown(f"""
            <div class="result-toxic">
                <h2 style="color: #fca5a5; margin:0;">🚨 Toxic Content Detected</h2>
                <div style="margin-top:12px;">{labels_html}</div>
                {uncertain_html}
            </div>
            """, unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown('<p class="section-header">📊 Label Probabilities</p>', unsafe_allow_html=True)

        probs = result["probabilities"]
        for lab in LABELS:
            p = probs[lab]
            thr = current_thresholds.get(lab, 0.5)
            pct = p * 100
            color = LABEL_COLORS[lab]
            status = "🔴" if p >= thr else ("🟡" if abs(p - thr) < 0.10 else "🟢")

            st.markdown(f"""
            <div class="prob-bar-container">
                <span class="prob-label">{status} {LABEL_ICONS[lab]} {lab}</span>
                <span class="prob-value">{p:.4f} (threshold: {thr:.3f})</span>
                <div style="background: rgba(148,163,184,0.15); border-radius:4px; height:8px; margin-top:8px; position: relative;">
                    <div class="prob-bar" style="width:{pct}%; background: linear-gradient(90deg, {color}88, {color});"></div>
                    <div style="position:absolute; top:-4px; left:{thr*100}%; width:2px; height:16px; background:#fbbf24; border-radius:1px;" title="Threshold: {thr:.3f}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Explanation ──
        st.markdown('<p class="section-header">🔬 Explanation — Token Importance</p>', unsafe_allow_html=True)

        try:
            from explainer import get_attention_explanation
            explanation = get_attention_explanation(
                predictor.model, predictor.tokenizer,
                result["cleaned_text"], predictor.device
            )
            st.markdown(
                f'<div class="info-panel">{explanation["html"]}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("🔍 Top Contributing Tokens"):
                for token, score in explanation["top_tokens"]:
                    bar_w = score * 100
                    st.markdown(
                        f'<div style="display:flex; align-items:center; margin:4px 0;">'
                        f'<span style="color:#e2e8f0; width:140px; font-weight:500;">{token}</span>'
                        f'<div style="flex:1; background:rgba(148,163,184,0.1); border-radius:4px; height:6px; margin:0 12px;">'
                        f'<div style="width:{bar_w}%; background:linear-gradient(90deg,#818cf8,#c084fc); height:6px; border-radius:4px;"></div>'
                        f'</div>'
                        f'<span style="color:#94a3b8; font-size:0.85rem;">{score:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            from explainer import get_simple_word_importance
            simple_html = get_simple_word_importance(result["cleaned_text"], probs)
            st.markdown(f'<div class="info-panel">{simple_html}</div>', unsafe_allow_html=True)
            st.caption(f"Using keyword-based highlighting (attention extraction error: {e})")

        # ── Uncertainty detail ──
        if result["review_reasons"]:
            st.markdown('<p class="section-header">⚠️ Uncertainty Details</p>', unsafe_allow_html=True)
            for reason in result["review_reasons"]:
                st.markdown(f"""
                <div class="review-item">
                    <p style="color: #fde68a; margin:0;">{reason}</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Metadata ──
        with st.expander("📋 Full Prediction Details"):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Inference Time", f"{elapsed*1000:.1f} ms")
            col_b.metric("Max Probability", f"{result['max_probability']:.4f}")
            col_c.metric("Labels Triggered", str(len(triggered)))

            st.json(result)

    elif analyze_btn:
        st.warning("Please enter a comment to analyse.")


# ══════════════════════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════

def page_performance():
    st.markdown('<p class="hero-title">📈 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">DistilBERT Evaluation Metrics on the Test Set</p>', unsafe_allow_html=True)
    st.markdown("---")

    test_df = load_test_predictions()

    if test_df is None:
        st.info(
            "📂 No test predictions file found. Place `test_predictions.parquet` in the project directory "
            "to see live metrics. Showing reference metrics from the training notebook."
        )
        _show_reference_metrics()
        return

    # Compute metrics from the predictions file
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    metrics_rows = []
    for lab in LABELS:
        true_col = f"true_{lab}"
        prob_col = f"prob_{lab}"

        if true_col not in test_df.columns or prob_col not in test_df.columns:
            continue

        y = test_df[true_col].to_numpy()
        p = test_df[prob_col].to_numpy()
        valid = np.isfinite(y) & np.isfinite(p)
        y, p = y[valid].astype(int), p[valid]

        thr = THRESHOLDS.get(lab, 0.5)
        pred = (p >= thr).astype(int)

        metrics_rows.append({
            "Label": f"{LABEL_ICONS[lab]} {lab}",
            "Positives": int(y.sum()),
            "Threshold": round(thr, 3),
            "Precision": round(precision_score(y, pred, zero_division=0), 4),
            "Recall": round(recall_score(y, pred, zero_division=0), 4),
            "F1": round(f1_score(y, pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y, p), 4) if y.sum() > 0 else None,
            "AP": round(average_precision_score(y, p), 4) if y.sum() > 0 else None,
        })

    mdf = pd.DataFrame(metrics_rows)

    # ── Top-line metrics ──
    cols = st.columns(4)
    avg_f1 = mdf["F1"].mean()
    avg_auc = mdf["ROC-AUC"].dropna().mean()
    avg_prec = mdf["Precision"].mean()
    avg_recall = mdf["Recall"].mean()

    for col, val, label in zip(
        cols,
        [f"{avg_f1:.3f}", f"{avg_auc:.3f}", f"{avg_prec:.3f}", f"{avg_recall:.3f}"],
        ["Macro F1", "Macro AUC", "Macro Precision", "Macro Recall"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics table ──
    st.markdown('<p class="section-header">Per-Label Metrics</p>', unsafe_allow_html=True)
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    # ── Charts ──
    tab1, tab2, tab3 = st.tabs(["🎯 F1 by Label", "📊 ROC-AUC by Label", "📈 Probability Distributions"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[r["Label"] for r in metrics_rows],
            y=[r["F1"] for r in metrics_rows],
            marker=dict(
                color=list(LABEL_COLORS.values()),
                line=dict(width=0),
            ),
            text=[f'{r["F1"]:.3f}' for r in metrics_rows],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig = dark_plotly_layout(fig, "F1 Score per Label", 450)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[r["Label"] for r in metrics_rows],
            y=[r["ROC-AUC"] or 0 for r in metrics_rows],
            marker=dict(
                color=[LABEL_COLORS.get(l, "#6366f1") for l in LABELS],
                line=dict(width=0),
            ),
            text=[f'{(r["ROC-AUC"] or 0):.3f}' for r in metrics_rows],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig = dark_plotly_layout(fig, "ROC-AUC per Label", 450)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_subplots(rows=2, cols=3, subplot_titles=[
            f"{LABEL_ICONS[l]} {l}" for l in LABELS
        ])
        for i, lab in enumerate(LABELS):
            r, c = divmod(i, 3)
            prob_col = f"prob_{lab}"
            if prob_col in test_df.columns:
                fig.add_trace(go.Histogram(
                    x=test_df[prob_col].dropna(),
                    nbinsx=50,
                    marker_color=LABEL_COLORS[lab],
                    opacity=0.7,
                    showlegend=False,
                ), row=r+1, col=c+1)
                # threshold line
                thr = THRESHOLDS.get(lab, 0.5)
                fig.add_vline(x=thr, line_dash="dash", line_color="#fbbf24",
                              row=r+1, col=c+1)
        fig = dark_plotly_layout(fig, "Probability Distributions (test set)", 600)
        st.plotly_chart(fig, use_container_width=True)

    # ── Calibration ──
    cal_df = load_calibration_results()
    if cal_df is not None:
        st.markdown('<p class="section-header">🌡️ Calibration Results</p>', unsafe_allow_html=True)
        st.dataframe(cal_df, use_container_width=True, hide_index=True)


def _show_reference_metrics():
    """Hardcoded reference metrics from the notebook output, used when parquet is unavailable."""
    ref = pd.DataFrame([
        {"Label": "☣️ toxic",      "Threshold": 0.720, "F1": 0.692, "ROC-AUC": 0.969, "Precision": 0.660, "Recall": 0.728},
        {"Label": "🤬 obscene",    "Threshold": 0.900, "F1": 0.682, "ROC-AUC": 0.987, "Precision": 0.795, "Recall": 0.598},
        {"Label": "⚠️ threat",     "Threshold": 0.863, "F1": 0.548, "ROC-AUC": 0.990, "Precision": 0.719, "Recall": 0.443},
        {"Label": "💢 insult",     "Threshold": 0.789, "F1": 0.639, "ROC-AUC": 0.975, "Precision": 0.690, "Recall": 0.595},
        {"Label": "🚫 identity_hate", "Threshold": 0.821, "F1": 0.543, "ROC-AUC": 0.990, "Precision": 0.715, "Recall": 0.438},
        {"Label": "🔞 sexual_explicit", "Threshold": 0.920, "F1": 0.469, "ROC-AUC": 0.988, "Precision": 0.672, "Recall": 0.360},
    ])

    cols = st.columns(4)
    for col, val, label in zip(
        cols,
        ["0.596", "0.983", "0.709", "0.527"],
        ["Macro F1", "Macro AUC", "Macro Precision", "Macro Recall"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Per-Label Metrics (Reference)</p>', unsafe_allow_html=True)
    st.dataframe(ref, use_container_width=True, hide_index=True)

    # Bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ref["Label"], y=ref["F1"],
        marker=dict(color=list(LABEL_COLORS.values())),
        text=[f'{v:.3f}' for v in ref["F1"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig = dark_plotly_layout(fig, "F1 Score per Label (Reference)", 450)
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE: FAIRNESS AUDIT
# ══════════════════════════════════════════════════════════════════════

def page_fairness():
    st.markdown('<p class="hero-title">⚖️ Fairness Audit</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Identity-Group Bias Analysis · FPR & FNR Disparities</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    test_df = load_test_predictions()

    # ── Try live computation from parquet ──
    if test_df is not None and all(c in test_df.columns for c in ["has_identity_labels"]):
        _fairness_from_parquet(test_df)
    else:
        # ── Try pre-computed CSV ──
        fair_df = load_fairness_data()
        if fair_df is not None:
            _fairness_from_csv(fair_df)
        else:
            _fairness_reference()


def _fairness_from_parquet(test_df):
    """Compute fairness metrics live from the test predictions parquet."""

    label = st.selectbox("Select label", LABELS, index=0)

    sub = test_df[test_df["has_identity_labels"].eq(True)].copy()
    st.markdown(f"""
    <div class="info-panel">
        <p style="color:#e2e8f0;"><strong>Total test comments:</strong> {len(test_df):,}</p>
        <p style="color:#e2e8f0;"><strong>Annotated fairness subset:</strong> {len(sub):,}</p>
    </div>
    """, unsafe_allow_html=True)

    thr = THRESHOLDS.get(label, 0.5)
    true_col = f"true_{label}"
    prob_col = f"prob_{label}"

    rows = []
    available_ids = [c for c in IDENTITY_COLS if c in sub.columns]

    for group in available_ids:
        member = sub[group].ge(0.5).fillna(False)
        g = sub.loc[member]
        if len(g) < 100:
            continue
        y = g[true_col].to_numpy()
        p = g[prob_col].to_numpy()
        valid = np.isfinite(y) & np.isfinite(p)
        y, p = y[valid].astype(int), p[valid]
        pred = p >= thr

        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        fpr = fp / n_neg if n_neg else 0
        fnr = fn / n_pos if n_pos else 0

        rows.append({
            "Group": group.replace("_", " ").title(),
            "N": len(g),
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "FP": fp,
            "FN": fn,
        })

    if not rows:
        st.warning("Not enough data to compute group-level fairness metrics.")
        return

    fdf = pd.DataFrame(rows)

    # Overall rates
    y_all = sub[true_col].to_numpy()
    p_all = sub[prob_col].to_numpy()
    valid = np.isfinite(y_all) & np.isfinite(p_all)
    y_all, p_all = y_all[valid].astype(int), p_all[valid]
    pred_all = p_all >= thr
    overall_fpr = int((pred_all & (y_all == 0)).sum()) / max(int((y_all == 0).sum()), 1)
    overall_fnr = int((~pred_all & (y_all == 1)).sum()) / max(int((y_all == 1).sum()), 1)

    fdf["FPR_diff"] = (fdf["FPR"] - overall_fpr).round(4)
    fdf["FNR_diff"] = (fdf["FNR"] - overall_fnr).round(4)

    # Top metrics
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{overall_fpr:.3f}</div>
        <div class="metric-label">Overall FPR</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{overall_fnr:.3f}</div>
        <div class="metric-label">Overall FNR</div>
    </div>""", unsafe_allow_html=True)
    max_disp = fdf["FPR_diff"].abs().max()
    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{max_disp:.3f}</div>
        <div class="metric-label">Max FPR Disparity</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table
    st.markdown(f'<p class="section-header">Group-Level Rates — {LABEL_ICONS[label]} {label}</p>', unsafe_allow_html=True)
    st.dataframe(
        fdf.sort_values("FPR", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # Charts
    tab1, tab2 = st.tabs(["📊 FPR by Group", "📊 FNR by Group"])

    with tab1:
        fig = go.Figure()
        sorted_fpr = fdf.sort_values("FPR", ascending=True)
        colors = ["#ef4444" if abs(d) > 0.02 else "#6366f1" for d in sorted_fpr["FPR_diff"]]
        fig.add_trace(go.Bar(
            y=sorted_fpr["Group"], x=sorted_fpr["FPR"],
            orientation="h",
            marker_color=colors,
            text=[f'{v:.3f}' for v in sorted_fpr["FPR"]],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig.add_vline(x=overall_fpr, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"Overall: {overall_fpr:.3f}",
                      annotation_font_color="#fbbf24")
        fig = dark_plotly_layout(fig, f"False Positive Rate by Group — {label}", 500)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        sorted_fnr = fdf.sort_values("FNR", ascending=True)
        colors = ["#f97316" if abs(d) > 0.02 else "#6366f1" for d in sorted_fnr["FNR_diff"]]
        fig.add_trace(go.Bar(
            y=sorted_fnr["Group"], x=sorted_fnr["FNR"],
            orientation="h",
            marker_color=colors,
            text=[f'{v:.3f}' for v in sorted_fnr["FNR"]],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig.add_vline(x=overall_fnr, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"Overall: {overall_fnr:.3f}",
                      annotation_font_color="#fbbf24")
        fig = dark_plotly_layout(fig, f"False Negative Rate by Group — {label}", 500)
        st.plotly_chart(fig, use_container_width=True)

    # Disparity heatmap
    st.markdown('<p class="section-header">🗺️ FPR Disparity Heatmap (vs Overall)</p>', unsafe_allow_html=True)
    fig = go.Figure(go.Heatmap(
        z=[fdf.sort_values("Group")["FPR_diff"].values],
        x=fdf.sort_values("Group")["Group"].values,
        y=[label],
        colorscale=[[0, "#10b981"], [0.5, "#1e293b"], [1, "#ef4444"]],
        zmid=0,
        text=[[f'{v:.3f}' for v in fdf.sort_values("Group")["FPR_diff"]]],
        texttemplate="%{text}",
        textfont=dict(color="#e2e8f0"),
    ))
    fig = dark_plotly_layout(fig, "", 200)
    st.plotly_chart(fig, use_container_width=True)


def _fairness_from_csv(fair_df):
    """Display pre-computed fairness CSV."""
    st.markdown('<p class="section-header">Group-Level Fairness Metrics (Pre-computed)</p>', unsafe_allow_html=True)
    st.dataframe(fair_df, use_container_width=True, hide_index=True)

    if "FPR" in fair_df.columns and "group" in fair_df.columns:
        fig = go.Figure()
        sorted_df = fair_df.sort_values("FPR", ascending=True)
        fig.add_trace(go.Bar(
            y=sorted_df["group"], x=sorted_df["FPR"],
            orientation="h",
            marker_color="#6366f1",
        ))
        fig = dark_plotly_layout(fig, "False Positive Rate by Group", 500)
        st.plotly_chart(fig, use_container_width=True)


def _fairness_reference():
    """Show reference fairness data from notebook output."""
    st.info("📂 No fairness data available. Showing reference results from the fairness audit notebook.")

    ref = pd.DataFrame([
        {"Group": "Muslim", "N": 2004, "FPR": 0.1253, "FNR": 0.2529, "FPR_diff": 0.0799},
        {"Group": "Jewish", "N": 1000, "FPR": 0.0909, "FNR": 0.3175, "FPR_diff": 0.0455},
        {"Group": "Black", "N": 2522, "FPR": 0.0857, "FNR": 0.2803, "FPR_diff": 0.0403},
        {"Group": "Homosexual", "N": 1732, "FPR": 0.0801, "FNR": 0.2939, "FPR_diff": 0.0347},
        {"Group": "Female", "N": 6586, "FPR": 0.0541, "FNR": 0.3154, "FPR_diff": 0.0087},
        {"Group": "Male", "N": 5740, "FPR": 0.0503, "FNR": 0.3098, "FPR_diff": 0.0049},
        {"Group": "Christian", "N": 4422, "FPR": 0.0492, "FNR": 0.3001, "FPR_diff": 0.0038},
        {"Group": "White", "N": 3912, "FPR": 0.0467, "FNR": 0.2950, "FPR_diff": 0.0013},
    ])

    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.045</div>
        <div class="metric-label">Overall FPR</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.306</div>
        <div class="metric-label">Overall FNR</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.080</div>
        <div class="metric-label">Max FPR Disparity</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(ref, use_container_width=True, hide_index=True)

    fig = go.Figure()
    sorted_df = ref.sort_values("FPR", ascending=True)
    colors = ["#ef4444" if d > 0.02 else "#6366f1" for d in sorted_df["FPR_diff"]]
    fig.add_trace(go.Bar(
        y=sorted_df["Group"], x=sorted_df["FPR"],
        orientation="h",
        marker_color=colors,
        text=[f'{v:.3f}' for v in sorted_df["FPR"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig.add_vline(x=0.045, line_dash="dash", line_color="#fbbf24",
                  annotation_text="Overall: 0.045", annotation_font_color="#fbbf24")
    fig = dark_plotly_layout(fig, "FPR by Identity Group — toxic (Reference)", 450)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE: REVIEW QUEUE
# ══════════════════════════════════════════════════════════════════════

def page_review():
    st.markdown('<p class="hero-title">📋 Review Queue</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Uncertain Predictions Flagged for Human Review</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    history = st.session_state.get("predictions_history", [])

    if not history:
        st.markdown("""
        <div class="info-panel" style="text-align:center;">
            <p style="color:#94a3b8; font-size:1.2rem;">🔍 No predictions yet</p>
            <p style="color:#64748b;">Go to the <strong>Predict</strong> page and analyse some comments.<br>
            Uncertain cases will appear here for review.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Separate uncertain vs confident
    uncertain = [h for h in history if h["is_uncertain"]]
    confident = [h for h in history if not h["is_uncertain"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(history)}</div>
        <div class="metric-label">Total Predictions</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(uncertain)}</div>
        <div class="metric-label">Flagged for Review</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{sum(1 for h in history if h['labels_triggered'])}</div>
        <div class="metric-label">Toxic Detected</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(confident)}</div>
        <div class="metric-label">Confident</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Uncertain items ──
    if uncertain:
        st.markdown('<p class="section-header">⚠️ Cases Requiring Review</p>', unsafe_allow_html=True)

        for i, item in enumerate(reversed(uncertain)):
            triggered = item["labels_triggered"]
            label_badges = " ".join([
                f'<span class="badge badge-danger">{LABEL_ICONS.get(l, "")} {l}</span>'
                for l in triggered
            ]) if triggered else '<span class="badge badge-safe">No labels triggered</span>'

            uncertain_badges = " ".join([
                f'<span class="badge badge-warning">{LABEL_ICONS.get(l, "")} {l}</span>'
                for l in item.get("uncertain_labels", [])
            ])

            text_preview = item["original_text"][:200] + ("..." if len(item["original_text"]) > 200 else "")

            st.markdown(f"""
            <div class="review-item">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="color:#fde68a; font-weight:600;">Review #{len(uncertain)-i}</span>
                    <span style="color:#64748b; font-size:0.8rem;">{item.get('timestamp', '')[:19]}</span>
                </div>
                <p style="color:#e2e8f0; font-style:italic; margin:8px 0;">"{text_preview}"</p>
                <div style="margin:8px 0;">
                    <span style="color:#94a3b8; font-size:0.85rem;">Predicted: </span>{label_badges}
                </div>
                <div style="margin:8px 0;">
                    <span style="color:#94a3b8; font-size:0.85rem;">Uncertain: </span>{uncertain_badges}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Details — Review #{len(uncertain)-i}"):
                for reason in item.get("review_reasons", []):
                    st.markdown(f"- {reason}")
                st.json(item["probabilities"])

                # Review action
                col_a, col_b, col_c = st.columns(3)
                col_a.button(f"✅ Approve", key=f"approve_{i}")
                col_b.button(f"❌ Reject", key=f"reject_{i}")
                col_c.button(f"🔄 Escalate", key=f"escalate_{i}")
    else:
        st.markdown("""
        <div class="info-panel" style="text-align:center;">
            <p style="color:#6ee7b7; font-size:1.1rem;">✅ No uncertain cases</p>
            <p style="color:#64748b;">All predictions had high confidence. No manual review is needed at this time.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── All history ──
    st.markdown('<p class="section-header">📜 Full Prediction History</p>', unsafe_allow_html=True)

    history_table = []
    for h in reversed(history):
        history_table.append({
            "Time": h.get("timestamp", "")[:19],
            "Text": h["original_text"][:80] + "...",
            "Triggered": ", ".join(h["labels_triggered"]) or "None",
            "Uncertain": "⚠️" if h["is_uncertain"] else "✅",
            "Max Prob": f'{h["max_probability"]:.3f}',
            "Inference": f'{h.get("inference_ms", 0):.0f}ms',
        })
    st.dataframe(pd.DataFrame(history_table), use_container_width=True, hide_index=True)

    # Clear button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.predictions_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
#  PAGE: BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def page_batch():
    st.markdown('<p class="hero-title">📦 Batch Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Upload a CSV or enter multiple comments for bulk classification</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tab1, tab2 = st.tabs(["📄 Upload CSV", "✏️ Enter Multiple Comments"])

    with tab1:
        uploaded = st.file_uploader("Upload a CSV file with a 'text' or 'comment_text' column", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            text_col = None
            for col in ["text", "comment_text", "text_raw", "comment", "content"]:
                if col in df.columns:
                    text_col = col
                    break
            if text_col is None:
                st.error("Could not find a text column. Expected one of: text, comment_text, text_raw, comment, content")
                return

            st.info(f"Found {len(df)} comments in column `{text_col}`")
            max_rows = st.slider("Max comments to process", 10, min(1000, len(df)), min(100, len(df)))

            if st.button("🚀 Run Batch Prediction", type="primary"):
                predictor = load_predictor()
                texts = df[text_col].dropna().tolist()[:max_rows]

                progress = st.progress(0)
                results = []
                for i, text in enumerate(texts):
                    results.append(predictor.predict(text))
                    progress.progress((i + 1) / len(texts))

                # Build results table
                result_rows = []
                for r in results:
                    row = {"text": r["original_text"][:100]}
                    for lab in LABELS:
                        row[f"prob_{lab}"] = round(r["probabilities"][lab], 4)
                        row[f"pred_{lab}"] = r["predictions"][lab]
                    row["triggered"] = ", ".join(r["labels_triggered"]) or "None"
                    row["uncertain"] = r["is_uncertain"]
                    result_rows.append(row)

                rdf = pd.DataFrame(result_rows)
                st.dataframe(rdf, use_container_width=True, hide_index=True)

                # Summary
                n_toxic = sum(1 for r in results if r["labels_triggered"])
                n_uncertain = sum(1 for r in results if r["is_uncertain"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Processed", len(results))
                c2.metric("Toxic Detected", n_toxic)
                c3.metric("Uncertain (Review)", n_uncertain)

                # Download
                csv = rdf.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv, "batch_predictions.csv", "text/csv")

    with tab2:
        multi_text = st.text_area(
            "Enter one comment per line",
            height=200,
            placeholder="Comment 1\nComment 2\nComment 3\n...",
        )
        if st.button("🔍 Analyse All", type="primary", key="batch_multi"):
            lines = [l.strip() for l in multi_text.strip().split("\n") if l.strip()]
            if not lines:
                st.warning("Please enter at least one comment.")
                return

            predictor = load_predictor()
            results = []
            progress = st.progress(0)
            for i, text in enumerate(lines):
                results.append(predictor.predict(text))
                progress.progress((i + 1) / len(lines))

            for r in results:
                triggered = r["labels_triggered"]
                if triggered:
                    badges = " ".join([f'<span class="badge badge-danger">{LABEL_ICONS[l]} {l}</span>' for l in triggered])
                    st.markdown(f"""
                    <div class="result-toxic" style="padding:16px;">
                        <p style="color:#fca5a5; margin:0 0 8px 0;">🚨 <em>"{r['original_text'][:100]}"</em></p>
                        {badges}
                    </div>""", unsafe_allow_html=True)
                elif r["is_uncertain"]:
                    st.markdown(f"""
                    <div class="result-uncertain" style="padding:16px;">
                        <p style="color:#fde68a; margin:0;">🔶 <em>"{r['original_text'][:100]}"</em> — Uncertain</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe" style="padding:16px;">
                        <p style="color:#6ee7b7; margin:0;">✅ <em>"{r['original_text'][:100]}"</em></p>
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT / SYSTEM INFO
# ══════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown('<p class="hero-title">ℹ️ About ToxiGuard</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">System Architecture & Team Integration</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class="info-panel">
        <h3 style="color:#c7d2fe; margin-top:0;">🏗️ System Architecture</h3>
        <p style="color:#94a3b8;">ToxiGuard integrates five team members' work into a unified toxic comment detection pipeline:</p>
    </div>
    """, unsafe_allow_html=True)

    members = [
        ("1️⃣", "Data Preprocessing", "EDA, cleaning, splitting, identity annotation", "Member 1"),
        ("2️⃣", "Model Training", "DistilBERT fine-tuning, threshold optimisation", "Member 2"),
        ("3️⃣", "Fairness Audit", "Identity-group FPR/FNR analysis, Wilson CIs", "Member 3"),
        ("4️⃣", "XAI & Calibration", "SHAP explanations, temperature scaling, abstention", "Member 4"),
        ("5️⃣", "Application & Integration", "Streamlit UI, deployment, full pipeline", "Member 5"),
    ]

    for icon, title, desc, member in members:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:12px;">
            <p style="color:#c7d2fe; font-weight:700; font-size:1.1rem; margin:0;">{icon} {title}</p>
            <p style="color:#94a3b8; margin:4px 0 0 0;">{desc}</p>
            <p style="color:#64748b; font-size:0.8rem; margin:2px 0 0 0;">{member}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="info-panel">
        <h3 style="color:#c7d2fe; margin-top:0;">🔧 Technical Details</h3>
        <table style="color:#94a3b8; width:100%; border-collapse:collapse;">
            <tr><td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);"><strong>Model</strong></td>
                <td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);">DistilBERT-base-cased</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);"><strong>Task</strong></td>
                <td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);">Multi-label classification (6 labels)</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);"><strong>Labels</strong></td>
                <td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);">toxic, obscene, threat, insult, identity_hate, sexual_explicit</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);"><strong>Dataset</strong></td>
                <td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);">Jigsaw Toxic Comment — 1.77M comments</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);"><strong>Thresholds</strong></td>
                <td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);">Per-label, tuned on validation set via F1 optimisation</td></tr>
            <tr><td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);"><strong>Calibration</strong></td>
                <td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.1);">Temperature scaling (fitted on validation logits)</td></tr>
            <tr><td style="padding:8px;"><strong>Explainability</strong></td>
                <td style="padding:8px;">Attention-based token importance + SHAP</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Files status
    st.markdown('<p class="section-header">📁 File Status</p>', unsafe_allow_html=True)

    files_to_check = {
        "thresholds.json": "Per-label decision thresholds",
        "model/config.json": "Fine-tuned model config",
        "model/model.safetensors": "Model weights",
        "test_predictions.parquet": "Test set predictions",
        "calibration_config.json": "Temperature scaling config",
        "member3_outputs/group_rates_toxic.csv": "Fairness audit results",
        "toxic_preprocessing.py": "Text preprocessing module",
    }

    for fpath, desc in files_to_check.items():
        exists = Path(fpath).exists()
        icon = "✅" if exists else "❌"
        color = "#6ee7b7" if exists else "#fca5a5"
        st.markdown(
            f'<p style="color:{color}; margin:4px 0;">{icon} <code>{fpath}</code> — {desc}</p>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px 0;">
            <p style="font-size:2rem; margin:0;">🛡️</p>
            <p style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#818cf8,#c084fc);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">ToxiGuard</p>
            <p style="color:#64748b; font-size:0.8rem; margin:4px 0;">v1.0 · DistilBERT</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "🔍 Predict",
                "📈 Model Performance",
                "⚖️ Fairness Audit",
                "📋 Review Queue",
                "📦 Batch Analysis",
                "ℹ️ About",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Session stats
        history = st.session_state.get("predictions_history", [])
        uncertain_count = sum(1 for h in history if h.get("is_uncertain", False))

        st.markdown(f"""
        <div style="padding:12px; background:rgba(30,41,59,0.5); border-radius:8px; border:1px solid rgba(148,163,184,0.1);">
            <p style="color:#94a3b8; font-size:0.8rem; margin:0 0 8px 0;">SESSION STATS</p>
            <p style="color:#e2e8f0; margin:2px 0;">📊 Predictions: <strong>{len(history)}</strong></p>
            <p style="color:#fde68a; margin:2px 0;">⚠️ For Review: <strong>{uncertain_count}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Route to page
    if "Predict" in page:
        page_predict()
    elif "Performance" in page:
        page_performance()
    elif "Fairness" in page:
        page_fairness()
    elif "Review" in page:
        page_review()
    elif "Batch" in page:
        page_batch()
    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()
