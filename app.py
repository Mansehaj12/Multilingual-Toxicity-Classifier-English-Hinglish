import streamlit as st
import pandas as pd
import numpy as np
from src.predict import ToxicityPredictor
import re

# Page Configuration
st.set_page_config(
    page_title="Hinglish Toxicity Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Dark Theme & Layout
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

/* Main Body & Background */
.stApp {
    background-color: #080b11;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #e2e8f0;
}

/* Custom Header with Gradient */
.app-header {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    text-align: center;
}

.app-subheader {
    font-size: 1.05rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* Style Native Streamlit Containers with Border (Dark Mode Glassmorphic) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 24, 39, 0.55) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3) !important;
    padding: 24px !important;
    margin-bottom: 15px !important;
}

/* Top Dashboard Metric Cards */
.dashboard-metric-card {
    background: rgba(17, 24, 39, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease;
}

.dashboard-metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.3);
}

.dashboard-metric-title {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    margin-bottom: 4px;
}

.dashboard-metric-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
}

.dashboard-metric-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 2px;
}

/* Custom Predictions Status Header */
.metric-container {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 12px;
    padding: 14px;
    border: 1px solid rgba(255, 255, 255, 0.04);
}

.metric-title {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 5px;
}

/* Custom Styled Progress Bars */
.progress-bar-container {
    margin-bottom: 11px;
}

.progress-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 4px;
    color: #cbd5e1;
}

.progress-bar-bg {
    background-color: #1e293b;
    border-radius: 8px;
    height: 7px;
    width: 100%;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease-in-out;
}

/* Explainability Highlight Spans */
.toxic-span {
    display: inline-block;
    padding: 3px 8px;
    margin: 4px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 1.05rem;
    transition: transform 0.2s ease;
}

.toxic-span:hover {
    transform: scale(1.08);
    cursor: default;
}

/* Button Customizations */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    box-shadow: 0 0 15px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(-2px) !important;
}

/* Form inputs & areas styling (Dark Mode) */
textarea {
    background-color: rgba(15, 23, 42, 0.5) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Hide Streamlit components sidebar */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize Predictor (Renamed to force cache reload)
@st.cache_resource
def get_toxicity_predictor_v2():
    return ToxicityPredictor(use_fallback=True)

predictor = get_toxicity_predictor_v2()

# Logo Image Centering
col_img1, col_img2, col_img3 = st.columns([10, 1.8, 10])
with col_img2:
    st.image("multilingual_shield_logo.png", use_container_width=True)

# App Header
st.markdown('<div class="app-header">🛡️ Hinglish Toxicity Guard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subheader">Explainable Content Moderation for Multilingual Social Media</div>', unsafe_allow_html=True)

# Center Model Selection Dropdown at the top
col_sel1, col_sel2, col_sel3 = st.columns([6, 5, 6])
with col_sel2:
    model_selection = st.selectbox(
        "🧠 Select Model Classifier Backbone:",
        ["XLM-RoBERTa", "mBERT", "Qwen2.5"],
        index=0
    )

# --- Top Structured Dashboard Metrics Header ---
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

# Determine status details for the selected model
best_metric = "Best F1 Macro" if model_selection == "XLM-RoBERTa" else ("Best F1 Micro" if model_selection == "Qwen2.5" else "Overall F1")
best_value = "0.6525" if model_selection == "XLM-RoBERTa" else ("0.7362" if model_selection == "Qwen2.5" else "0.7182 (mBERT)")

with metric_col1:
    st.markdown(
        f'<div class="dashboard-metric-card">'
        f'  <div class="dashboard-metric-title">Active Backbone</div>'
        f'  <div class="dashboard-metric-value" style="color: #818cf8;">{model_selection}</div>'
        f'  <div class="dashboard-metric-sub">Mode: {"Live" if predictor.mode == "live" else "Demo Mode"}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with metric_col2:
    st.markdown(
        f'<div class="dashboard-metric-card">'
        f'  <div class="dashboard-metric-title">Training Corpus</div>'
        f'  <div class="dashboard-metric-value">354,895 posts</div>'
        f'  <div class="dashboard-metric-sub">English + Hinglish</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with metric_col3:
    st.markdown(
        f'<div class="dashboard-metric-card">'
        f'  <div class="dashboard-metric-title">Model Specialty</div>'
        f'  <div class="dashboard-metric-value" style="color: #34d399;">{best_metric}</div>'
        f'  <div class="dashboard-metric-sub">Score: {best_value}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with metric_col4:
    st.markdown(
        f'<div class="dashboard-metric-card">'
        f'  <div class="dashboard-metric-title">Explainability Layer</div>'
        f'  <div class="dashboard-metric-value" style="color: #fbbf24;">{"Gradient Saliency" if "qwen" not in model_selection.lower() else "Length Saliency"}</div>'
        f'  <div class="dashboard-metric-sub">Feature: Toxic Spans</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# Model Comparison Expander right below metrics
with st.expander("📊 View Detailed Model Comparison Benchmarks"):
    metrics_data = {
        "Metric": ["F1 Micro", "F1 Macro", "F1 Weighted", "Precision", "Recall"],
        "mBERT": ["0.7182", "0.6345", "0.7179", "0.7374", "0.7000"],
        "XLM-R": ["0.7270", "0.6525", "0.7266", "0.7513", "0.7042"],
        "Qwen2.5": ["0.7362", "0.6064", "0.7353", "0.7353", "0.7371"]
    }
    st.table(pd.DataFrame(metrics_data).set_index("Metric"))

st.markdown("<br/>", unsafe_allow_html=True)

# Initialize text_input state if not exists
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Callback function to set text securely before widgets are instantiated
def set_text(val):
    st.session_state.text_input = val

# Layout Configuration
col1, col2 = st.columns([11, 10], gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### 📝 Analyze Comment")
        
        # Input Area - mapped to session state key
        user_input = st.text_area(
            "Enter a social media comment to moderate (English, Hindi, or Hinglish):",
            placeholder="Type something here, or select a preset below...",
            height=120,
            key="text_input"
        )
        
        # Preset Quick-tags using safe callbacks
        st.markdown("**💡 Try These Test Presets (from the PDF Report):**")
        preset_col1, preset_col2 = st.columns(2)
        
        with preset_col1:
            st.button("tu bahut stupid hai", on_click=set_text, args=("tu bahut stupid hai",))
            st.button("you are a disgusting idiot", on_click=set_text, args=("you are a disgusting idiot",))
            st.button("tum ekdum useless ho", on_click=set_text, args=("tum ekdum useless ho",))
            st.button("bhai tu thoda dumb lag raha hai", on_click=set_text, args=("bhai tu thoda dumb lag raha hai",))
                
        with preset_col2:
            st.button("bhai tu mast kaam kar raha hai", on_click=set_text, args=("bhai tu mast kaam kar raha hai",))
            st.button("you are a very kind person", on_click=set_text, args=("you are a very kind person",))
            st.button("I hate you so much", on_click=set_text, args=("I hate you so much",))
            st.button("bhai tu mental hai", on_click=set_text, args=("bhai tu mental hai",))

        analyze_clicked = st.button("Run Moderation Analysis", use_container_width=True)

# Run Inference or Show Startup Guide
if user_input:
    with st.spinner(f"Analyzing text semantic vectors using {model_selection}..."):
        scores, attributions = predictor.predict_and_explain(user_input, model_name=model_selection)
        
    with col2:
        with st.container(border=True):
            st.markdown(f"### 📊 Toxicity Categorization ({model_selection})")
            
            # Max probability determines status
            max_label = max(scores, key=scores.get)
            max_score = scores[max_label]
            
            if max_score > 0.5:
                st.markdown(f'<div class="metric-container" style="border-left: 6px solid #ef4444; margin-bottom: 16px;">'
                            f'<div class="metric-title" style="color: #fca5a5;">🚨 High Toxicity Alert</div>'
                            f'<div class="metric-value" style="color: #f87171;">Toxicity Detected ({max_score:.1%})</div>'
                            f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-container" style="border-left: 6px solid #10b981; margin-bottom: 16px;">'
                            f'<div class="metric-title" style="color: #a7f3d0;">✅ Content Passed</div>'
                            f'<div class="metric-value" style="color: #34d399;">Safe / Clear (Max: {max_score:.1%})</div>'
                            f'</div>', unsafe_allow_html=True)
                
            # Draw custom progress bars for labels
            label_colors = {
                "toxic": "linear-gradient(90deg, #f87171, #ef4444)",
                "obscene": "linear-gradient(90deg, #fbbf24, #f59e0b)",
                "insult": "linear-gradient(90deg, #c084fc, #a855f7)",
                "identity_hate": "linear-gradient(90deg, #fb7185, #f43f5e)",
                "threat": "linear-gradient(90deg, #fb923c, #f97316)",
                "severe_toxic": "linear-gradient(90deg, #ec4899, #d946ef)"
            }
            
            for label in predictor.labels:
                score = scores[label]
                color = label_colors.get(label, "linear-gradient(90deg, #38bdf8, #0284c7)")
                
                st.markdown(
                    f'<div class="progress-bar-container">'
                    f'  <div class="progress-bar-label">'
                    f'    <span>{label.upper().replace("_", " ")}</span>'
                    f'    <span>{score:.1%}</span>'
                    f'  </div>'
                    f'  <div class="progress-bar-bg">'
                    f'    <div class="progress-bar-fill" style="width: {score * 100}%; background: {color};"></div>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
    # Explainability Section (Full-width container below)
    with st.container(border=True):
        st.markdown(f"### 🔍 Gradient-Based Token Attribution ({model_selection})")
        st.markdown("This section highlights which specific words driven by the selected model embeddings triggered the toxicity prediction. "
                    "The **intensity of the highlight** corresponds to the token's gradient attribution score.")
        
        # Custom HTML display for token attributions
        highlight_html = '<div style="background: rgba(15, 23, 42, 0.4); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05);">'
        
        for token, score in attributions:
            if max_score > 0.5:
                bg_color = f"rgba(239, 68, 68, {score * 0.85})"
                border_color = f"rgba(239, 68, 68, {score * 0.3})"
                text_color = "#ffffff" if score > 0.3 else "#cbd5e1"
            else:
                bg_color = f"rgba(56, 189, 248, {score * 0.15})"
                border_color = f"rgba(56, 189, 248, {score * 0.05})"
                text_color = "#cbd5e1"
                
            highlight_html += (
                f'<span class="toxic-span" style="background: {bg_color}; border: 1px solid {border_color}; color: {text_color};" title="Attribution: {score:.3f}">'
                f'{token}'
                f'</span>'
            )
            
        highlight_html += '</div>'
        st.markdown(highlight_html, unsafe_allow_html=True)
else:
    # Balancing column placeholder on startup
    with col2:
        with st.container(border=True):
            st.markdown("### ℹ️ Moderation System Status")
            st.markdown(
                """
                **Waiting for input...** 
                
                Enter a comment in the input area or select one of the test presets to begin. 
                The system will automatically run the following analysis:
                
                1. **Toxicity Categorization:** Real-time probability scoring across 6 content-moderation classes.
                2. **Explainability Mapping:** Saliency scores highlighting token contribution directly on the text.
                """
            )
            
            # Quick Category Legend inside card
            st.markdown("**🛡️ Moderation Class Description:**")
            st.caption("• **TOXIC:** General hate, slurs, or hostile language")
            st.caption("• **INSULT:** Targeted personal attacks")
            st.caption("• **IDENTITY HATE:** Bias targeting race, religion, gender, etc.")
            st.caption("• **THREAT:** Incitement of physical harm or violence")
            st.caption("• **OBSCENE / SEVERE:** Vulgar vocabulary or extreme hostility")

    # Bottom full-width container placeholder
    with st.container(border=True):
        st.markdown("### 🔍 Gradient-Based Token Attribution")
        st.info("No attribution scores generated yet. Run analysis on a comment to identify toxic word highlights.")
