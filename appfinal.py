import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentiAI · Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True   # default dark

dark = st.session_state.dark_mode

# ─────────────────────────────────────────────
# THEME TOKENS  (resolved in Python, injected as CSS vars)
# ─────────────────────────────────────────────
if dark:
    T = dict(
        bg         = "#0F172A",
        surface    = "#1E293B",
        surface2   = "#0F172A",
        border     = "#334155",
        border2    = "#1E293B",
        text       = "#F8FAFC",
        text2      = "#CBD5E1",
        muted      = "#64748B",
        accent     = "#06B6D4",
        accent_fg  = "#0F172A",
        nav_bg     = "#1E293B",
        nav_pill   = "#94A3B8",
        chart_bg   = "#1E293B",
        chart_tick = "#94A3B8",
        chart_axis = "#334155",
    )
else:
    T = dict(
        bg         = "#F8FAFC",
        surface    = "#FFFFFF",
        surface2   = "#F1F5F9",
        border     = "#E2E8F0",
        border2    = "#CBD5E1",
        text       = "#0F172A",
        text2      = "#334155",
        muted      = "#64748B",
        accent     = "#0284C7",
        accent_fg  = "#FFFFFF",
        nav_bg     = "#E2E8F0",
        nav_pill   = "#475569",
        chart_bg   = "#FFFFFF",
        chart_tick = "#334155",
        chart_axis = "#CBD5E1",
    )

# ─────────────────────────────────────────────
# INJECT CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* ── Completely remove Streamlit header gap ── */
#root > div:first-child {{
    padding-top: 0 !important;
}}
[data-testid="stAppViewContainer"] > section:first-child {{
    padding-top: 0 !important;
}}
[data-testid="stHeader"] {{
    display: none !important;
    height: 0 !important;
}}
[data-testid="stToolbar"] {{
    display: none !important;
}}
#MainMenu {{ visibility: hidden !important; }}
footer {{ visibility: hidden !important; }}
[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="collapsedControl"] {{ display: none !important; }}

/* ── Base ── */
html, body {{
    font-family: 'Inter', sans-serif !important;
    background-color: {T['bg']} !important;
    color: {T['text']} !important;
    margin: 0; padding: 0;
}}
[class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}
.stApp {{
    background-color: {T['bg']} !important;
}}

/* ── Block container ── */
.block-container {{
    padding: 0 1rem 4rem !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
}}

/* ══════════════════════════════════════
   TOP NAV
══════════════════════════════════════ */
.sai-nav {{
    position: sticky;
    top: 0;
    z-index: 999;
    background: {T['bg']};
    border-bottom: 1px solid {T['border']};
    padding: 0.75rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}}
.sai-brand {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: {T['text']};
    letter-spacing: -0.02em;
    white-space: nowrap;
}}
.sai-brand span {{ color: {T['accent']}; }}

.sai-pills {{
    display: flex;
    gap: 0.25rem;
    background: {T['nav_bg']};
    padding: 0.25rem;
    border-radius: 12px;
    flex-wrap: wrap;
}}
.sai-pill {{
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    color: {T['nav_pill']};
    white-space: nowrap;
    border: none;
    background: transparent;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
}}
.sai-pill:hover {{
    background: {T['border']};
    color: {T['text']};
}}
.sai-pill.active {{
    background: {T['accent']};
    color: {T['accent_fg']};
    font-weight: 600;
}}

/* Theme toggle */
.sai-theme-btn {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 8px;
    padding: 0.38rem 0.65rem;
    font-size: 1rem;
    cursor: pointer;
    color: {T['text']};
    transition: background 0.15s;
    white-space: nowrap;
    flex-shrink: 0;
}}
.sai-theme-btn:hover {{ background: {T['border']}; }}

/* ══════════════════════════════════════
   TYPOGRAPHY
══════════════════════════════════════ */
.eyebrow {{
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {T['accent']};
    margin-bottom: 0.3rem;
}}
.display-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.4rem, 4vw, 2rem);
    font-weight: 700;
    color: {T['text']};
    line-height: 1.2;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}}
.sub-text {{
    font-size: 1rem;
    color: {T['muted']};
    line-height: 1.65;
    max-width: 540px;
}}

/* ══════════════════════════════════════
   CARDS
══════════════════════════════════════ */
.card {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}}
.card-sm {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    padding: 1rem 1.25rem;
}}

/* ══════════════════════════════════════
   METRIC TILES
══════════════════════════════════════ */
.metric-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}}
.metric-tile {{
    background: {T['surface2']};
    border: 1px solid {T['border2']};
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}}
.metric-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: {T['accent']};
}}
.metric-lbl {{
    font-size: 0.85rem;
    color: {T['muted']};
    margin-top: 0.15rem;
}}

/* ══════════════════════════════════════
   STEPS GRID
══════════════════════════════════════ */
.steps-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-top: 0.75rem;
    margin-bottom: 1.5rem;
}}
.step-tile {{
    background: {T['surface2']};
    border: 1px solid {T['border2']};
    border-radius: 12px;
    padding: 1rem 0.75rem;
    text-align: center;
}}
.step-num {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: {T['accent']};
    opacity: 0.45;
    line-height: 1;
}}
.step-icon {{ font-size: 1.3rem; margin: 0.3rem 0; }}
.step-desc {{ font-size: 0.88rem; color: {T['text2']}; line-height: 1.5; }}

/* ══════════════════════════════════════
   TEAM
══════════════════════════════════════ */
.team-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-top: 0.75rem;
}}
.team-card {{
    background: {T['surface2']};
    border: 1px solid {T['border2']};
    border-radius: 12px;
    padding: 1rem 0.75rem;
    text-align: center;
}}
.team-avatar {{
    width: 40px; height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, {T['accent']}, #8B5CF6);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700; font-size: 0.95rem;
    color: {T['accent_fg']};
    margin: 0 auto 0.6rem;
}}
.team-name {{ font-weight: 600; font-size: 0.88rem; color: {T['text']}; }}
.team-role {{ font-size: 0.82rem; color: {T['muted']}; margin-top: 0.15rem; }}

/* ══════════════════════════════════════
   SENTIMENT BADGES
══════════════════════════════════════ */
.badge-pos {{
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    background: rgba(16,185,129,0.1);
    border: 1.5px solid rgba(16,185,129,0.4);
    color: #10B981;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    padding: 0.85rem 1rem; border-radius: 12px;
}}
.badge-neg {{
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    background: rgba(239,68,68,0.1);
    border: 1.5px solid rgba(239,68,68,0.4);
    color: #EF4444;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    padding: 0.85rem 1rem; border-radius: 12px;
}}
.badge-neu {{
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    background: rgba(251,191,36,0.1);
    border: 1.5px solid rgba(251,191,36,0.4);
    color: #FBBF24;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    padding: 0.85rem 1rem; border-radius: 12px;
}}

/* ── Confidence bar ── */
.conf-track {{
    background: {T['surface2']};
    border-radius: 999px; height: 10px; overflow: hidden;
    margin: 0.5rem 0 0.25rem;
}}
.conf-pos {{ background: linear-gradient(90deg,{T['accent']},#10B981); border-radius:999px; height:100%; }}
.conf-neg {{ background: linear-gradient(90deg,#F97316,#EF4444);        border-radius:999px; height:100%; }}
.conf-neu {{ background: linear-gradient(90deg,#8B5CF6,#FBBF24);        border-radius:999px; height:100%; }}
.conf-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: {T['text']};
}}
.conf-lbl {{ font-size: 0.88rem; color: {T['muted']}; }}

/* ══════════════════════════════════════
   MODEL TABLE
══════════════════════════════════════ */
.model-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
.model-table th {{
    color: {T['muted']}; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.5rem 0.75rem; text-align: right; border-bottom: 1px solid {T['border']};
}}
.model-table th:first-child {{ text-align: left; }}
.model-table td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid {T['border2']}; color: {T['text2']}; text-align: right; }}
.model-table td:first-child {{ text-align: left; color: {T['text']}; font-weight: 500; }}
.model-table tr.winner td {{ color: {T['accent']}; font-weight: 600; background: rgba(6,182,212,0.06); }}
.model-table tr.winner td:first-child {{ color: {T['accent']}; }}
.winner-badge {{
    display: inline-block;
    background: {T['accent']}; color: {T['accent_fg']};
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    padding: 0.12rem 0.45rem; border-radius: 999px; margin-left: 0.4rem;
    vertical-align: middle;
}}
.detail-table {{ width:100%; border-collapse:collapse; font-size:0.83rem; }}
.detail-table td {{ padding: 0.45rem 0; border-bottom: 1px solid {T['border']}; }}
.detail-table td:first-child {{ color: {T['muted']}; }}
.detail-table td:last-child  {{ color: {T['text']}; text-align: right; }}

/* ══════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
══════════════════════════════════════ */
.stTextArea textarea {{
    background: {T['surface2']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    color: {T['text']} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}}
.stTextArea textarea:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 2px {T['accent']}22 !important;
}}
.stButton > button {{
    background: {T['accent']} !important;
    color: {T['accent_fg']} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.75rem !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}
[data-testid="stTabs"] button {{
    color: {T['muted']} !important;
    font-size: 0.85rem !important;
}}
[data-testid="stHorizontalBlock"] .stButton > button {{
    background: transparent !important;
    color: {T['nav_pill']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    padding: 0.5rem 0.75rem !important;
}}
[data-testid="stHorizontalBlock"] .stButton > button:hover {{
    background: {T['border']} !important;
    color: {T['text']} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {T['accent']} !important;
    border-bottom-color: {T['accent']} !important;
}}
[data-testid="stDataFrame"] {{ background: {T['surface']} !important; border-radius: 10px !important; }}
.stProgress > div > div {{ background: {T['accent']} !important; }}

/* ══════════════════════════════════════
   RESPONSIVE / MOBILE-FIRST
══════════════════════════════════════ */
@media (max-width: 640px) {{
    .block-container {{
        padding: 0 1.25rem 3rem !important;
    }}
    .sai-nav {{
        padding: 0.6rem 0.75rem;
        gap: 0.5rem;
    }}
    .sai-pills {{
        gap: 0.15rem;
        padding: 0.2rem;
    }}
    .sai-pill {{
        padding: 0.35rem 0.6rem;
        font-size: 0.95rem;
    }}
    .sai-brand {{ font-size: 1rem; }}
    .display-title {{ font-size: 1.3rem !important; }}
    .metric-row {{ grid-template-columns: repeat(3,1fr); gap: 0.5rem; }}
    .metric-val  {{ font-size: 1.2rem; }}
    .steps-grid  {{ grid-template-columns: repeat(2,1fr); }}
    .team-grid   {{ grid-template-columns: repeat(2,1fr); }}
    .conf-val    {{ font-size: 1.5rem; }}
    .model-table {{ font-size: 0.72rem; }}
    .model-table th, .model-table td {{ padding: 0.4rem 0.4rem; }}
}}
@media (max-width: 400px) {{
    .sai-pill {{ padding: 0.3rem 0.45rem; font-size: 0.68rem; }}
    .steps-grid {{ grid-template-columns: 1fr 1fr; }}
    .team-grid  {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL / DATA HELPERS
# ─────────────────────────────────────────────
def get_setting(name, default=None):
    v = os.environ.get(name)
    if v: return v
    try:
        s = st.secrets.get(name)
    except Exception:
        s = None
    return s if s is not None else default

MODEL_PATH    = get_setting("MODEL_PATH", "models/best_model_bert")
MODEL_REPO_ID = get_setting("MODEL_REPO_ID")
MODEL_REVISION= get_setting("MODEL_REVISION", "main")
LABEL_MAP     = {0: "Negative", 1: "Neutral", 2: "Positive"}

def ensure_model_available():
    if os.path.isdir(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        return MODEL_PATH
    if MODEL_REPO_ID:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub required.")
        os.makedirs("models", exist_ok=True)
        ph = st.empty()
        ph.info("Downloading model from Hugging Face Hub…")
        try:
            p = snapshot_download(repo_id=MODEL_REPO_ID, revision=MODEL_REVISION,
                                  local_dir=MODEL_PATH, local_dir_use_symlinks=False)
            ph.success("Model ready.")
            return p
        except Exception as e:
            ph.error(str(e)); raise
    raise FileNotFoundError(f"Model dir '{MODEL_PATH}' not found.")

@st.cache_resource
def load_model():
    p = ensure_model_available()
    m = DistilBertForSequenceClassification.from_pretrained(p)
    t = DistilBertTokenizerFast.from_pretrained(p)
    m.eval()
    return m, t

@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    model, tokenizer = None, None

df = load_data()

# ─────────────────────────────────────────────
# NAV HELPER  — renders the sticky top bar
# ─────────────────────────────────────────────
PAGES = [
    ("Home",           "🏠"),
    ("Analyzer",       "✍️"),
    ("Dataset",        "📊"),
    ("Visualizations", "📈"),
    ("Model",          "🤖"),
]

def render_nav():
    accent = T['accent']
    border = T['border']

    b1, b2 = st.columns([6, 1])
    with b1:
        st.markdown(f'<div class="sai-brand" style="padding:0.5rem 0;">Sentiment<span style="color:{accent}">Classifier</span></div>', unsafe_allow_html=True)
    with b2:
        if st.button("☀️" if dark else "🌙", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    cols = st.columns(len(PAGES))
    for i, (name, icon) in enumerate(PAGES):
        with cols[i]:
            if st.button(f"{icon} {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.page = name
                st.rerun()

    st.markdown(f'<hr style="margin:0.5rem 0 1.5rem;border-color:{border};">', unsafe_allow_html=True)

render_nav()

page = st.session_state.page

# ─────────────────────────────────────────────
# CHART HELPER
# ─────────────────────────────────────────────
def styled_fig(w=7, h=3):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(T['chart_bg'])
    ax.set_facecolor(T['chart_bg'])
    ax.tick_params(colors=T['chart_tick'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(T['chart_axis'])
    return fig, ax

# ══════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="eyebrow">NLP · Transformer · DistilBERT</div>
    <div class="display-title">Understand what customers<br>actually feel.</div>
    <div class="sub-text" style="margin-bottom:1.75rem;">
        A fine-tuned DistilBERT model that classifies reviews as Positive, Negative, or Neutral —
        instantly, at scale, with word-level explainability.
    </div>
    <div class="metric-row">
        <div class="metric-tile"><div class="metric-val">89.6%</div><div class="metric-lbl">Accuracy</div></div>
        <div class="metric-tile"><div class="metric-val">88.5%</div><div class="metric-lbl">F1-Weighted</div></div>
        <div class="metric-tile"><div class="metric-val">3</div><div class="metric-lbl">Sentiment Classes</div></div>
    </div>
    <div class="eyebrow" style="margin-top:1.75rem;">How It Works</div>
    <div class="steps-grid">
        <div class="step-tile"><div class="step-num">01</div><div class="step-icon">✍️</div><div class="step-desc">Type or paste any customer review</div></div>
        <div class="step-tile"><div class="step-num">02</div><div class="step-icon">⚡</div><div class="step-desc">DistilBERT tokenizes and encodes your text</div></div>
        <div class="step-tile"><div class="step-num">03</div><div class="step-icon">🎯</div><div class="step-desc">Model predicts sentiment with a confidence score</div></div>
        <div class="step-tile"><div class="step-num">04</div><div class="step-icon">🔍</div><div class="step-desc">See which words drove the prediction</div></div>
    </div>
    <div class="eyebrow" style="margin-top:1.75rem;">Team</div>
    <div class="team-grid">
        <div class="team-card"><div class="team-avatar">A</div><div class="team-name">Alif</div><div class="team-role">ML Engineer</div></div>
        <div class="team-card"><div class="team-avatar">R</div><div class="team-name">Rania</div><div class="team-role">NLP Engineer</div></div>
        <div class="team-card"><div class="team-avatar">K</div><div class="team-name">Kanesh</div><div class="team-role">Data Engineer</div></div>
        <div class="team-card"><div class="team-avatar">N</div><div class="team-name">Nurien</div><div class="team-role">UI / Evaluation</div></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ANALYZER
# ══════════════════════════════════════════════
elif page == "Analyzer":
    st.markdown("""
    <div class="eyebrow">Live Prediction</div>
    <div class="display-title">Text Analyzer</div>
    <div class="sub-text" style="margin-bottom:1.25rem;">Paste a review and click Analyze to get an instant sentiment prediction with word-level explanation.</div>
    """, unsafe_allow_html=True)

    review = st.text_area(
        "Your review",
        placeholder="e.g. The product quality is excellent but delivery was slower than expected…",
        height=120,
        label_visibility="collapsed",
    )

    if st.button("Analyze Sentiment →", key="analyze_btn"):
        if not review.strip():
            st.warning("Please enter a review before analyzing.")
        elif model is None:
            st.error("Model is not loaded.")
        else:
            inputs = tokenizer(review, return_tensors="pt", truncation=True,
                               padding=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
            probs      = F.softmax(outputs.logits, dim=1).numpy()[0]
            prediction = int(np.argmax(probs))
            confidence = float(np.max(probs)) * 100
            label      = LABEL_MAP[prediction]

            # Result + Confidence
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown('<div class="eyebrow">Result</div>', unsafe_allow_html=True)
                badge = {"Positive": ("badge-pos","✅ Positive"),
                         "Negative": ("badge-neg","❌ Negative"),
                         "Neutral":  ("badge-neu","➖ Neutral")}[label]
                st.markdown(f'<div class="{badge[0]}">{badge[1]}</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="eyebrow">Confidence</div>', unsafe_allow_html=True)
                fill = {"Positive":"conf-pos","Negative":"conf-neg","Neutral":"conf-neu"}[label]
                st.markdown(f"""
                <div class="card-sm">
                    <div class="conf-val">{confidence:.1f}%</div>
                    <div class="conf-track"><div class="{fill}" style="width:{confidence:.1f}%"></div></div>
                    <div class="conf-lbl">Model confidence in the predicted class</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

            # Probability chart
            st.markdown('<div class="eyebrow">Class Probabilities</div>', unsafe_allow_html=True)
            fig, ax = styled_fig(6, 2.2)
            cls_names  = [LABEL_MAP[i] for i in range(len(probs))]
            cls_colors = {"Negative":"#EF4444","Neutral":"#FBBF24","Positive":"#10B981"}
            bars = ax.barh(cls_names, probs,
                           color=[cls_colors[c] for c in cls_names],
                           height=0.45, edgecolor='none')
            for bar, p in zip(bars, probs):
                ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                        f'{p:.1%}', va='center', ha='left',
                        color=T['chart_tick'], fontsize=9)
            ax.set_xlim(0, 1.18)
            ax.xaxis.set_visible(False)
            ax.spines[:].set_visible(False)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig); plt.close(fig)

            # Word influence
            st.markdown('<div class="eyebrow" style="margin-top:1.25rem;">Word Influence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sub-text" style="margin-bottom:0.6rem;">Each word is temporarily removed to measure how much the model\'s confidence drops.</div>', unsafe_allow_html=True)

            words = review.split()
            if len(words) > 1:
                influences = []
                prog = st.progress(0, text="Computing word influence…")
                for i, w in enumerate(words):
                    masked = " ".join(words[:i] + words[i+1:])
                    mi = tokenizer(masked, return_tensors="pt",
                                   truncation=True, padding=True, max_length=256)
                    with torch.no_grad():
                        mo = model(**mi)
                    mp = F.softmax(mo.logits, dim=1).numpy()[0]
                    influences.append(confidence - mp[prediction]*100)
                    prog.progress((i+1)/len(words), text=f"Word {i+1}/{len(words)}…")
                prog.empty()

                plot_df = pd.DataFrame({"Word": words, "Influence": influences}).sort_values("Influence")
                fig2, ax2 = styled_fig(7, max(2.5, len(words)*0.38))
                ax2.barh(plot_df["Word"], plot_df["Influence"],
                         color=["#EF4444" if v < 0 else T['accent'] for v in plot_df["Influence"]],
                         height=0.6, edgecolor='none')
                ax2.axvline(0, color=T['chart_axis'], linewidth=1)
                ax2.set_xlabel("Confidence drop (%)", color=T['muted'], fontsize=9)
                ax2.spines[:].set_color(T['chart_axis'])
                plt.tight_layout(pad=0.4)
                st.pyplot(fig2); plt.close(fig2)
            else:
                st.info("Enter a longer review to see word-level influence.")

# ══════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════
elif page == "Dataset":
    st.markdown("""
    <div class="eyebrow">Training Data</div>
    <div class="display-title">Dataset Explorer</div>
    <div class="sub-text" style="margin-bottom:1.25rem;">Browse and inspect the dataset used to train the sentiment model.</div>
    """, unsafe_allow_html=True)

    missing = int(df.isnull().sum().sum())
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile"><div class="metric-val">{df.shape[0]:,}</div><div class="metric-lbl">Total Rows</div></div>
        <div class="metric-tile"><div class="metric-val">{df.shape[1]}</div><div class="metric-lbl">Columns</div></div>
        <div class="metric-tile"><div class="metric-val">{"✅ 0" if missing==0 else missing}</div><div class="metric-lbl">Missing Values</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Sample Rows", "📐 Statistics"])
    with tab1:
        st.dataframe(df.head(25), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

# ══════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════
elif page == "Visualizations":
    st.markdown("""
    <div class="eyebrow">Analysis</div>
    <div class="display-title">Visualizations</div>
    <div class="sub-text" style="margin-bottom:1.25rem;">Data distributions, model comparison charts, and evaluation metrics.</div>
    """, unsafe_allow_html=True)

    def show_img(path, caption):
        try:
            from PIL import Image
            img = Image.open(path)
            # Crop/pad to consistent 16:9 frame
            target_w, target_h = 800, 450
            img_w, img_h = img.size
            scale = min(target_w / img_w, target_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            # Paste onto blank canvas of exact size
            canvas = Image.new("RGB", (target_w, target_h),
                            color=(30, 41, 59) if dark else (255, 255, 255))
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            canvas.paste(img, (offset_x, offset_y))
            st.image(canvas, use_container_width=True, caption=caption)
        except FileNotFoundError:
            st.markdown(
                f'<div class="card-sm" style="height:200px;display:flex;align-items:center;'
                f'justify-content:center;color:{T["muted"]};text-align:center;">'
                f'📁 Add <code>{path}</code></div>',
                unsafe_allow_html=True)
        except Exception:
            st.markdown(
                f'<div class="card-sm" style="height:200px;display:flex;align-items:center;'
                f'justify-content:center;color:{T["muted"]};text-align:center;">'
                f'📁 Add <code>{path}</code></div>',
                unsafe_allow_html=True)

    sections = [
        ("Sentiment Distribution",  "images/label_distribution.png"),
        ("Word Cloud",               "images/wc_new.png"),
        ("Model Comparison",         "images/model_comparison_chart.png"),
        ("Accuracy Across Models",   "images/viz_accuracy_comparison.png"),
        ("F1 Score Per Class",       "images/viz_f1_per_class.png"),
        ("Confusion Matrices",       "images/viz_confusion_matrices.png"),
        ("ROC Curves",               "images/viz_roc_curves.png"),
    ]
    for i in range(0, len(sections), 2):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(f'<div class="eyebrow">{sections[i][0]}</div>', unsafe_allow_html=True)
            show_img(sections[i][1], sections[i][0])
        if i+1 < len(sections):
            with c2:
                st.markdown(f'<div class="eyebrow">{sections[i+1][0]}</div>', unsafe_allow_html=True)
                show_img(sections[i+1][1], sections[i+1][0])
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════════
elif page == "Model":
    st.markdown("""
    <div class="eyebrow">Architecture & Results</div>
    <div class="display-title">Model Information</div>
    <div class="sub-text" style="margin-bottom:1.25rem;">Three configurations were trained and compared. DistilBERT was selected as the deployed model.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
    <table class="model-table">
      <tr>
        <th>Configuration</th>
        <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Macro</th><th>F1-Weighted</th>
      </tr>
      <tr>
        <td>LR × TF-IDF</td>
        <td>76.81%</td><td>61.30%</td><td>64.36%</td><td>60.42%</td><td>80.32%</td>
      </tr>
      <tr>
        <td>LR × Word2Vec</td>
        <td>73.95%</td><td>59.44%</td><td>61.88%</td><td>57.95%</td><td>78.19%</td>
      </tr>
      <tr class="winner">
        <td>DistilBERT <span class="winner-badge">Deployed</span></td>
        <td>89.64%</td><td>74.92%</td><td>65.99%</td><td>67.81%</td><td>88.46%</td>
      </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown('<div class="eyebrow">Deployed Model Details</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
        <table class="detail-table">
          <tr><td>Algorithm</td><td>DistilBERT Fine-Tuned</td></tr>
          <tr><td>Feature Extraction</td><td>Native tokenization</td></tr>
          <tr><td>Imbalance Handling</td><td>None (raw distribution)</td></tr>
          <tr><td>Training Epochs</td><td>2</td></tr>
          <tr><td>Training Loss (Ep 1)</td><td>0.4142</td></tr>
          <tr><td>Training Loss (Ep 2)</td><td>0.3339</td></tr>
        </table>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="eyebrow">Performance Metrics</div>', unsafe_allow_html=True)
        metrics = [("Accuracy","89.64"),("Precision","74.92"),
                   ("Recall","65.99"),("F1-Macro","67.81"),("F1-Weighted","88.46")]
        fig3, ax3 = styled_fig(5, 2.8)
        vals  = [float(v) for _, v in metrics]
        lbls  = [l for l, _ in metrics]
        pal   = [T['accent'],'#8B5CF6','#10B981','#FBBF24',T['accent']]
        ax3.barh(lbls, vals, color=pal, height=0.52, edgecolor='none')
        for i, v in enumerate(vals):
            ax3.text(v+0.5, i, f'{v}%', va='center', ha='left',
                     color=T['chart_tick'], fontsize=8.5)
        ax3.set_xlim(0, 112)
        ax3.spines[:].set_color(T['chart_axis'])
        ax3.xaxis.set_visible(False)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig3); plt.close(fig3)

    st.markdown(f"""
    <div class="card-sm" style="margin-top:0.75rem;border-left:3px solid #FBBF24;">
      <div style="font-size:0.78rem;color:#FBBF24;font-weight:600;margin-bottom:0.2rem;">ℹ️ Note on F1-Macro</div>
      <div style="font-size:0.8rem;color:{T['text2']};">
        F1-Macro is lower than F1-Weighted because the <strong>Neutral</strong> class is a challenging
        minority — the model scores only F1≈0.23 on it, pulling the macro average down despite
        strong Positive and Negative performance.
      </div>
    </div>""", unsafe_allow_html=True)