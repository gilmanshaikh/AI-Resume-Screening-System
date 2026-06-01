import streamlit as st
import pdfplumber
import io
import re
import pandas as pd
from datetime import datetime
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def render_footer():
    st.markdown(
        "<p class='footer'>🚀 DevOps AI-Resume-ScreeningSystem · Built by <strong>GILMAN SHAIKH</strong></p>",
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle="", icon="TS"):
    st.markdown(
        f"""
        <div class="page-header">
            <div class="role-banner">
                <span class="banner-icon">{icon}</span>
                <span>{subtitle}</span>
            </div>
            <h1>{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(stats):
    """stats: list of (label, value, accent_bool)"""
    cards = "".join(
        f'<div class="stat-card"><div class="label">{lbl}</div>'
        f'<div class="value{" accent" if accent else ""}">{val}</div></div>'
        for lbl, val, accent in stats
    )
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="AI Resume Screening System V2.1 (DevOps Project...)  | Gilman Shaikh",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "🤖 AI Resume Screening System V2.1 (DevOps Project...)"
APP_TAGLINE = "⚙️ CI/CD-powered AI resume screening & candidate matching"

# ----- Custom CSS: futuristic professional design system -----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
    }

    :root {
        --c-cyan: 34, 211, 238;
        --c-violet: 139, 92, 246;
        --c-blue: 59, 130, 246;
        --c-emerald: 52, 211, 153;
        --c-slate: 7, 10, 22;
        --glass-bg: rgba(15, 23, 42, 0.55);
        --glass-border: rgba(148, 163, 184, 0.14);
        --text-primary: #f8fafc;
        --text-muted: #94a3b8;
    }

    @keyframes floatGlow {
        0%, 100% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.45; }
        50%      { transform: translate3d(0, -12px, 0) scale(1.02); opacity: 0.65; }
    }

    @keyframes sheen {
        0%   { transform: translateX(-50%) rotate(14deg); opacity: 0; }
        30%  { opacity: 0.35; }
        100% { transform: translateX(160%) rotate(14deg); opacity: 0; }
    }

    @keyframes borderShift {
        0%, 100% { background-position: 0% 50%; }
        50%      { background-position: 100% 50%; }
    }

    @keyframes pulseDot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(var(--c-cyan), 0.5); }
        50%      { opacity: 0.85; box-shadow: 0 0 0 6px rgba(var(--c-cyan), 0); }
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .stApp {
        background:
            linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
            radial-gradient(900px 520px at 8% 0%, rgba(var(--c-cyan), 0.22) 0%, transparent 58%),
            radial-gradient(1000px 620px at 92% 8%, rgba(var(--c-violet), 0.18) 0%, transparent 60%),
            radial-gradient(800px 500px at 50% 100%, rgba(var(--c-blue), 0.10) 0%, transparent 55%),
            linear-gradient(180deg, #050810 0%, #0a0f1e 45%, #0f172a 100%);
        background-size: 48px 48px, 48px 48px, auto, auto, auto, auto;
    }

    .stApp:before {
        content: "";
        position: fixed;
        inset: -15%;
        background:
            radial-gradient(480px 300px at 15% 20%, rgba(var(--c-cyan), 0.18) 0%, transparent 62%),
            radial-gradient(520px 320px at 85% 15%, rgba(var(--c-violet), 0.16) 0%, transparent 62%);
        filter: blur(12px);
        opacity: 0.6;
        pointer-events: none;
        z-index: 0;
        animation: floatGlow 12s ease-in-out infinite;
    }

    .main { position: relative; z-index: 1; }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1180px;
        animation: fadeUp 0.45s ease-out;
    }

    /* Brand mark */
    .brand-mark {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.05rem;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }

    .brand-mark .logo-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.75rem;
        height: 1.75rem;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(var(--c-cyan), 0.9), rgba(var(--c-violet), 0.85));
        font-size: 0.75rem;
        color: #fff;
        box-shadow: 0 0 20px rgba(var(--c-cyan), 0.35);
    }

    .glass-panel {
        border: 1px solid var(--glass-border);
        background: linear-gradient(155deg, rgba(30, 41, 59, 0.52) 0%, rgba(2, 6, 23, 0.78) 100%);
        border-radius: 16px;
        padding: 1.5rem 1.65rem;
        box-shadow: 0 24px 60px -28px rgba(0, 0, 0, 0.75), inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }

    .glass-panel:before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(var(--c-cyan), 0.5), rgba(var(--c-violet), 0.4), transparent);
    }

    .login-card {
        border: 1px solid transparent;
        background:
            linear-gradient(155deg, rgba(30, 41, 59, 0.62) 0%, rgba(2, 6, 23, 0.82) 100%) padding-box,
            linear-gradient(120deg, rgba(var(--c-cyan),0.55), rgba(var(--c-violet),0.5), rgba(var(--c-blue),0.4)) border-box;
        background-size: 100% 100%, 280% 280%;
        animation: borderShift 12s ease-in-out infinite;
        padding: 2.25rem 2rem 2rem;
        border-radius: 18px;
        box-shadow: 0 32px 80px -30px rgba(0,0,0,0.8);
        margin: 0.5rem auto 1rem auto;
        max-width: 440px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .login-card:after {
        content: "";
        position: absolute;
        top: -45%;
        left: -70%;
        width: 65%;
        height: 210%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
        transform: rotate(14deg);
        animation: sheen 8s ease-in-out infinite;
        pointer-events: none;
    }

    .login-card > * { position: relative; z-index: 2; }

    .login-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.35rem;
        text-align: center;
        letter-spacing: -0.02em;
    }

    .login-subtitle {
        color: var(--text-muted);
        font-size: 0.92rem;
        text-align: center;
        margin-bottom: 1.35rem;
    }

    .hero {
        border: 1px solid var(--glass-border);
        background: linear-gradient(155deg, rgba(30, 41, 59, 0.48) 0%, rgba(2, 6, 23, 0.74) 100%);
        border-radius: 20px;
        padding: 2rem 2.25rem;
        box-shadow: 0 28px 70px -32px rgba(0,0,0,0.65);
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }

    .hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(420px 240px at 12% 25%, rgba(var(--c-cyan), 0.20) 0%, transparent 58%),
            radial-gradient(420px 240px at 88% 30%, rgba(var(--c-violet), 0.18) 0%, transparent 58%);
        pointer-events: none;
    }

    .hero > * { position: relative; z-index: 1; }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: rgba(var(--c-cyan), 1);
        margin-bottom: 0.75rem;
    }

    .hero-eyebrow .live-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: rgba(var(--c-emerald), 1);
        animation: pulseDot 2s ease-in-out infinite;
    }

    .hero h1 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        margin: 0 0 0.5rem 0 !important;
        padding: 0 !important;
        background: linear-gradient(92deg, #f8fafc 0%, rgba(var(--c-cyan), 1) 45%, rgba(var(--c-violet), 1) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #cbd5e1;
        margin: 0 0 1.1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 640px;
    }

    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #e2e8f0;
        font-weight: 500;
        font-size: 0.82rem;
        transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }

    .chip:hover {
        transform: translateY(-1px);
        border-color: rgba(var(--c-cyan), 0.4);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }

    .chip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(var(--c-cyan),1), rgba(var(--c-violet),1));
    }

    .page-header {
        margin-bottom: 1.25rem;
    }

    .page-header h1 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.35rem !important;
        padding-bottom: 0 !important;
    }

    .page-header .subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin: 0;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.85rem;
        margin: 0.5rem 0 1.25rem 0;
    }

    .stat-card {
        background: linear-gradient(160deg, rgba(30,41,59,0.55) 0%, rgba(2,6,23,0.75) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 1.1rem 1.15rem;
        backdrop-filter: blur(12px);
        transition: transform 180ms ease, border-color 180ms ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(var(--c-cyan), 0.28);
    }

    .stat-card .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }

    .stat-card .value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }

    .stat-card .value.accent {
        background: linear-gradient(90deg, rgba(var(--c-cyan),1), rgba(var(--c-violet),1));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
    }

    h1 { color: var(--text-primary) !important; font-weight: 700 !important; }
    h2, h3 { color: #e2e8f0 !important; font-weight: 600 !important; }

    [data-testid="stWidgetLabel"] p,
    label,
    .stMarkdown, .stCaption {
        color: #cbd5e1;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #070a14 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }

    .sidebar-brand {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin-bottom: 0.15rem;
    }

    .sidebar-tagline {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    .role-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.28rem 0.8rem;
        border-radius: 9999px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(var(--c-cyan), 0.25);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(var(--c-cyan), 1);
    }

    .role-banner {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background: linear-gradient(90deg, rgba(var(--c-cyan),0.08), rgba(var(--c-violet),0.06));
        border: 1px solid rgba(148,163,184,0.15);
        font-size: 0.88rem;
        color: #e2e8f0;
        margin-bottom: 1rem;
    }

    .role-banner .banner-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(var(--c-cyan),0.25), rgba(var(--c-violet),0.2));
        font-size: 0.85rem;
        font-weight: 700;
        color: rgba(var(--c-cyan), 1);
        font-family: 'JetBrains Mono', monospace;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 0.52rem 1.25rem;
        transition: all 0.2s;
        border: 1px solid rgba(var(--c-cyan), 0.35);
        background: linear-gradient(135deg, rgba(var(--c-cyan),0.35), rgba(var(--c-violet),0.35));
        color: #f8fafc !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 32px rgba(var(--c-cyan), 0.18);
        border-color: rgba(var(--c-cyan), 0.55);
    }

    .stButton > button:active { transform: translateY(0) scale(0.99); }

    /* Inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    textarea {
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        background: rgba(2, 6, 23, 0.55) !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stTextInput"] input::placeholder { color: rgba(148, 163, 184, 0.65) !important; }

    [data-testid="stTextInput"] input:focus {
        border-color: rgba(var(--c-cyan), 0.55) !important;
        box-shadow: 0 0 0 3px rgba(var(--c-cyan), 0.15) !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(30,41,59,0.5) 0%, rgba(2,6,23,0.7) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        backdrop-filter: blur(10px);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, rgba(var(--c-cyan),1), rgba(var(--c-violet),1));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* File uploader — dark glass */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.45);
        border-radius: 14px;
        padding: 1.15rem;
        border: 1px dashed rgba(var(--c-cyan), 0.28);
        transition: border-color 0.2s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(var(--c-cyan), 0.45);
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: #cbd5e1 !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.35);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, rgba(var(--c-cyan),1), rgba(var(--c-violet),1)) !important;
        border-radius: 999px;
    }

    /* Radio nav in sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
        transition: background 0.15s;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,0.04);
    }

    hr {
        margin: 1.25rem 0 !important;
        border-color: rgba(148, 163, 184, 0.15) !important;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.85rem;
        margin: 1.25rem 0;
    }

    .feature-item {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid var(--glass-border);
    }

    .feature-item h4 {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 0 0 0.35rem 0;
    }

    .feature-item p {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin: 0;
        line-height: 1.5;
    }

    .footer {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: rgba(148, 163, 184, 0.65);
        font-size: 0.72rem;
        margin-top: 3rem;
        padding-top: 1.25rem;
        border-top: 1px solid rgba(148, 163, 184, 0.12);
    }

    .footer strong {
        background: linear-gradient(90deg, rgba(var(--c-cyan), 1), rgba(var(--c-violet), 1));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .badge {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-excellent { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .badge-good      { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
    .badge-fair      { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.28); }
    .badge-poor      { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.28); }

    #MainMenu, footer, header { visibility: hidden; }

    @media (prefers-reduced-motion: reduce) {
        .stApp:before, .login-card:after, .login-card, .hero, .hero-eyebrow .live-dot {
            animation: none !important;
        }
        .main .block-container { animation: none; }
    }
</style>
""", unsafe_allow_html=True)

# ----- Demo credentials (in production use a real auth/DB) -----
DEMO_CREDENTIALS = {
    "admin": "admin123",
    "recruiter": "recruiter123",
    "jobseeker": "jobseeker123",
}

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "shortlisted" not in st.session_state:
    st.session_state.shortlisted = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_jd_text" not in st.session_state:
    st.session_state.last_jd_text = None
if "last_job_role" not in st.session_state:
    st.session_state.last_job_role = ""
if "admin_activity" not in st.session_state:
    st.session_state.admin_activity = []
if "js_last_jd" not in st.session_state:
    st.session_state.js_last_jd = None
if "js_last_resume" not in st.session_state:
    st.session_state.js_last_resume = None
if "js_last_missing" not in st.session_state:
    st.session_state.js_last_missing = None
if "js_show_ai_suggestions" not in st.session_state:
    st.session_state.js_show_ai_suggestions = False


def log_activity(role, username, action, detail=""):
    st.session_state.admin_activity.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "role": role,
        "user": username,
        "action": action,
        "detail": detail,
    })


def extract_text_from_pdf(uploaded_file, max_pages=5):
    if uploaded_file is None:
        return None
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        uploaded_file.seek(0)
        return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None


RESUME_KEYWORDS = {
    "experience", "education", "skills", "summary", "objective", "work",
    "employment", "projects", "certifications", "achievements", "profile",
    "references", "languages", "internship", "volunteer", "university",
    "college", "degree", "bachelor", "master", "gpa", "resume", "cv",
}

def is_resume_pdf(text):
    """Return True if the extracted text looks like a resume."""
    if not text:
        return False
    lower = text.lower()
    hits = sum(1 for kw in RESUME_KEYWORDS if kw in lower)
    return hits >= 3


def get_match(JD_txt, resume_txt):
    if not JD_txt or not resume_txt:
        return None
    content = [JD_txt, resume_txt]
    cv = CountVectorizer()
    matrix = cv.fit_transform(content)
    similarity = cosine_similarity(matrix)[0][1] * 100
    return round(similarity, 2)


def match_label(percent):
    if percent is None:
        return "—", "gray"
    if percent >= 70:
        return "Strong match", "green"
    if percent >= 50:
        return "Moderate match", "orange"
    return "Low match", "red"


def extract_keywords(text, min_length=3):
    """
    Extract skill-like tokens from text.
    - Filters common English stopwords + extra generic job words
    - Keeps meaningful unigrams and bigrams (e.g., 'machine learning', 'power bi')
    """
    if not text:
        return set()

    extra_stop = {
        # generic verbs/adjectives/role words that are not "skills"
        "develop", "development", "quality", "decision", "support", "analyst", "analysis",
        "learning", "work", "present", "analytical", "stakeholder", "stakeholders",
        "modeling", "cross", "functional", "responsible", "responsibilities",
        "requirements", "required", "preferred", "role", "job", "position", "candidate",
        "experience", "years", "year", "must", "should", "ability", "strong", "good",
        "excellent", "team", "teams", "collaborate", "collaboration", "communication",
        "understanding", "knowledge", "skills", "skill", "using", "use", "within",
        "including", "etc", "also", "will",
    }

    text = text.lower()
    # keep words like "powerbi" and "c++" won't be captured; keep alnum tokens here
    raw_words = re.findall(r"[a-z0-9]+", text)
    words = [
        w for w in raw_words
        if len(w) >= min_length and w not in ENGLISH_STOP_WORDS and w not in extra_stop
    ]

    # unigrams
    terms = set(words)

    # bigrams from filtered words (more skill-like phrases)
    for a, b in zip(words, words[1:]):
        if a == b:
            continue
        terms.add(f"{a} {b}")

    return terms


def gap_analysis(jd_text, resume_text, top_n=25):
    if not jd_text or not resume_text:
        return []
    jd_terms = extract_keywords(jd_text)
    resume_terms = extract_keywords(resume_text)
    missing = jd_terms - resume_terms

    # Prefer phrases + tech-like terms, downrank single generic leftovers
    jd_lower = jd_text.lower()
    scored = []
    for t in missing:
        base = jd_lower.count(t)
        bonus = 2 if " " in t else 0  # prefer phrases
        scored.append((base + bonus, base, t))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    # keep only those that appear at least once in JD
    filtered = [t for _, base, t in scored if base > 0]
    return filtered[:top_n]

COURSE_CATALOG = [
    {
        "tags": {"python", "pandas", "numpy"},
        "title": "Python for Data Science",
        "provider": "Coursera / IBM",
        "type": "Course",
        "url": "https://www.coursera.org/learn/python-for-applied-data-science-ai",
    },
    {
        "tags": {"sql", "postgres", "mysql", "database"},
        "title": "SQL for Data Analysis",
        "provider": "Mode Analytics",
        "type": "Course",
        "url": "https://mode.com/sql-tutorial/",
    },
    {
        "tags": {"excel", "spreadsheets"},
        "title": "Excel Skills for Business",
        "provider": "Coursera / Macquarie University",
        "type": "Course",
        "url": "https://www.coursera.org/specializations/excel",
    },
    {
        "tags": {"machine", "learning", "ml", "sklearn", "scikit"},
        "title": "Machine Learning",
        "provider": "Coursera / Andrew Ng",
        "type": "Course",
        "url": "https://www.coursera.org/learn/machine-learning",
    },
    {
        "tags": {"deep", "learning", "neural", "tensorflow", "pytorch"},
        "title": "Deep Learning Specialization",
        "provider": "Coursera / DeepLearning.AI",
        "type": "Specialization",
        "url": "https://www.coursera.org/specializations/deep-learning",
    },
    {
        "tags": {"nlp", "bert", "transformers", "language"},
        "title": "Natural Language Processing Specialization",
        "provider": "Coursera / DeepLearning.AI",
        "type": "Specialization",
        "url": "https://www.coursera.org/specializations/natural-language-processing",
    },
    {
        "tags": {"powerbi", "bi", "dashboard"},
        "title": "Power BI Guided Learning",
        "provider": "Microsoft Learn",
        "type": "Learning path",
        "url": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi/",
    },
    {
        "tags": {"tableau", "dashboard"},
        "title": "Data Visualization with Tableau",
        "provider": "Coursera / UC Davis",
        "type": "Course",
        "url": "https://www.coursera.org/learn/datavisualization",
    },
    {
        "tags": {"aws", "cloud"},
        "title": "AWS Cloud Practitioner Essentials",
        "provider": "AWS",
        "type": "Course",
        "url": "https://explore.skillbuilder.aws/learn/course/134/aws-cloud-practitioner-essentials",
    },
    {
        "tags": {"azure", "cloud"},
        "title": "Azure Fundamentals",
        "provider": "Microsoft Learn",
        "type": "Learning path",
        "url": "https://learn.microsoft.com/en-us/training/paths/azure-fundamentals/",
    },
    {
        "tags": {"docker", "containers"},
        "title": "Docker — Get Started",
        "provider": "Docker Docs",
        "type": "Docs",
        "url": "https://docs.docker.com/get-started/",
    },
    {
        "tags": {"kubernetes", "k8s"},
        "title": "Kubernetes Basics",
        "provider": "Kubernetes.io",
        "type": "Tutorial",
        "url": "https://kubernetes.io/docs/tutorials/kubernetes-basics/",
    },
    {
        "tags": {"react", "frontend", "javascript"},
        "title": "React Foundations",
        "provider": "Next.js Learn",
        "type": "Course",
        "url": "https://nextjs.org/learn/react-foundations",
    },
    {
        "tags": {"javascript", "js"},
        "title": "JavaScript Guide",
        "provider": "MDN Web Docs",
        "type": "Docs",
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
    },
    {
        "tags": {"git", "github", "version"},
        "title": "Introduction to Git and GitHub",
        "provider": "Coursera / Google",
        "type": "Course",
        "url": "https://www.coursera.org/learn/introduction-git-github",
    },
    {
        "tags": {"dsa", "algorithms", "data", "structures"},
        "title": "Data Structures and Algorithms",
        "provider": "Coursera / UC San Diego",
        "type": "Specialization",
        "url": "https://www.coursera.org/specializations/data-structures-algorithms",
    },
    {
        "tags": {"system", "design", "scalability"},
        "title": "System Design Basics",
        "provider": "GitHub (free)",
        "type": "Resource",
        "url": "https://github.com/donnemartin/system-design-primer",
    },
]


def suggest_courses_from_gaps(missing_keywords, max_recs=8):
    """Recommend courses/resources based on missing JD keywords."""
    if not missing_keywords:
        return []
    missing = {m.lower() for m in missing_keywords}
    scored = []
    for item in COURSE_CATALOG:
        score = len(item["tags"] & missing)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], x[1]["title"]))
    recs = []
    seen = set()
    for _, it in scored:
        k = (it["title"], it["provider"])
        if k in seen:
            continue
        seen.add(k)
        recs.append(it)
        if len(recs) >= max_recs:
            break
    return recs


def _google_query_url(q: str) -> str:
    return "https://www.google.com/search?q=" + re.sub(r"\s+", "+", q.strip())


def suggest_course_search_links(missing_keywords, max_skills=8):
    """
    Provide Google search links for courses per missing skill.
    This guarantees suggestions even when our curated catalog doesn’t match.
    """
    if not missing_keywords:
        return []
    skills = [s for s in missing_keywords if s][:max_skills]
    links = []
    for s in skills:
        q = f"{s} course online certificate"
        links.append({"skill": s, "label": f"Google courses for {s}", "url": _google_query_url(q)})
    return links


def ats_compatibility_label(percent):
    if percent is None:
        return "N/A"
    if percent >= 80:
        return "Excellent"
    if percent >= 60:
        return "Good"
    if percent >= 40:
        return "Fair"
    return "Poor"


def simple_match_chart(df: pd.DataFrame, title: str = "Match % by candidate"):
    """Simple sorted bar chart with readable value labels."""
    if df is None or df.empty:
        return
    if "Candidate" not in df.columns or "Match %" not in df.columns:
        return

    chart_df = df[["Candidate", "Match %"]].copy()
    chart_df["Match %"] = pd.to_numeric(chart_df["Match %"], errors="coerce").fillna(0)
    chart_df = chart_df.sort_values("Match %", ascending=False)

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Candidate:N", sort=None, title="Candidate", axis=alt.Axis(labelColor="#94A3B8", titleColor="#CBD5E1")),
            y=alt.Y("Match %:Q", title="Match %", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelColor="#94A3B8", titleColor="#CBD5E1", gridColor="rgba(148,163,184,0.15)")),
            color=alt.Color("Match %:Q", scale=alt.Scale(scheme="turbo"), legend=None),
            tooltip=[alt.Tooltip("Candidate:N"), alt.Tooltip("Match %:Q", format=".0f")],
        )
    )

    labels = (
        alt.Chart(chart_df)
        .mark_text(dy=-10, color="#F1F5F9", fontSize=12, fontWeight="bold")
        .encode(x=alt.X("Candidate:N", sort=None), y=alt.Y("Match %:Q"), text=alt.Text("Match %:Q", format=".0f"))
    )

    chart = (bars + labels).properties(
        title=alt.TitleParams(text=title, color="#F1F5F9", fontSize=16, font="Outfit"),
        height=300,
        background="transparent",
    ).configure_view(strokeWidth=0, fill="transparent").configure_axis(domainColor="rgba(148,163,184,0.2)")

    st.altair_chart(chart.interactive(), use_container_width=True)


# ==================== LOGIN SCREEN ====================
if not st.session_state.logged_in:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow"><span class="live-dot"></span> 🚀 DevOps Pipeline · AI Screening v2.0</div>
            <h1>{APP_NAME}</h1>
            <p>{APP_TAGLINE} — Screen candidates with 📈 ATS scoring, 🧩 skill gap analysis, and 🎓 AI learning paths in one dashboard.</p>
            <div class="chips">
                <span class="chip">📈 Match analytics</span>
                <span class="chip">🧩 Skill gap engine</span>
                <span class="chip">🎓 AI course mapping</span>
                <span class="chip">📄 CSV exports</span>
                <span class="chip">🔐 Role-based access</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_sp1, c_mid, c_sp2 = st.columns([1, 1.25, 1])
    with c_mid:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='login-title'><span class='brand-mark'><span class='logo-icon'>🤖</span>DevOps AI Resume Screening</span></p>",
            unsafe_allow_html=True,
        )
        st.markdown("<p class='login-subtitle'>🔐 Sign in to your screening workspace</p>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            role = st.selectbox("Access role", ["Recruiter", "Job Seeker"], index=0)
            submitted = st.form_submit_button("🚀 Sign in", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                un = username.strip().lower()
                pw = password
                ok = False
                if un == "admin" and pw == DEMO_CREDENTIALS.get("admin"):
                    st.session_state.logged_in = True
                    st.session_state.user_role = "admin"
                    st.session_state.username = username
                    log_activity("Admin", username, "Login", "Admin panel access")
                    ok = True
                elif role == "Recruiter" and un == "recruiter" and pw == DEMO_CREDENTIALS.get("recruiter"):
                    st.session_state.logged_in = True
                    st.session_state.user_role = "recruiter"
                    st.session_state.username = username
                    log_activity("Recruiter", username, "Login", "")
                    ok = True
                elif role == "Job Seeker" and un == "jobseeker" and pw == DEMO_CREDENTIALS.get("jobseeker"):
                    st.session_state.logged_in = True
                    st.session_state.user_role = "job_seeker"
                    st.session_state.username = username
                    log_activity("Job Seeker", username, "Login", "")
                    ok = True
                if ok:
                    _rerun()
                else:
                    st.error("Invalid username or password for the selected role.")

        with st.expander("Demo credentials"):
            st.code(
                "admin / admin123\nrecruiter / recruiter123\njobseeker / jobseeker123"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-item">
                <h4>📈 ATS Match Scoring</h4>
                <p>Cosine similarity engine compares job descriptions against resumes for instant compatibility scores.</p>
            </div>
            <div class="feature-item">
                <h4>🧩 Skill Gap Analysis</h4>
                <p>Identify missing keywords and competencies to prioritize upskilling or candidate outreach.</p>
            </div>
            <div class="feature-item">
                <h4>🎓 AI Learning Paths</h4>
                <p>Curated course recommendations mapped to detected skill gaps for job seekers.</p>
            </div>
            <div class="feature-item">
                <h4>👔 Recruiter Dashboard</h4>
                <p>Batch-screen resumes, visualize rankings, export CSV reports, and manage shortlists.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()
    st.stop()

# ==================== SIDEBAR (logged in) ====================
with st.sidebar:
    st.markdown("<p class='sidebar-brand'>🤖 DevOps AI Resume Screening</p>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-tagline'>⚙️ CI/CD-powered talent matching</p>", unsafe_allow_html=True)

    role = st.session_state.user_role
    if role == "recruiter":
        role_label = "👔 Recruiter"
    elif role == "job_seeker":
        role_label = "📄 Job Seeker"
    else:
        role_label = "🛡️ Admin"

    st.markdown(f"**{st.session_state.username}**")
    st.markdown(
        f"<span class='role-badge'>{role_label}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if st.session_state.user_role == "admin":
        page = st.radio("Navigation", ["Dashboard", "Activity log", "User info", "Settings"], label_visibility="collapsed")
    elif st.session_state.user_role == "recruiter":
        page = st.radio("Navigation", ["Screening", "Dashboard", "Gap analysis"], label_visibility="collapsed")
    else:
        page = st.radio("Navigation", ["Check match", "Gap analysis"], label_visibility="collapsed")

    st.markdown("---")
    if st.sidebar.button("🚪 Sign out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.shortlisted = []
        _rerun()

# ==================== ADMIN PANEL ====================
if st.session_state.user_role == "admin":
    render_page_header("🛡️ Admin Console", "System overview, activity monitoring, and configuration", "🛡️")

    if page == "Dashboard":
        render_stat_cards([
            ("Activity entries", str(len(st.session_state.admin_activity)), True),
            ("User roles", "3", False),
            ("System status", "Online", True),
        ])
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### Platform overview")
        st.info("Navigate to **Activity log** in the sidebar to review sign-ins and screening events across all roles.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Activity log":
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### Activity log")
        if not st.session_state.admin_activity:
            st.caption("No activity recorded yet.")
        else:
            log_df = pd.DataFrame(st.session_state.admin_activity)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "User info":
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### Demo accounts")
        st.dataframe(pd.DataFrame([
            {"Username": "admin", "Role": "Admin", "Purpose": "Admin panel"},
            {"Username": "recruiter", "Role": "Recruiter", "Purpose": "Screen multiple resumes"},
            {"Username": "jobseeker", "Role": "Job Seeker", "Purpose": "Check resume vs JD"},
        ]), use_container_width=True, hide_index=True)
        st.caption("Passwords: admin123 · recruiter123 · jobseeker123")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Settings":
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### Settings")
        st.caption("Configuration placeholder. In production, add theme, rate limits, and integration settings.")
        st.markdown("</div>", unsafe_allow_html=True)

    render_footer()
    st.stop()

# ==================== RECRUITER ====================
if st.session_state.user_role == "recruiter":
    if page == "Dashboard":
        render_page_header("📊 Analytics Dashboard", "Candidate match scores and ATS compatibility at a glance", "📊")
        if st.session_state.last_results:
            df = pd.DataFrame([
                {
                    "Job role": st.session_state.last_job_role or "—",
                    "Candidate": r["name"],
                    "Match %": r["match"] or 0,
                    "ATS": r.get("ats", "—"),
                }
                for r in st.session_state.last_results
            ])
            avg_match = round(df["Match %"].mean(), 1)
            top_match = df["Match %"].max()
            render_stat_cards([
                ("Candidates screened", str(len(df)), False),
                ("Average match", f"{avg_match}%", True),
                ("Top score", f"{top_match}%", True),
            ])
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            with st.expander("Understanding Match % and ATS labels", expanded=False):
                st.markdown(
                    "- **Match %**: Text similarity between the job description and each resume.\n"
                    "- **ATS**: Compatibility tier — Excellent, Good, Fair, or Poor."
                )
            st.dataframe(df, use_container_width=True, hide_index=True)
            simple_match_chart(df, title="Candidate ranking by match score")
            st.download_button(
                "📥 Export dashboard (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resume_matching_dashboard.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Run a screening session from **Screening** to populate this dashboard.")

    elif page == "Gap analysis":
        render_page_header("🧩 Skill Gap Analysis", "Missing competencies per candidate from your last screening run", "🧩")
        if st.session_state.last_results and st.session_state.last_jd_text:
            st.caption("Keywords present in the JD but absent or weak in each resume.")
            for r in st.session_state.last_results:
                res_text = r.get("resume_text")
                if not res_text:
                    continue
                missing = gap_analysis(st.session_state.last_jd_text, res_text, top_n=20)
                with st.expander(f"{r['name']} — {r['match']}% match"):
                    st.write(", ".join(missing) if missing else "No significant gaps detected.")
                    st.markdown("**Recommended learning resources**")
                    recs = suggest_courses_from_gaps(missing, max_recs=6)
                    if recs:
                        for it in recs:
                            st.markdown(f"- [{it['title']}]({it['url']}) — {it['provider']} ({it['type']})")
                    else:
                        st.caption("No course suggestions needed.")
        else:
            st.info("Complete a **Screening** session first to unlock gap analysis.")

    else:  # Screening
        render_page_header("📂 Resume Screening", "Upload a job description and batch-process candidate resumes", "📂")
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload documents")
        job_role = st.text_input("Job role", value=st.session_state.last_job_role, placeholder="e.g. Data Analyst · ML Engineer · DevOps Lead")
        uploaded_jd = st.file_uploader("Job description (PDF)", type="pdf", key="jd")
        uploaded_resumes = st.file_uploader("Resumes (PDF, multiple allowed)", type="pdf", accept_multiple_files=True, key="resumes")
        
        if uploaded_jd and uploaded_resumes:
            jd_text = extract_text_from_pdf(uploaded_jd)
            if not jd_text:
                st.warning("Could not read the job description PDF.")
            else:
                with st.expander("Screening best practices", expanded=False):
                    st.markdown(
                        "- Use a **detailed JD PDF** for accurate keyword extraction.\n"
                        "- Upload **multiple resumes** to compare candidates side-by-side.\n"
                        "- Prioritize candidates with **Excellent/Good ATS** ratings."
                    )
                if st.button("⚡ Run screening analysis", use_container_width=True):
                    results = []
                    progress = st.progress(0)
                    for idx, res_file in enumerate(uploaded_resumes):
                        progress.progress((idx + 1) / len(uploaded_resumes))
                        res_text = extract_text_from_pdf(res_file)
                        name = res_file.name
                        if res_text and not is_resume_pdf(res_text):
                            st.warning(f"**{name}** does not appear to be a valid resume PDF.")
                            continue
                        match_pct = get_match(jd_text, res_text) if res_text else None
                        label, _ = match_label(match_pct)
                        ats = ats_compatibility_label(match_pct)
                        results.append({
                            "name": name, "match": match_pct, "label": label,
                            "ats": ats, "resume_text": res_text,
                        })
                    progress.empty()
                    st.session_state.last_results = results
                    st.session_state.last_jd_text = jd_text
                    st.session_state.last_job_role = job_role.strip()
                    log_activity("Recruiter", st.session_state.username, "Screening", f"{len(results)} resumes · role={st.session_state.last_job_role or '—'}")
                    _rerun()
            
            if st.session_state.last_results:
                st.markdown("#### Screening results")
                df = pd.DataFrame([
                    {
                        "Job role": st.session_state.last_job_role or "—",
                        "Candidate": r["name"],
                        "Match %": r["match"] or 0,
                        "ATS": r.get("ats", "—"),
                    }
                    for r in st.session_state.last_results
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
                simple_match_chart(df, title="Candidate ranking by match score")
                st.download_button(
                    "📥 Export results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="screening_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                
                st.markdown("#### Shortlist management")
                for r in st.session_state.last_results:
                    if r["match"] is not None and r["match"] >= 50:
                        if st.button(f"⭐ Shortlist: {r['name']}", key=f"short_{r['name']}"):
                            if r["name"] not in st.session_state.shortlisted:
                                st.session_state.shortlisted.append(r["name"])
                                _rerun()
                if st.session_state.shortlisted:
                    st.write("**Shortlisted:** " + ", ".join(st.session_state.shortlisted))
                    shortlist_df = pd.DataFrame([{"Candidate": n} for n in st.session_state.shortlisted])
                    st.download_button(
                        "📥 Export shortlist (CSV)",
                        data=shortlist_df.to_csv(index=False).encode("utf-8"),
                        file_name="shortlist.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    if st.button("🗑️ Clear shortlist"):
                        st.session_state.shortlisted = []
                        _rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    render_footer()
    st.stop()

# ==================== JOB SEEKER ====================
if st.session_state.user_role == "job_seeker" and page == "Gap analysis":
    render_page_header("🧩 Skill Gap Report", "Keywords from the JD missing or underrepresented in your resume", "🧩")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    if st.session_state.js_last_jd and st.session_state.js_last_resume:
        missing = gap_analysis(st.session_state.js_last_jd, st.session_state.js_last_resume, top_n=25)
        st.caption("Terms extracted from the job description not found in your resume.")
        if missing:
            st.write(", ".join(missing))
        else:
            st.success("Your resume covers the JD keywords comprehensively.")
    else:
        st.info("Complete a match analysis under **Check match** to generate your gap report.")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    render_page_header("🎯 Resume Match Analysis", "Compare your resume against a target job description", "🎯")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    with st.expander("How matching works", expanded=False):
        st.markdown(
            "- Text is extracted from both PDF documents.\n"
            "- A **Match %** (0–100) is computed via cosine similarity.\n"
            "- An **ATS label** provides a quick compatibility tier."
        )
    uploaded_jd = st.file_uploader("Job description (PDF)", type="pdf", key="jd_js")
    uploaded_resume = st.file_uploader("Your resume (PDF)", type="pdf", key="res_js")
    click = st.button("🔍 Analyze match", use_container_width=True)

    job_description = extract_text_from_pdf(uploaded_jd) if uploaded_jd else None
    resume = extract_text_from_pdf(uploaded_resume) if uploaded_resume else None

    if uploaded_resume and resume is not None and not is_resume_pdf(resume):
        st.warning("The uploaded file does not appear to be a valid resume PDF.")
        resume = None

    if click:
        if not job_description or not resume:
            st.warning("Please upload both a job description and a valid resume.")
        else:
            st.session_state.js_last_jd = job_description
            st.session_state.js_last_resume = resume
            match = get_match(job_description, resume)
            label, _ = match_label(match)
            ats = ats_compatibility_label(match)
            log_activity("Job Seeker", st.session_state.username, "Match check", f"{match}%")

            st.markdown("---")
            render_stat_cards([
                ("Match score", f"{match}%", True),
                ("Assessment", label, False),
                ("ATS tier", ats, False),
            ])
            if match >= 70:
                st.success("Strong fit — your profile aligns well with this role.")
            elif match >= 50:
                st.info("Moderate fit — consider highlighting relevant experience and skills.")
            else:
                st.warning("Low fit — review missing keywords in the Gap analysis tab.")

            st.markdown("#### Missing skills")
            missing = gap_analysis(job_description, resume, top_n=25)
            st.session_state.js_last_missing = missing
            st.session_state.js_show_ai_suggestions = False
            if missing:
                st.write(", ".join(missing[:12]))
                st.caption("Top missing keywords. Use AI recommendations below for targeted upskilling.")
            else:
                st.success("No significant gaps detected.")
                st.caption("AI recommendations will be limited when no gaps are found.")

    if st.session_state.js_last_missing is not None:
        st.markdown("---")
        st.markdown("### 🤖 AI learning recommendations")
        st.caption("Generate course suggestions based on your skill gaps.")

        if st.button("🤖 Generate AI recommendations", key="ask_ai_courses", use_container_width=True):
            st.session_state.js_show_ai_suggestions = True
            log_activity(
                "Job Seeker",
                st.session_state.username,
                "AI course suggestions",
                f"{len(st.session_state.js_last_missing or [])} gaps",
            )

        if st.session_state.js_show_ai_suggestions:
            missing_now = st.session_state.js_last_missing or []
            st.markdown("#### Recommended courses")

            curated = suggest_courses_from_gaps(missing_now, max_recs=8)
            google_links = suggest_course_search_links(missing_now, max_skills=8)

            if curated:
                st.markdown("**Curated resources**")
                for it in curated:
                    st.markdown(f"- [{it['title']}]({it['url']}) — {it['provider']} ({it['type']})")

            if google_links:
                st.markdown("**Additional search links**")
                for l in google_links:
                    st.markdown(f"- [{l['label']}]({l['url']})")

            if not curated and not google_links:
                st.info("No recommendations needed — your resume already covers the JD keywords.")

    st.markdown("</div>", unsafe_allow_html=True)

render_footer()
print("CI/CD Local Test Baseline")
