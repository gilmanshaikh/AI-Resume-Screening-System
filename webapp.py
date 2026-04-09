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

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Resume Screening | Gilman",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- Custom CSS for modern UI -----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    :root {
        --c-cyan: 34, 211, 238;
        --c-violet: 168, 85, 247;
        --c-amber: 251, 191, 36;
        --c-slate: 15, 23, 42;
    }

    @keyframes floatGlow {
        0%   { transform: translate3d(0, 0, 0) scale(1); opacity: 0.55; }
        50%  { transform: translate3d(0, -10px, 0) scale(1.03); opacity: 0.70; }
        100% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.55; }
    }

    @keyframes sheen {
        0%   { transform: translateX(-40%) rotate(12deg); opacity: 0.0; }
        25%  { opacity: 0.55; }
        55%  { opacity: 0.25; }
        100% { transform: translateX(140%) rotate(12deg); opacity: 0.0; }
    }

    @keyframes borderShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background:
            radial-gradient(900px 520px at 10% 0%, rgba(var(--c-cyan), 0.28) 0%, rgba(var(--c-slate), 0.0) 60%),
            radial-gradient(1000px 620px at 90% 10%, rgba(var(--c-violet), 0.22) 0%, rgba(var(--c-slate), 0.0) 62%),
            radial-gradient(900px 600px at 50% 110%, rgba(var(--c-amber), 0.12) 0%, rgba(var(--c-slate), 0.0) 55%),
            linear-gradient(180deg, #070a16 0%, #0b1220 40%, #0f172a 100%);
    }

    /* Animated ambient glow (subtle) */
    .stApp:before {
        content: "";
        position: fixed;
        inset: -20%;
        background:
            radial-gradient(520px 320px at 18% 18%, rgba(var(--c-cyan), 0.22) 0%, rgba(0,0,0,0) 62%),
            radial-gradient(560px 340px at 78% 22%, rgba(var(--c-violet), 0.20) 0%, rgba(0,0,0,0) 62%),
            radial-gradient(520px 360px at 50% 92%, rgba(var(--c-amber), 0.10) 0%, rgba(0,0,0,0) 58%);
        filter: blur(8px);
        opacity: 0.55;
        pointer-events: none;
        z-index: 0;
        animation: floatGlow 10s ease-in-out infinite;
    }

    .main {
        position: relative;
        z-index: 1;
    }
    
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Login card */
    .login-card {
        border: 1px solid transparent;
        background:
            linear-gradient(145deg, rgba(30, 41, 59, 0.58) 0%, rgba(2, 6, 23, 0.76) 100%) padding-box,
            linear-gradient(90deg, rgba(var(--c-cyan),0.55), rgba(var(--c-violet),0.55), rgba(var(--c-amber),0.45)) border-box;
        background-size: 100% 100%, 260% 260%;
        animation: borderShift 10s ease-in-out infinite;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 30px 70px -25px rgba(0,0,0,0.65);
        margin: 0.75rem auto 1.25rem auto;
        max-width: 420px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }

    .login-card:before {
        content: "";
        position: absolute;
        inset: -2px;
        background: linear-gradient(90deg, rgba(var(--c-cyan),0.55), rgba(var(--c-violet),0.55), rgba(var(--c-amber),0.45));
        filter: blur(14px);
        opacity: 0.45;
        z-index: 0;
    }

    .login-card:after {
        content: "";
        position: absolute;
        top: -40%;
        left: -60%;
        width: 70%;
        height: 200%;
        background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.16), rgba(255,255,255,0.0));
        transform: rotate(12deg);
        opacity: 0.0;
        z-index: 1;
        animation: sheen 7.5s ease-in-out infinite;
        pointer-events: none;
    }

    .login-card > * {
        position: relative;
        z-index: 2;
    }
    
    .login-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .login-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .hero {
        border: 1px solid transparent;
        background:
            linear-gradient(145deg, rgba(30, 41, 59, 0.46) 0%, rgba(2, 6, 23, 0.72) 100%) padding-box,
            linear-gradient(90deg, rgba(var(--c-cyan),0.35), rgba(var(--c-violet),0.30), rgba(var(--c-amber),0.22)) border-box;
        background-size: 100% 100%, 260% 260%;
        animation: borderShift 14s ease-in-out infinite;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 1.75rem 1.75rem;
        box-shadow: 0 30px 70px -30px rgba(0,0,0,0.55);
        margin-bottom: 0.9rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .hero:after {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(420px 240px at 18% 20%, rgba(34, 211, 238, 0.22) 0%, rgba(0,0,0,0) 60%),
            radial-gradient(420px 240px at 82% 30%, rgba(168, 85, 247, 0.20) 0%, rgba(0,0,0,0) 60%);
        pointer-events: none;
    }

    .hero h1 {
        color: #f8fafc !important;
        margin: 0 0 0.35rem 0 !important;
        padding: 0 !important;
        font-size: 2.25rem !important;
        letter-spacing: -0.02em;
    }

    /* Gradient headline effect */
    .hero h1 {
        background: linear-gradient(90deg, rgba(var(--c-cyan), 1), rgba(var(--c-violet), 1), rgba(var(--c-amber), 0.95));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }

    .hero p {
        color: #cbd5e1;
        margin: 0.2rem 0 0.9rem 0;
        font-size: 1.05rem;
    }

    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.7rem;
        border-radius: 9999px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(148, 163, 184, 0.22);
        color: #f1f5f9;
        font-weight: 600;
        font-size: 0.85rem;
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }

    .chip:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(0,0,0,0.25);
        border-color: rgba(var(--c-cyan), 0.45);
    }
    
    /* Content cards */
    .stCard {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    
    /* Headers */
    h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        padding-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Keep headers readable inside light cards */
    .stCard h1, .stCard h2, .stCard h3 {
        color: #0f172a !important;
    }

    /* Widget labels */
    [data-testid="stWidgetLabel"] p,
    label,
    .stMarkdown, .stCaption {
        color: #e2e8f0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    .role-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.2rem 0.75rem;
        border-radius: 9999px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.5);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #e2e8f0;
    }

    .role-badge span.icon {
        font-size: 0.95rem;
    }

    .role-banner {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.6rem 0.85rem;
        border-radius: 9999px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.4);
        font-size: 0.85rem;
        color: #e2e8f0;
        margin-bottom: 0.6rem;
    }

    .role-banner span.icon {
        font-size: 1.1rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        transition: all 0.2s;
        border: 1px solid rgba(255,255,255,0.14);
        background: linear-gradient(90deg, rgba(var(--c-cyan),0.42), rgba(var(--c-violet),0.42));
        color: #f8fafc !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 26px rgba(0,0,0,0.35);
        border-color: rgba(255,255,255,0.22);
    }

    .stButton > button:active {
        transform: translateY(0px) scale(0.99);
    }

    /* Inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[role="combobox"] {
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        background: rgba(2, 6, 23, 0.35) !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: rgba(226, 232, 240, 0.55) !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stSelectbox"] div[role="combobox"]:focus-within {
        outline: none !important;
        border-color: rgba(34, 211, 238, 0.55) !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18) !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        font-family: 'Space Grotesk', 'DM Sans', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        color: rgba(226, 232, 240, 0.70);
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1.1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.22);
    }

    .footer strong {
        background: linear-gradient(90deg, rgba(var(--c-cyan), 1), rgba(var(--c-violet), 1), rgba(var(--c-amber), 0.95));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-excellent { background: #dcfce7; color: #166534; }
    .badge-good { background: #dbeafe; color: #1e40af; }
    .badge-fair { background: #fef3c7; color: #b45309; }
    .badge-poor { background: #fee2e2; color: #b91c1c; }
    
    /* Alert box styling */
    .stAlert {
        border-radius: 10px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 1px dashed #cbd5e1;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0 !important;
        border-color: #e2e8f0 !important;
    }

    /* Respect reduced motion */
    @media (prefers-reduced-motion: reduce) {
        .stApp:before,
        .login-card:after,
        .login-card,
        .hero {
            animation: none !important;
        }
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
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Candidate:N", sort=None, title="Candidate"),
            y=alt.Y("Match %:Q", title="Match %", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("Candidate:N"), alt.Tooltip("Match %:Q", format=".0f")],
        )
    )

    labels = (
        alt.Chart(chart_df)
        .mark_text(dy=-8, color="#E2E8F0", fontSize=12, fontWeight="bold")
        .encode(x=alt.X("Candidate:N", sort=None), y=alt.Y("Match %:Q"), text=alt.Text("Match %:Q", format=".0f"))
    )

    st.altair_chart((bars + labels).properties(title=title, height=280).interactive(), use_container_width=True)


# ==================== LOGIN SCREEN ====================
if not st.session_state.logged_in:
    st.markdown(
        """
        <div class="hero">
            <h1>Resume Screening</h1>
            <p>ATS-style matching, gap analysis, and learning suggestions — in one clean dashboard.</p>
            <div class="chips">
                <span class="chip">📈 Match dashboard</span>
                <span class="chip">🧩 Skill gaps</span>
                <span class="chip">🎓 Course suggestions</span>
                <span class="chip">📄 CSV exports</span>
                <span class="chip">🔐 Role-based login</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_sp1, c_mid, c_sp2 = st.columns([1, 1.25, 1])
    with c_mid:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.markdown("<p class='login-title'>📋 Resume Screening</p>", unsafe_allow_html=True)
        st.markdown("<p class='login-subtitle'>Sign in to continue</p>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            role = st.selectbox("Login as", ["Recruiter", "Job Seeker"], index=0)
            submitted = st.form_submit_button("Sign in", use_container_width=True)
    
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
                "admin / admin123\\nrecruiter / recruiter123\\njobseeker / jobseeker123"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Hire smarter. Apply smarter.")
    st.markdown(
        """A modern resume screening app with:
- **ATS-style match scoring**
- **Skill gap analysis**
- **AI course suggestions** (Job Seeker)
- **Recruiter dashboard + CSV exports**"""
    )
    st.markdown("<p class='footer'>MADE BY <strong>GILMAN</strong></p>", unsafe_allow_html=True)
    st.stop()

# ==================== SIDEBAR (logged in) ====================
with st.sidebar:
    st.markdown("### 📋 Resume Screening")

    role = st.session_state.user_role
    if role == "recruiter":
        role_label = "RECRUITER"
        role_icon = "👔"
    elif role == "job_seeker":
        role_label = "JOB SEEKER"
        role_icon = "📄"
    else:
        role_label = "ADMIN"
        role_icon = "🛡️"

    st.markdown(f"**{st.session_state.username}**")
    st.markdown(
        f"<span class='role-badge'><span class='icon'>{role_icon}</span>{role_label}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    if st.session_state.user_role == "admin":
        page = st.radio("Menu", ["Dashboard", "Activity log", "User info", "Settings"], label_visibility="collapsed")
    elif st.session_state.user_role == "recruiter":
        page = st.radio("Menu", ["Screening", "Dashboard", "Gap analysis"], label_visibility="collapsed")
    else:
        page = st.radio("Menu", ["Check match", "Gap analysis"], label_visibility="collapsed")
    
    st.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.shortlisted = []
        _rerun()

# ==================== ADMIN PANEL ====================
if st.session_state.user_role == "admin":
    st.title("Admin Panel")
    st.markdown(
        "<div class='role-banner'><span class='icon'>🛡️</span><span>Admin overview & activity log</span></div>",
        unsafe_allow_html=True,
    )
    
    if page == "Dashboard":
        st.markdown("#### Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Activity entries", len(st.session_state.admin_activity))
        with c2:
            st.metric("Roles", "3")
        with c3:
            st.metric("Status", "Active")
        st.markdown("---")
        st.info("Use **Activity log** in the sidebar to view recent sign-ins and screening actions.")
    
    elif page == "Activity log":
        st.markdown("#### Recent activity")
        if not st.session_state.admin_activity:
            st.caption("No activity yet.")
        else:
            log_df = pd.DataFrame(st.session_state.admin_activity)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
    
    elif page == "User info":
        st.markdown("#### Demo accounts")
        st.dataframe(pd.DataFrame([
            {"Username": "admin", "Role": "Admin", "Purpose": "Admin panel"},
            {"Username": "recruiter", "Role": "Recruiter", "Purpose": "Screen multiple resumes"},
            {"Username": "jobseeker", "Role": "Job Seeker", "Purpose": "Check resume vs JD"},
        ]), use_container_width=True, hide_index=True)
        st.caption("Passwords: admin123, recruiter123, jobseeker123")
    
    elif page == "Settings":
        st.markdown("#### Settings")
        st.caption("App configuration (placeholder). In production, add theme, limits, etc.")
    
    st.markdown("<p class='footer'>MADE BY <strong>GILMAN</strong></p>", unsafe_allow_html=True)
    st.stop()

# ==================== RECRUITER ====================
if st.session_state.user_role == "recruiter":
    if page == "Dashboard":
        st.title("Dashboard")
        st.markdown(
            "<div class='role-banner'><span class='icon'>👔</span><span>See candidate match scores at a glance</span></div>",
            unsafe_allow_html=True,
        )
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
            with st.expander("ℹ️ What do Match % and ATS mean?", expanded=False):
                st.markdown(
                    "- **Match %**: similarity between the job description and each resume, based on the text content.\n"
                    "- **ATS**: a simple label for how an Applicant Tracking System might view the match — Excellent, Good, Fair, or Poor."
                )
            st.dataframe(df, use_container_width=True, hide_index=True)
            simple_match_chart(df, title="Match % (sorted)")
            st.download_button(
                "Download dashboard (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resume_matching_dashboard.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Process resumes from **Screening** to see the dashboard.")
    
    elif page == "Gap analysis":
        st.title("Gap analysis")
        if st.session_state.last_results and st.session_state.last_jd_text:
            st.caption("Missing skills per candidate (from last run).")
            for r in st.session_state.last_results:
                res_text = r.get("resume_text")
                if not res_text:
                    continue
                missing = gap_analysis(st.session_state.last_jd_text, res_text, top_n=20)
                with st.expander(f"**{r['name']}** — {r['match']}% match"):
                    st.write(", ".join(missing) if missing else "No significant gaps.")
                    st.markdown("**AI course suggestions**")
                    recs = suggest_courses_from_gaps(missing, max_recs=6)
                    if recs:
                        for it in recs:
                            st.markdown(f"- [{it['title']}]({it['url']}) — {it['provider']} ({it['type']})")
                    else:
                        st.caption("No course suggestions needed (few/no gaps detected).")
        else:
            st.info("Run **Screening** first to see gap analysis.")
    
    else:  # Screening
        st.title("Screening")
        st.markdown(
            "<div class='role-banner'><span class='icon'>📂</span><span>Upload a JD and multiple resumes to compare</span></div>",
            unsafe_allow_html=True,
        )
        st.subheader("Upload JD and resumes")
        job_role = st.text_input("Job role (for dashboard)", value=st.session_state.last_job_role, placeholder="e.g. Data Analyst / Java Developer / ML Engineer")
        uploaded_jd = st.file_uploader("Job description (PDF)", type="pdf", key="jd")
        uploaded_resumes = st.file_uploader("Resume(s) (PDF)", type="pdf", accept_multiple_files=True, key="resumes")
        
        if uploaded_jd and uploaded_resumes:
            jd_text = extract_text_from_pdf(uploaded_jd)
            if not jd_text:
                st.warning("Could not read the job description PDF.")
            else:
                with st.expander("ℹ️ Tips for better screening", expanded=False):
                    st.markdown(
                        "- Use a **clear, detailed JD PDF** so the keywords are accurate.\n"
                        "- Upload **several resumes at once** to quickly see who is closest.\n"
                        "- Higher **Match %** and **Excellent/Good ATS** usually mean a stronger fit."
                    )
                if st.button("Process all resumes"):
                    results = []
                    progress = st.progress(0)
                    for idx, res_file in enumerate(uploaded_resumes):
                        progress.progress((idx + 1) / len(uploaded_resumes))
                        res_text = extract_text_from_pdf(res_file)
                        name = res_file.name
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
                st.markdown("#### Results (ATS compatibility)")
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
                simple_match_chart(df, title="Match % (sorted)")
                st.download_button(
                    "Download results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="screening_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                
                st.markdown("#### Shortlist")
                for r in st.session_state.last_results:
                    if r["match"] is not None and r["match"] >= 50:
                        if st.button(f"Add: {r['name']}", key=f"short_{r['name']}"):
                            if r["name"] not in st.session_state.shortlisted:
                                st.session_state.shortlisted.append(r["name"])
                                _rerun()
                if st.session_state.shortlisted:
                    st.write("**Shortlisted:** " + ", ".join(st.session_state.shortlisted))
                    shortlist_df = pd.DataFrame([{"Candidate": n} for n in st.session_state.shortlisted])
                    st.download_button(
                        "Download shortlist (CSV)",
                        data=shortlist_df.to_csv(index=False).encode("utf-8"),
                        file_name="shortlist.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    if st.button("Clear shortlist"):
                        st.session_state.shortlisted = []
                        _rerun()
    
    st.markdown("<p class='footer'>MADE BY <strong>GILMAN</strong></p>", unsafe_allow_html=True)
    st.stop()

# ==================== JOB SEEKER ====================
if st.session_state.user_role == "job_seeker" and page == "Gap analysis":
    st.title("Gap analysis")
    st.markdown(
        "<div class='role-banner'><span class='icon'>📄</span><span>See which JD skills are missing on your resume</span></div>",
        unsafe_allow_html=True,
    )
    st.subheader("Missing skills in your resume")
    if st.session_state.js_last_jd and st.session_state.js_last_resume:
        missing = gap_analysis(st.session_state.js_last_jd, st.session_state.js_last_resume, top_n=25)
        st.caption("Keywords from the job description that are missing or weak in your resume.")
        if missing:
            st.write(", ".join(missing))
        else:
            st.info("No significant gaps — your resume covers the JD keywords well.")
    else:
        st.info("Go to **Check match**, upload JD and resume, then click Process to see gap analysis here.")

else:
    st.title("Check match")
    st.markdown(
        "<div class='role-banner'><span class='icon'>🎯</span><span>Upload a JD and your resume to get a match score</span></div>",
        unsafe_allow_html=True,
    )
    st.subheader("See how your resume fits the job")
    with st.expander("ℹ️ How this match works", expanded=False):
        st.markdown(
            "- We read the **text** from your JD PDF and resume PDF.\n"
            "- We compare them and compute a **Match %** (0–100).\n"
            "- We also show an **ATS label** (Excellent / Good / Fair / Poor) as a simple interpretation."
        )
    uploaded_jd = st.file_uploader("Job description (PDF)", type="pdf", key="jd_js")
    uploaded_resume = st.file_uploader("Your resume (PDF)", type="pdf", key="res_js")
    click = st.button("Process")

    job_description = extract_text_from_pdf(uploaded_jd) if uploaded_jd else None
    resume = extract_text_from_pdf(uploaded_resume) if uploaded_resume else None

    if click:
        if not job_description or not resume:
            st.warning("Upload both job description and resume.")
        else:
            st.session_state.js_last_jd = job_description
            st.session_state.js_last_resume = resume
            match = get_match(job_description, resume)
            label, _ = match_label(match)
            ats = ats_compatibility_label(match)
            log_activity("Job Seeker", st.session_state.username, "Match check", f"{match}%")

            st.metric("Match", f"{match}%")
            st.markdown(f"**{label}** · ATS: **{ats}**")
            if match >= 70:
                st.success("Strong fit. Consider applying.")
            elif match >= 50:
                st.info("Moderate fit. Highlight relevant skills.")
            else:
                st.warning("Improve resume with missing keywords (see Gap analysis in sidebar).")

            st.markdown("#### Remaining skills (missing from your resume)")
            missing = gap_analysis(job_description, resume, top_n=25)
            st.session_state.js_last_missing = missing
            st.session_state.js_show_ai_suggestions = False
            if missing:
                st.write(", ".join(missing[:12]))
                st.caption("Showing top missing skills. Use Ask AI to get course suggestions.")
            else:
                st.info("No significant gaps — your resume covers the JD keywords well.")
                st.caption("Ask AI will have no suggestions if there are no missing skills.")

    # Show Ask AI section whenever we have a processed result
    if st.session_state.js_last_missing is not None:
        st.markdown("---")
        st.markdown("### Ask AI to match this job profile")
        st.caption("Click the button to get course suggestions based on your remaining (missing) skills.")

        if st.button("Ask AI", key="ask_ai_courses"):
            st.session_state.js_show_ai_suggestions = True
            log_activity(
                "Job Seeker",
                st.session_state.username,
                "AI course suggestions",
                f"{len(st.session_state.js_last_missing or [])} gaps",
            )

        if st.session_state.js_show_ai_suggestions:
            missing_now = st.session_state.js_last_missing or []
            st.markdown("#### Recommended courses/resources")

            curated = suggest_courses_from_gaps(missing_now, max_recs=8)
            google_links = suggest_course_search_links(missing_now, max_skills=8)

            if curated:
                st.markdown("**Curated picks**")
                for it in curated:
                    st.markdown(f"- [{it['title']}]({it['url']}) — {it['provider']} ({it['type']})")

            if google_links:
                st.markdown("**Google course searches (per missing skill)**")
                for l in google_links:
                    st.markdown(f"- [{l['label']}]({l['url']})")

            if not curated and not google_links:
                st.info("No course suggestions needed — your resume already covers the JD keywords well.")

st.markdown("<p class='footer'>MADE BY <strong>GILMAN</strong></p>", unsafe_allow_html=True)
