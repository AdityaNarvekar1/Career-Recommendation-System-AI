# streamlit_app/app.py — Main Entry Point
import streamlit as st
st.set_page_config(
    page_title="CareerAI — O*NET Powered Career Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import streamlit.components.v1 as components

from config import *
from src.engine.recommender import HybridCareerRecommender
from src.utils.feature_engineering import vectorize_user_profile

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background: #0a0e1a; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1529 50%, #0a1020 100%);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #7b61ff, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}

.hero-sub {
    text-align: center;
    color: #8892a4;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #1a2035, #1e2540);
    border: 1px solid #2a3550;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s;
}

.metric-card:hover { transform: translateY(-3px); }

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    color: #8892a4;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

.career-card {
    background: linear-gradient(135deg, #141929, #1a2240);
    border: 1px solid #2a3550;
    border-left: 4px solid #00d4ff;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.2s;
}

.career-card:hover {
    border-left-color: #7b61ff;
    transform: translateX(4px);
}

.career-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e8eaf6;
    margin-bottom: 0.3rem;
}

.career-score {
    display: inline-block;
    background: linear-gradient(135deg, #00d4ff22, #7b61ff22);
    border: 1px solid #00d4ff44;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    color: #00d4ff;
    font-weight: 600;
}

.riasec-badge {
    display: inline-block;
    background: #7b61ff22;
    border: 1px solid #7b61ff44;
    border-radius: 12px;
    padding: 0.1rem 0.6rem;
    font-size: 0.78rem;
    color: #a78bfa;
    margin: 0.1rem;
}

.gap-tag {
    display: inline-block;
    background: #ff6b9d22;
    border: 1px solid #ff6b9d44;
    border-radius: 10px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    color: #ff9ebe;
    margin: 0.1rem;
}

.section-header {
    color: #00d4ff;
    font-size: 1.4rem;
    font-weight: 600;
    border-bottom: 1px solid #2a3550;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.transition-card {
    background: linear-gradient(135deg, #141929, #1a1f35);
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 1.2rem;
}

.feasibility-easy     { color: #2ecc71; font-weight: 600; }
.feasibility-moderate { color: #f39c12; font-weight: 600; }
.feasibility-hard     { color: #e74c3c; font-weight: 600; }

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1529, #0a0e1a);
    border-right: 1px solid #1e2a45;
}

.stSlider > div > div > div { background: #00d4ff !important; }

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7b61ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.85; }

.info-box {
    background: #1a2035;
    border: 1px solid #00d4ff33;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #aab4c8;
    font-size: 0.9rem;
}

/* Search result buttons — look like a dropdown list, not buttons */
div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: transparent !important;
    color: #c8d0e0 !important;
    border: none !important;
    border-bottom: 1px solid #1e2a45 !important;
    border-radius: 0 !important;
    padding: 8px 14px !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    width: 100% !important;
    margin: 0 !important;
    transition: background 0.1s !important;
}
div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #1e2a45 !important;
    color: #00d4ff !important;
    opacity: 1 !important;
}

/* Clear and run buttons keep their normal style */
div[data-testid="stSidebar"] div[data-testid="stButton"]:last-child > button {
    background: linear-gradient(135deg, #00d4ff, #7b61ff) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Models (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_system():
    try:
        X           = joblib.load(MASTER_FEATURES_FILE)
        y           = joblib.load(MASTER_LABELS_FILE)
        occ_meta    = joblib.load(OCCUPATION_META_FILE)
        scaler      = joblib.load(SCALER_FILE)
        best_model  = joblib.load(BEST_MODEL_FILE)
        km_model    = joblib.load(KMEANS_MODEL_FILE)
        leaderboard = joblib.load(MODELS_DIR / "leaderboard.pkl")

        if hasattr(scaler, "feature_names_in_"):
            del scaler.feature_names_in_
        joblib.dump(scaler, SCALER_FILE)

        cluster_cols = [c for c in X.columns if c.startswith("wa__") or c.startswith("skill__")]
        cluster_assignments = pd.Series(km_model.labels_, index=X.index, name="cluster")

        from src.models.clustering import label_clusters
        cluster_labels_dict, _ = label_clusters(km_model, X[cluster_cols].values,
                                                  X.index, y, occ_meta)

        recommender = HybridCareerRecommender(
            best_model=best_model,
            job_vectors=X,
            riasec_labels=y,
            occupation_meta=occ_meta,
            cluster_labels=cluster_labels_dict,
            cluster_assignments=cluster_assignments,
            feature_columns=list(X.columns),
            scaler=scaler,
        )
        return recommender, occ_meta, X, y, scaler, leaderboard, True
    except Exception as e:
        st.error(f"Load error: {e}")
        return None, None, None, None, None, None, False


# ─── Build self-contained modal HTML ─────────────────────────────────────────
def build_modal_html(mr, occ_meta, riasec_labels):
    """
    Returns a complete HTML page rendered via components.v1.html().
    The iframe it lives in is sized to cover the full viewport so
    position:fixed works correctly — no dependency on st.dialog.
    Closing sends a postMessage to the parent; the parent page listens
    via a small inline <script> and clicks a hidden Streamlit button
    to trigger st.rerun() and clear modal_soc.
    """
    msoc      = mr["soc_code"]
    full_desc = str(occ_meta.loc[msoc, "description"] or "No description available.") \
                if msoc in occ_meta.index else "No description available."
    full_desc = full_desc.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    zone_desc = {
        1: "Little preparation", 2: "Some preparation",
        3: "Medium preparation", 4: "Considerable preparation",
        5: "Extensive preparation",
    }.get(mr["job_zone"], "")

    color_map = {
        "Realistic": "#e74c3c", "Investigative": "#3498db",
        "Artistic": "#2ecc71",  "Social": "#f39c12",
        "Enterprising": "#9b59b6", "Conventional": "#1abc9c",
    }

    riasec_rows = ""
    if msoc in riasec_labels.index:
        for code in RIASEC_CODES:
            jval = float(riasec_labels.loc[msoc, code])
            pct  = jval * 100
            c    = color_map.get(code, "#00d4ff")
            bar_w = int(pct * 1.5)
            riasec_rows += (
                f'<div style="margin:6px 0;font-size:0.83rem;color:#c8d0e0;">'
                f'<span style="display:inline-block;width:110px;">{code}</span>'
                f'<span style="display:inline-block;background:{c};width:{bar_w}px;'
                f'height:10px;border-radius:3px;vertical-align:middle;"></span>'
                f'<span style="color:#aab4c8;margin-left:6px;">{pct:.1f}%</span>'
                f'</div>'
            )

    gaps = mr["skill_gaps"]
    gaps_rows = ""
    if gaps:
        for sk, gv in gaps[:8]:
            sev = "#e74c3c" if gv > 0.4 else "#f39c12" if gv > 0.2 else "#2ecc71"
            gaps_rows += (
                f'<div style="font-size:0.83rem;color:#c8d0e0;margin:5px 0;">'
                f'<span style="color:{sev};margin-right:6px;">&#9679;</span>'
                f'{sk} <span style="color:#8892a4;font-size:0.78rem;">(gap: {gv:.2f})</span>'
                f'</div>'
            )
    else:
        gaps_rows = '<div style="color:#2ecc71;">&#10003; No significant skill gaps</div>'

    explanation = (mr["explanation"]
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{
  font-family:'Segoe UI',sans-serif;
  background:rgba(5,8,20,0.85);
  overflow:hidden;
}}
.backdrop{{
  position:fixed;inset:0;
  display:flex;align-items:center;justify-content:center;
  z-index:9999;
}}
.modal{{
  background:linear-gradient(135deg,#141929,#1a2240);
  border:1px solid #2a3550;
  border-radius:20px;
  padding:2rem 2.2rem;
  width:min(720px,92vw);
  max-height:88vh;
  overflow-y:auto;
  box-shadow:0 24px 80px rgba(0,0,0,0.8);
  position:relative;
}}
.modal::-webkit-scrollbar{{width:5px;}}
.modal::-webkit-scrollbar-track{{background:#0d1422;}}
.modal::-webkit-scrollbar-thumb{{background:#2a3550;border-radius:3px;}}
.close-btn{{
  position:absolute;top:1rem;right:1.1rem;
  background:#1e2a40;border:1px solid #2a3550;
  color:#aab4c8;font-size:1rem;border-radius:8px;
  padding:4px 12px;cursor:pointer;transition:background 0.15s;
}}
.close-btn:hover{{background:#2a3550;color:#fff;}}
.sec{{color:#7b61ff;font-weight:600;font-size:0.8rem;
       text-transform:uppercase;letter-spacing:1px;margin-bottom:0.45rem;}}
.inset{{background:#0d1422;border-radius:10px;padding:0.85rem 1rem;margin-bottom:1.1rem;}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;margin-bottom:1.1rem;}}
@media(max-width:520px){{.two{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>
<div class="backdrop" id="bd">
  <div class="modal">
    <button class="close-btn" onclick="close_modal()">&#10005; Close</button>

    <div style="margin-bottom:1.1rem;padding-right:3.5rem;">
      <div style="color:#00d4ff;font-size:1.3rem;font-weight:700;
                  letter-spacing:-0.3px;margin-bottom:4px;">{mr["title"]}</div>
      <div style="color:#8892a4;font-size:0.82rem;">
        &#128193; {mr["career_domain"]} &nbsp;|&nbsp;
        &#127891; Job Zone {mr["job_zone"]}/5 &mdash; {zone_desc} &nbsp;|&nbsp;
        SOC: {mr["soc_code"]}
      </div>
    </div>
    <div style="position:absolute;top:1.5rem;right:4.2rem;
                background:linear-gradient(135deg,#00d4ff22,#7b61ff22);
                border:1px solid #00d4ff44;border-radius:20px;
                padding:5px 14px;color:#00d4ff;font-weight:700;font-size:1rem;">
      {mr["match_score"]}% match
    </div>

    <div class="sec">&#128203; Job Description</div>
    <div class="inset" style="color:#c8d0e0;font-size:0.87rem;line-height:1.7;">
      {full_desc}
    </div>

    <div class="two">
      <div>
        <div class="sec">&#127919; Job RIASEC Profile</div>
        <div class="inset">{riasec_rows}</div>
      </div>
      <div>
        <div class="sec">&#128295; Your Skill Gaps</div>
        <div class="inset">{gaps_rows}</div>
      </div>
    </div>

    <div style="border-left:3px solid #7b61ff;padding:0.7rem 1rem;
                background:#1e2a40;border-radius:0 8px 8px 0;
                color:#aab4c8;font-size:0.84rem;white-space:pre-line;">
      <div class="sec" style="margin-bottom:0.4rem;">&#128161; Why Recommended</div>
      {explanation}
    </div>
  </div>
</div>
<script>
function close_modal(){{
  window.parent.postMessage({{type:"career_modal_close"}},"*");
}}
document.getElementById("bd").addEventListener("click",function(e){{
  if(e.target===this) close_modal();
}});
</script>
</body>
</html>"""


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">&#127919; CareerAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI-Powered Career Recommendation System · Powered by O*NET Database</p>',
            unsafe_allow_html=True)

# ─── Load System ─────────────────────────────────────────────────────────────
recommender, occ_meta, X, riasec_labels, scaler, leaderboard, system_ready = load_system()

if not system_ready:
    st.error("⚠️ Models not found. Please run the Jupyter Notebook first to train and save models.")
    st.info("📓 Open `notebooks/career_recommendation_system.ipynb` and run all cells.")
    st.stop()

# ─── Sidebar: User Profile Builder ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Build Your Profile")
    st.markdown("---")

    st.markdown("### 🔧 Your Skills (1=Beginner → 5=Expert)")
    user_skills = {}
    skill_groups = {
        "Analytical":  ["Mathematics", "Critical Thinking", "Data Analysis", "Research"],
        "Technical":   ["Programming", "Engineering", "Science", "Technology"],
        "People":      ["Communication", "Teaching", "Leadership", "Teamwork"],
        "Creative":    ["Writing", "Design", "Arts", "Innovation"],
        "Business":    ["Management", "Sales", "Finance", "Planning"],
    }

    for group, skills in skill_groups.items():
        with st.expander(f"**{group} Skills**", expanded=(group == "Analytical")):
            for skill in skills:
                val = st.slider(skill, 1, 5, 3, key=f"skill_{skill}")
                user_skills[skill] = val

    st.markdown("---")
    st.markdown("### 🎯 Your Interests")
    st.caption("Set higher for types you enjoy — leave at 1 if not interested")
    user_interests = {}
    riasec_emojis = {
        "Realistic":     "🔧",
        "Investigative": "🔬",
        "Artistic":      "🎨",
        "Social":        "🤝",
        "Enterprising":  "💼",
        "Conventional":  "📊",
    }
    riasec_hints = {
        "Realistic":     "Hands-on, mechanical, outdoors",
        "Investigative": "Research, analysis, science",
        "Artistic":      "Creative, design, expression",
        "Social":        "Helping, teaching, counseling",
        "Enterprising":  "Leadership, business, persuasion",
        "Conventional":  "Data, organizing, administration",
    }
    for code in RIASEC_CODES:
        val = st.slider(
            f"{riasec_emojis[code]} {code}",
            1, 5, 1,
            help=riasec_hints[code],
            key=f"interest_{code}"
        )
        user_interests[code] = val

    st.markdown("---")
    top_n = st.selectbox("🏆 Number of Recommendations", [5, 8, 10, 15], index=2)

    st.markdown("---")
    st.markdown("### 🛤️ Career Transition (Optional)")

    if "selected_career" not in st.session_state:
        st.session_state.selected_career = None
    if "career_search_query" not in st.session_state:
        st.session_state.career_search_query = ""

    career_search = st.text_input(
        "Search your current career",
        placeholder="Type to search e.g. nurse, web developer...",
        key="career_search",
    )

    if career_search != st.session_state.career_search_query:
        st.session_state.career_search_query = career_search
        st.session_state.selected_career     = None

    current_soc = None

    if career_search and len(career_search.strip()) >= 2:
        q       = career_search.strip().lower()
        q_words = q.split()

        def fuzzy_score(title: str) -> float:
            t = title.lower()
            t_words = t.split()
            if q == t:                                       return 1.00
            if t.startswith(q):                              return 0.95
            if q in t:                                       return 0.85
            hits_exact = sum(1 for qw in q_words if qw in t_words)
            if hits_exact == len(q_words):                   return 0.80
            hits_sub = sum(1 for qw in q_words
                           if any(qw in tw for tw in t_words))
            if hits_sub > 0: return 0.55 + 0.20*(hits_sub/len(q_words))
            hits_start = sum(1 for qw in q_words
                             if any(tw.startswith(qw) for tw in t_words))
            if hits_start > 0: return 0.35 + 0.10*(hits_start/len(q_words))
            q_bi = {q[i:i+2] for i in range(len(q)-1)}
            t_bi = {title.lower()[i:i+2] for i in range(len(title)-3)}
            if q_bi and t_bi:
                ov = len(q_bi & t_bi) / len(q_bi)
                if ov > 0.5: return 0.20 + 0.10*ov
            return 0.0

        matches = [t for t, s in sorted(
            [(t, fuzzy_score(t)) for t in occ_meta["title"].tolist()],
            key=lambda x: -x[1]
        ) if s > 0.0][:8]

        if st.session_state.selected_career is None:
            if matches:
                st.markdown(
                    "<div style='background:#141929;border:1px solid #2a3550;"
                    "border-radius:10px;overflow:hidden;margin-top:4px;'>",
                    unsafe_allow_html=True
                )
                for t in matches:
                    if st.button(f"🔍  {t}", key=f"srch_{t[:40]}",
                                 use_container_width=True):
                        st.session_state.selected_career = t
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption("No matches — try shorter keywords like 'web' or 'nurse'")

    if st.session_state.selected_career:
        sel = st.session_state.selected_career
        current_soc = occ_meta[occ_meta["title"] == sel].index[0]
        st.success(f"✅ **{sel}**")
        if st.button("✕ Clear", key="clear_career"):
            st.session_state.selected_career = None
            st.rerun()

    st.markdown("---")
    run_btn = st.button("🚀 Find My Careers", use_container_width=True)


# ─── Session state init ───────────────────────────────────────────────────────
if "modal_soc"    not in st.session_state: st.session_state.modal_soc    = None
if "last_results" not in st.session_state: st.session_state.last_results = None
if "user_vector"  not in st.session_state: st.session_state.user_vector  = None

# ─── Modal: rendered at the very top via components.v1.html ──────────────────
# components.html() creates its own <iframe> where position:fixed IS relative
# to the viewport. We set its height to 100vh equivalent (window.screen.height)
# and make its background cover everything, creating a true overlay effect.
if st.session_state.modal_soc is not None and st.session_state.last_results is not None:
    _recs = st.session_state.last_results["recommendations"]
    _mr   = next((r for r in _recs if r["soc_code"] == st.session_state.modal_soc), None)
    if _mr:
        # Inject a postMessage listener into the parent window that clicks
        # a hidden button when the modal sends "career_modal_close".
        # This hidden button is a real Streamlit button that clears modal_soc
        # and triggers st.rerun().
        st.markdown("""
        <script>
        (function(){
          if(window.__careerModalListenerActive) return;
          window.__careerModalListenerActive = true;
          window.addEventListener("message", function(e){
            if(e.data && e.data.type === "career_modal_close"){
              var btn = document.querySelector('[data-testid="baseButton-secondary"]'
                + '[kind="secondary"]#career_modal_close_btn');
              // fallback: find by aria-label
              var btns = document.querySelectorAll("button");
              for(var i=0;i<btns.length;i++){
                if(btns[i].innerText.trim() === "__CLOSE_MODAL__"){
                  btns[i].click(); break;
                }
              }
            }
          });
        })();
        </script>
        """, unsafe_allow_html=True)

        # Render the full-page modal iframe
        # Height covers most screens; the modal card itself is scrollable
        modal_html = build_modal_html(_mr, occ_meta, riasec_labels)
        components.html(modal_html, height=900, scrolling=False)

        # Hidden close trigger — its label is matched by the JS above
        col_close, _ = st.columns([1, 4])
        with col_close:
            if st.button("✕ Close Details", key="career_modal_close_btn"):
                st.session_state.modal_soc = None
                st.rerun()


# ─── Main Content ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("🤖 Running AI Recommendation Engine..."):
        st.session_state.user_vector, user_riasec_norm = vectorize_user_profile(
            user_skills=user_skills,
            user_interests=user_interests,
            all_feature_columns=list(X.columns),
            scaler=scaler,
            riasec_codes=RIASEC_CODES
        )

        results = recommender.recommend(
            user_vector=st.session_state.user_vector,
            user_riasec_input=user_interests,
            top_n=top_n,
            current_career_soc=current_soc
        )
    st.session_state.last_results = results
    st.session_state.modal_soc    = None

if st.session_state.last_results is not None:
    results      = st.session_state.last_results
    recs         = results["recommendations"]
    riasec_dist  = results["riasec_profile"]
    dom_types    = results["dominant_types"]

    # ── Metric Row ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{recs[0]['match_score']}%</div>
            <div class="metric-label">Top Match Score</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(recs)}</div>
            <div class="metric-label">Careers Found</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{dom_types[0][:4]}</div>
            <div class="metric-label">Primary Type</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{results['total_candidates']}</div>
            <div class="metric-label">Jobs Analyzed</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Recommendations", "📊 RIASEC Profile",
        "🔍 Skill Gaps", "🛤️ Career Path", "📈 Model Report", "🧪 Sanity Check"
    ])

    # ── Tab 1: Recommendations ─────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Top Career Recommendations</div>',
                    unsafe_allow_html=True)
        st.caption("Click **View Details** on any card to open full job details")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            for r in recs:
                gaps_html = "".join([
                    f'<span class="gap-tag">↑ {g[0]}</span>'
                    for g in r["skill_gaps"][:3]
                ])
                riasec_top  = sorted(r["riasec_dist"].items(), key=lambda x: -x[1])[:2]
                riasec_html = " ".join([
                    f'<span class="riasec-badge">{k} {v*100:.0f}%</span>'
                    for k, v in riasec_top
                ])

                st.markdown(f"""
                <div class="career-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="career-title">#{r['rank']} {r['title']}</div>
                        <span class="career-score">{r['match_score']}%</span>
                    </div>
                    <div style="margin:0.3rem 0;color:#8892a4;font-size:0.83rem;">
                        📁 {r['career_domain']} &nbsp;|&nbsp; 🎓 Job Zone {r['job_zone']}/5
                    </div>
                    <div style="margin-top:0.4rem;">
                        {riasec_html}
                    </div>
                    <div style="margin-top:0.4rem;">
                        {'<span style="color:#8892a4;font-size:0.78rem;">Gaps: </span>' + gaps_html if gaps_html else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("View Details", key=f"modal_btn_{r['soc_code']}"):
                    st.session_state.modal_soc = r["soc_code"]
                    st.rerun()

        with col_right:
            fig = go.Figure(go.Bar(
                x=[r["match_score"] for r in recs],
                y=[r["title"][:30] for r in recs],
                orientation="h",
                marker=dict(
                    color=[r["match_score"] for r in recs],
                    colorscale=[[0, "#1a2240"], [0.5, "#7b61ff"], [1.0, "#00d4ff"]],
                    showscale=False
                ),
                text=[f"{r['match_score']}%" for r in recs],
                textposition="outside",
            ))
            fig.update_layout(
                title="Match Score Ranking",
                paper_bgcolor="#141929", plot_bgcolor="#141929",
                font=dict(color="#aab4c8"),
                height=420, margin=dict(l=10, r=50, t=40, b=10),
                xaxis=dict(range=[0, 108], showgrid=True,
                           gridcolor="#2a3550", title="Match Score (%)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: RIASEC Profile ───────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Your RIASEC Profile</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            r_values = [riasec_dist[c] * 100 for c in RIASEC_CODES]
            fig_radar = go.Figure(go.Scatterpolar(
                r=r_values + [r_values[0]],
                theta=RIASEC_CODES + [RIASEC_CODES[0]],
                fill="toself",
                fillcolor="rgba(123,97,255,0.25)",
                line=dict(color="#7b61ff", width=2),
                name="Your Profile"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#141929",
                    radialaxis=dict(visible=True, range=[0, 50], gridcolor="#2a3550",
                                    tickcolor="#8892a4", tickfont=dict(color="#8892a4")),
                    angularaxis=dict(gridcolor="#2a3550", tickfont=dict(color="#e8eaf6", size=13))
                ),
                paper_bgcolor="#141929",
                font=dict(color="#aab4c8"),
                showlegend=False,
                title=dict(text="RIASEC Radar", font=dict(color="#e8eaf6")),
                height=380,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_b:
            fig_bar = go.Figure(go.Bar(
                x=RIASEC_CODES,
                y=[riasec_dist[c] * 100 for c in RIASEC_CODES],
                marker_color=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"],
                text=[f"{riasec_dist[c]*100:.1f}%" for c in RIASEC_CODES],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="RIASEC Score Breakdown",
                paper_bgcolor="#141929", plot_bgcolor="#141929",
                font=dict(color="#aab4c8"), height=380,
                yaxis=dict(range=[0, 55], showgrid=True, gridcolor="#2a3550"),
                xaxis=dict(tickfont=dict(size=12)),
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        dom = sorted(riasec_dist, key=riasec_dist.get, reverse=True)[:2]
        st.markdown(f"""
        <div class="info-box">
        🧠 <strong>Profile Interpretation:</strong><br>
        Your dominant type is <strong style="color:#00d4ff">{dom[0]}</strong>
        ({riasec_dist[dom[0]]*100:.1f}%) — you excel in <em>{RIASEC_DESCRIPTIONS[dom[0]]}</em>.<br>
        Secondary type is <strong style="color:#7b61ff">{dom[1]}</strong>
        ({riasec_dist[dom[1]]*100:.1f}%) — strong in <em>{RIASEC_DESCRIPTIONS[dom[1]]}</em>.<br><br>
        🎯 Best fit career domain: <strong style="color:#ff9ebe">{RIASEC_CAREER_DOMAINS[dom[0]]}</strong>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 How was this RIASEC profile calculated? (click to see)"):
            st.markdown("Your RIASEC profile is computed as a **weighted blend**:")
            st.markdown("- **70%** from your **Interest sliders** (what you told us you enjoy)")
            st.markdown("- **30%** from the **ML model** (what your skills suggest)")
            st.markdown("---")

            total_int = sum(user_interests.values())
            st.markdown("**Your raw interest slider inputs (normalized):**")
            for code in RIASEC_CODES:
                raw  = user_interests[code]
                norm = raw / total_int * 100
                st.markdown(f"  - {riasec_emojis[code]} {code}: slider={raw}/5 → {norm:.1f}% weight")

            st.markdown("---")
            st.markdown("**ML model prediction (from your skill vector):**")
            model_pred = recommender.predict_riasec(st.session_state.user_vector)
            for code in RIASEC_CODES:
                st.markdown(f"  - {code}: {model_pred[code]*100:.1f}%")

            st.markdown("---")
            st.markdown("**Final blended RIASEC (70% interests + 30% model):**")
            for code in RIASEC_CODES:
                st.markdown(f"  - {code}: {riasec_dist[code]*100:.1f}%")

    # ── Tab 3: Skill Gaps ──────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Skill Gap Intelligence</div>', unsafe_allow_html=True)
        st.markdown("Skills you need to develop to qualify for each career:")

        for r in recs[:5]:
            if not r["skill_gaps"]:
                continue
            with st.expander(f"#{r['rank']} {r['title']} — {r['match_score']}% match"):
                gaps    = r["skill_gaps"]
                skills_g = [g[0] for g in gaps]
                scores_g = [g[1] for g in gaps]
                fig_gap = go.Figure(go.Bar(
                    x=scores_g[::-1], y=skills_g[::-1],
                    orientation="h",
                    marker_color="#ff6b9d",
                    text=[f"{v:.3f}" for v in scores_g[::-1]],
                    textposition="outside",
                ))
                fig_gap.update_layout(
                    paper_bgcolor="#141929", plot_bgcolor="#141929",
                    font=dict(color="#aab4c8"), height=280,
                    margin=dict(l=10, r=60, t=10, b=10),
                    xaxis=dict(showgrid=True, gridcolor="#2a3550"),
                )
                st.plotly_chart(fig_gap, use_container_width=True)

    # ── Tab 4: Career Transition Path ─────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Career Transition Path</div>', unsafe_allow_html=True)

        if current_soc and recs[0].get("transition"):
            trans = recs[0]["transition"]
            feasibility_class = (
                "feasibility-easy" if "Easy" in trans["feasibility"]
                else "feasibility-moderate" if "Moderate" in trans["feasibility"]
                else "feasibility-hard"
            )

            st.markdown(f"""
            <div class="transition-card">
                <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                    <div style="background:#1e2a45; border-radius:8px; padding:0.7rem 1rem;">
                        <div style="color:#8892a4;font-size:0.8rem;">FROM</div>
                        <div style="color:#e8eaf6;font-weight:600;">{trans['current']}</div>
                    </div>
                    <div style="font-size:1.5rem;">→</div>
                    <div style="background:#1e2a45; border-radius:8px; padding:0.7rem 1rem; border: 1px solid #00d4ff44;">
                        <div style="color:#8892a4;font-size:0.8rem;">TO</div>
                        <div style="color:#00d4ff;font-weight:600;">{trans['target']}</div>
                    </div>
                    <div style="margin-left:auto;">
                        <span class="{feasibility_class}">{trans['feasibility']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("#### 📚 Skills to Learn")
                for skill in trans["skills_to_learn"]:
                    st.markdown(f"• {skill}")

            with col_t2:
                if trans["intermediate_roles"]:
                    st.markdown("#### 🪜 Stepping-Stone Roles")
                    for i, role in enumerate(trans["intermediate_roles"]):
                        st.markdown(f"{i+1}. {role}")
                st.markdown(f"#### 📈 Job Zone Change: `{trans['job_zone_change']:+d}` levels")
        else:
            st.info("👈 Select your current career in the sidebar to see your transition path.")

    # ── Tab 5: Model Report ────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Model Performance Report</div>', unsafe_allow_html=True)

        st.dataframe(
            leaderboard.style
            .background_gradient(subset=["R² Score", "Cosine Sim"], cmap="YlGn")
            .background_gradient(subset=["RMSE"], cmap="YlOrRd_r")
            .format({"R² Score": "{:.4f}", "RMSE": "{:.4f}", "Cosine Sim": "{:.4f}"}),
            use_container_width=True
        )

        best_model_name = leaderboard.iloc[0]["Model"]
        st.success(f"🥇 **Best Model: {best_model_name}** — R²={leaderboard.iloc[0]['R² Score']:.4f} | "
                   f"RMSE={leaderboard.iloc[0]['RMSE']:.4f} | CosSim={leaderboard.iloc[0]['Cosine Sim']:.4f}")

        fig_lb = go.Figure()
        fig_lb.add_trace(go.Bar(
            name="R² Score", x=leaderboard["Model"], y=leaderboard["R² Score"],
            marker_color="#00d4ff", opacity=0.85
        ))
        fig_lb.add_trace(go.Bar(
            name="Cosine Sim", x=leaderboard["Model"], y=leaderboard["Cosine Sim"],
            marker_color="#7b61ff", opacity=0.85
        ))
        fig_lb.update_layout(
            barmode="group",
            title="All Models: R² vs Cosine Similarity",
            paper_bgcolor="#141929", plot_bgcolor="#141929",
            font=dict(color="#aab4c8"), height=400,
            xaxis=dict(tickangle=-30, gridcolor="#2a3550"),
            yaxis=dict(showgrid=True, gridcolor="#2a3550"),
            legend=dict(bgcolor="#1a2035"),
        )
        st.plotly_chart(fig_lb, use_container_width=True)

    # ── Tab 6: Sanity Check ────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-header">🧪 Sanity Check — Is the Model Working?</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        This tab runs **3 controlled tests** to verify the model gives different,
        logically correct results for different interest profiles.
        If the model is working correctly, each test should return clearly
        different top careers matching the stated interest.
        """)

        test_cases = [
            {
                "label":    "Test 1: Pure Investigative (Scientist/Researcher)",
                "interests": {"Realistic":1,"Investigative":5,"Artistic":1,"Social":1,"Enterprising":1,"Conventional":1},
                "expected_keywords": ["research","scientist","analyst","biolog","chemi","physics","data","laborator"],
                "color": "#3498db"
            },
            {
                "label":    "Test 2: Pure Social (Counselor/Teacher/Nurse)",
                "interests": {"Realistic":1,"Investigative":1,"Artistic":1,"Social":5,"Enterprising":1,"Conventional":1},
                "expected_keywords": ["counsel","teach","social","nurs","therapist","care","health","educat"],
                "color": "#2ecc71"
            },
            {
                "label":    "Test 3: Pure Realistic (Engineer/Mechanic/Technician)",
                "interests": {"Realistic":5,"Investigative":1,"Artistic":1,"Social":1,"Enterprising":1,"Conventional":1},
                "expected_keywords": ["engineer","mechanic","technician","construct","electri","install","repair","operat"],
                "color": "#e74c3c"
            },
        ]

        neutral_skills = {s: 3 for group in [
            ["Mathematics","Critical Thinking","Data Analysis","Research"],
            ["Programming","Engineering","Science","Technology"],
            ["Communication","Teaching","Leadership","Teamwork"],
            ["Writing","Design","Arts","Innovation"],
            ["Management","Sales","Finance","Planning"],
        ] for s in group}

        all_passed = True
        for tc in test_cases:
            with st.expander(tc["label"], expanded=True):
                test_vec, test_riasec = vectorize_user_profile(
                    user_skills=neutral_skills,
                    user_interests=tc["interests"],
                    all_feature_columns=list(X.columns),
                    scaler=scaler,
                    riasec_codes=RIASEC_CODES
                )
                test_results = recommender.recommend(
                    user_vector=test_vec,
                    user_riasec_input=tc["interests"],
                    top_n=5
                )
                test_recs = test_results["recommendations"]

                hits = 0
                for r in test_recs:
                    title_lower = r["title"].lower()
                    if any(kw in title_lower for kw in tc["expected_keywords"]):
                        hits += 1

                passed     = hits >= 2
                all_passed = all_passed and passed

                test_riasec_dist = test_results["riasec_profile"]
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.markdown(f"**RIASEC Profile:**")
                    for code, val in sorted(test_riasec_dist.items(), key=lambda x: -x[1]):
                        bar_w = int(val * 20)
                        st.markdown(f"`{code[:3]}` {'█'*bar_w} {val*100:.1f}%")

                with col_r2:
                    st.markdown(f"**Top 5 Recommended Careers:**")
                    for r in test_recs:
                        title_lower = r["title"].lower()
                        is_match    = any(kw in title_lower for kw in tc["expected_keywords"])
                        icon        = "✅" if is_match else "⚠️"
                        st.markdown(
                            f"{icon} **#{r['rank']}** {r['title']} "
                            f"({r['match_score']:.1f}%) — "
                            f"RIASEC: {r['riasec_dist'][test_results['dominant_types'][0]]*100:.0f}%"
                        )

                result_label = "✅ PASS" if passed else "❌ FAIL"
                result_color = "#2ecc71" if passed else "#e74c3c"
                st.markdown(
                    f"<div style='background:{result_color}22; border:1px solid {result_color}; "
                    f"border-radius:8px; padding:0.5rem 1rem; margin-top:0.5rem;'>"
                    f"<strong style='color:{result_color}'>{result_label}</strong> — "
                    f"{hits}/5 careers match expected domain keywords</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        overall_color = "#2ecc71" if all_passed else "#e74c3c"
        overall_label = "ALL TESTS PASSED — Model is working correctly!" if all_passed \
                        else "SOME TESTS FAILED — Results may not be accurate. Retrain the model."
        st.markdown(
            f"<div style='background:{overall_color}22; border:2px solid {overall_color}; "
            f"border-radius:12px; padding:1rem 1.5rem; text-align:center;'>"
            f"<strong style='color:{overall_color}; font-size:1.1rem'>{overall_label}</strong></div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("**📋 How to interpret this:**")
        st.markdown("""
        - **PASS**: At least 2 of the top-5 careers match the expected domain — model is correctly
          differentiating career domains based on interest input ✅
        - **FAIL**: Results are random/unrelated to the stated interest — this usually means
          the model needs retraining, or the RIASEC filter is too loose ❌
        - **All scores ~same** (~75%): RIASEC alignment is computing equally for all candidates
          (happens if candidate pool is too small or RIASEC labels have low variance)
        - **Top score always 95%**: Score rescaling is overriding real differences —
          look at the raw scores instead to compare runs
        """)

else:
    # ── Default landing state ────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem; background: linear-gradient(135deg, #141929, #1a2240);
                border-radius: 20px; border: 1px solid #2a3550; margin: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
        <div style="font-size: 1.4rem; color: #e8eaf6; font-weight: 600; margin-bottom: 0.8rem;">
            Welcome to CareerAI
        </div>
        <div style="color: #8892a4; font-size: 1rem; max-width: 500px; margin: 0 auto;">
            Fill in your skills and interests in the sidebar, then click
            <strong style="color:#00d4ff">Find My Careers</strong> to get personalized
            AI-powered career recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    features = [
        ("🤖", "Hybrid AI Engine", "Multi-label RIASEC classifier + Cosine Similarity ranking"),
        ("📊", "11 ML Models", "From Logistic Regression to XGBoost & LightGBM"),
        ("🛤️", "Career Paths", "See what skills you need to transition careers"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:1.5rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="color:#e8eaf6; font-weight:600; margin-bottom:0.4rem;">{title}</div>
                <div style="color:#8892a4; font-size:0.85rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)