import math
import base64
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Mental Well-being Assessment",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# Assets (optional — app still runs cleanly if a file is missing)
# ------------------------------
def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

brain = get_base64("images/brain.png")

# ------------------------------
# Load comparative-insights dataset
# ------------------------------
df = pd.read_csv("encoded_dataset.csv")

if "start_assessment" not in st.session_state:
    st.session_state.start_assessment = False

# ------------------------------
# Load Model + Threshold
# ------------------------------
model = joblib.load("final_random_forest_model.pkl")
final_threshold = joblib.load("final_threshold.pkl")

# Exact column order the model was trained on (main_encoded, minus 'Depression')
FEATURE_COLUMNS = [
    'Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction',
    'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
    'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness',
    'New_Degree',
    'City_Agra', 'City_Ahmedabad', 'City_Bangalore', 'City_Bhopal', 'City_Chennai',
    'City_Delhi', 'City_Faridabad', 'City_Ghaziabad', 'City_Hyderabad', 'City_Indore',
    'City_Jaipur', 'City_Kalyan', 'City_Kanpur', 'City_Kolkata', 'City_Lucknow',
    'City_Ludhiana', 'City_Meerut', 'City_Mumbai', 'City_Nagpur', 'City_Nashik',
    'City_Patna', 'City_Pune', 'City_Rajkot', 'City_Srinagar', 'City_Surat',
    'City_Thane', 'City_Vadodara', 'City_Varanasi', 'City_Vasai-Virar', 'City_Visakhapatnam'
]

CITY_OPTIONS = [c.replace('City_', '') for c in FEATURE_COLUMNS if c.startswith('City_')]

# ==============================================================
# DESIGN SYSTEM
# ==============================================================
# Palette   : clinical teal/ink, warm amber accent, muted brick-red alert
# Type      : Fraunces (display, editorial authority) + Inter (UI/body)
#             + IBM Plex Mono (data readouts — confidence %, stats)
# Signature : semicircular risk gauge with a needle, styled like a
#             physical instrument rather than a progress bar.
# ==============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
    --bg:#F3F7F5;
    --surface:#FFFFFF;
    --surface-alt:#EAF1EE;
    --ink:#1D2E2A;
    --ink-soft:#5B6E68;
    --primary:#173E38;
    --primary-soft:#2C6E62;
    --accent:#C08A3E;
    --safe:#2F8F63;
    --caution:#C08A3E;
    --alert:#B14A3B;
    --line:#DCE6E2;
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
    color:var(--ink);
}
.stApp{ background:var(--bg); }

h1, h2, h3, .display{
    font-family:'Fraunces', serif;
    color:var(--primary);
    letter-spacing:-0.01em;
}

.eyebrow{
    font-family:'IBM Plex Mono', monospace;
    font-size:12px;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:var(--primary-soft);
    font-weight:600;
}

hr{ border:none; height:1px; background:var(--line); margin:28px 0; }

/* ---------- Hero ---------- */
.hero{
    background:linear-gradient(135deg, var(--primary) 0%, #1F5049 100%);
    padding:40px 44px;
    border-radius:20px;
    margin-bottom:28px;
    color:#F3F7F5;
}
.hero h1{ color:#FFFFFF; font-size:2.1rem; margin:6px 0 10px 0; }
.hero p{ color:#D7E6E0; font-size:1.02rem; max-width:640px; margin:0; }

/* ---------- Cards ---------- */
.card{
    background:var(--surface);
    padding:26px 28px;
    border-radius:16px;
    border:1px solid var(--line);
    margin-bottom:20px;
}
.section-card{
    background:var(--surface);
    padding:22px 26px;
    border-radius:16px;
    border:1px solid var(--line);
    margin-bottom:22px;
}
.section-card h4{
    margin-top:0; margin-bottom:4px; font-size:1.05rem;
}
.section-sub{
    color:var(--ink-soft); font-size:0.86rem; margin-bottom:18px;
}

/* ---------- Sidebar dossier ---------- */
.sidebar-card{
    background:var(--surface);
    padding:28px 24px;
    border-radius:20px;
    border:1px solid var(--line);
    margin-bottom:20px;
    position:relative;
    overflow:hidden;
}
.sidebar-card h2{
    text-align:center; font-size:1.3rem; margin-bottom:4px;
}
.sidebar-rule{
    width:56px; height:3px; margin:10px auto 20px auto;
    background:var(--accent); border-radius:6px;
}
.meta-label{
    font-family:'IBM Plex Mono', monospace;
    font-size:11px; letter-spacing:0.1em; text-transform:uppercase;
    color:var(--primary-soft); font-weight:600; margin-bottom:2px;
}
.meta-value{ margin-bottom:16px; color:var(--ink); font-size:0.92rem; }
.quote-card{
    background:var(--surface-alt);
    padding:16px 18px;
    border-radius:14px;
    font-size:0.88rem;
    color:var(--ink-soft);
    font-style:italic;
    border-left:3px solid var(--accent);
}

/* ---------- Buttons ---------- */
div.stButton > button{
    background:var(--primary) !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:12px !important;
    padding:0.7em 1.4em !important;
    height:auto !important;
    font-family:'Inter', sans-serif !important;
    font-weight:600 !important;
    font-size:16px !important;
    letter-spacing:0.01em;
    transition:.2s ease;
}
div.stButton > button:hover{
    background:var(--primary-soft) !important;
    transform:translateY(-2px);
    box-shadow:0 8px 20px rgba(23,62,56,0.22);
}
div.stButton > button:focus-visible{
    outline:3px solid var(--accent) !important;
    outline-offset:2px;
}
div.stButton > button span{
    font-size:16px !important;
    font-weight:600 !important;
    color:#FFFFFF !important;
}

/* ---------- Inputs ---------- */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input{
    border:1.5px solid var(--line) !important;
    border-radius:10px !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus{
    border:1.5px solid var(--primary-soft) !important;
    box-shadow:0 0 0 1px var(--primary-soft) !important;
}
div[data-testid="stSelectbox"] > div > div{
    border:1.5px solid var(--line) !important;
    border-radius:10px !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within{
    border:1.5px solid var(--primary-soft) !important;
    box-shadow:0 0 0 1px var(--primary-soft) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] + div{
    background:var(--primary-soft) !important;
}

/* ---------- Result tiers ---------- */
.tier-banner{
    padding:16px 20px;
    border-radius:12px;
    font-weight:600;
    margin-bottom:18px;
    display:flex; align-items:center; gap:10px;
}
.tier-safe{ background:#E7F4EC; color:#1E6B48; border:1px solid #BFE3CD; }
.tier-caution{ background:#FBF1E1; color:#8A5E1F; border:1px solid #EAD3A6; }
.tier-alert{ background:#F8E7E4; color:#8A3327; border:1px solid #E9BCB3; }

.stat-readout{
    font-family:'IBM Plex Mono', monospace;
    font-weight:600;
    color:var(--primary);
}

.support-card{
    background:var(--surface);
    border:1px solid var(--line);
    border-left:4px solid var(--alert);
    padding:22px 26px;
    border-radius:16px;
    margin-top:8px;
}
.support-card h3{ margin-top:0; }
.support-number{
    font-family:'IBM Plex Mono', monospace;
    font-size:1.1rem;
    font-weight:600;
    color:var(--primary);
}
</style>
""", unsafe_allow_html=True)

# ==============================================================
# Gauge component (signature visual)
# ==============================================================
def render_gauge(probability: float, threshold: float) -> str:
    cx, cy, r = 150, 150, 118
    stroke = 26

    def point(theta_deg):
        theta = math.radians(theta_deg)
        return cx + r * math.cos(theta), cy - r * math.sin(theta)

    zones = [(180, 108, "#2F8F63"), (108, 54, "#C08A3E"), (54, 0, "#B14A3B")]
    arcs = ""
    for start, end, color in zones:
        x1, y1 = point(start)
        x2, y2 = point(end)
        arcs += (f'<path d="M{x1:.1f},{y1:.1f} A{r},{r} 0 0 1 {x2:.1f},{y2:.1f}" '
                 f'stroke="{color}" stroke-width="{stroke}" fill="none" stroke-linecap="butt"/>')

    needle_theta = 180 - (probability * 180)
    nx, ny = point(needle_theta)
    needle_x = cx + (nx - cx) * 0.8
    needle_y = cy - (cy - ny) * 0.8

    if probability < 0.40:
        tier_color = "#2F8F63"
    elif probability < 0.70:
        tier_color = "#C08A3E"
    else:
        tier_color = "#B14A3B"

    pct = f"{probability*100:.1f}"

    return f"""
    <svg viewBox="0 0 300 195" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:340px;display:block;margin:0 auto;">
        {arcs}
        <circle cx="{cx}" cy="{cy}" r="10" fill="#173E38"/>
        <line x1="{cx}" y1="{cy}" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
              stroke="#173E38" stroke-width="5" stroke-linecap="round"/>
        <text x="150" y="178" text-anchor="middle"
              font-family="'IBM Plex Mono', monospace" font-size="30" font-weight="600"
              fill="{tier_color}">{pct}%</text>
        <text x="150" y="193" text-anchor="middle"
              font-family="'IBM Plex Mono', monospace" font-size="10" letter-spacing="1"
              fill="#5B6E68">MODEL CONFIDENCE</text>
        <text x="{point(180)[0]:.1f}" y="168" text-anchor="middle" font-family="'IBM Plex Mono', monospace" font-size="10" fill="#5B6E68">LOW</text>
        <text x="{point(0)[0]:.1f}" y="168" text-anchor="middle" font-family="'IBM Plex Mono', monospace" font-size="10" fill="#5B6E68">HIGH</text>
    </svg>
    """

# ==============================================================
# Layout
# ==============================================================
left, right = st.columns([1.2, 3], gap="large")

with left:
    watermark = (f'<img src="data:image/png;base64,{brain}" '
                 'style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
                 'width:220px;opacity:0.06;z-index:0;">') if brain else ""
    st.markdown(f"""
    <div class="sidebar-card">
    {watermark}
    <div style="position:relative;z-index:2;">
        <h2>About This Tool</h2>
        <div class="sidebar-rule"></div>
        <div class="meta-label">Purpose</div>
        <div class="meta-value">Machine-learning-based depression screening
        using academic and lifestyle indicators.</div>
        <div class="meta-label">Model</div>
        <div class="meta-value">Random Forest Classifier — tuned and
        threshold-optimized.</div>
        <div class="meta-label">Developed by</div>
        <div class="meta-value">
            <b style="color:var(--primary);">Iqra Fatima Umang</b><br>
            Master of Computer Science and Applications <br>
            Aligarh Muslim University
        </div>
        <div class="quote-card">
            "Taking care of your mind is just as important as taking care
            of your body."
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="hero">
        <div class="eyebrow" style="color:#B7D9CE;">Screening Tool</div>
        <h1>Mental Well-being Assessment</h1>
        <p>A short, evidence-informed questionnaire that estimates depression
        risk from academic pressure, lifestyle, and personal history —
        for educational and research use.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.start_assessment:
        if st.button("Start Assessment →", use_container_width=True):
            st.session_state.start_assessment = True
            st.rerun()

    if st.session_state.start_assessment:

        st.markdown('<div class="eyebrow">Step 1 of 2 — Details</div>', unsafe_allow_html=True)
        st.write("")

        st.markdown("""
        <div class="section-card">
        <h4>👤 Personal &amp; Academic</h4>
        <div class="section-sub">Basic profile and academic standing.</div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            city = st.selectbox("City", sorted(CITY_OPTIONS))
        with c2:
            age = st.number_input("Age", min_value=11, max_value=34, value=22, step=1)
            degree_level = st.selectbox(
                "Education Level",
                ["Higher Secondary", "Graduated", "Post Graduated"]
            )
        with c3:
            cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.01)
            academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
        study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
        <h4>🌙 Lifestyle &amp; Wellbeing</h4>
        <div class="section-sub">Daily habits that influence mental load.</div>
        """, unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            sleep_duration = st.selectbox(
                "Sleep Duration",
                ["Less than 5 Hours", "5-6 Hours", "7-8 Hours", "More than 8 Hours"]
            )
            dietary = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
            study_hours = st.slider("Work/Study Hours", 0, 12, 6)
        with d2:
            financial = st.slider("Financial Stress", 1, 5, 3)
            family = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
            suicidal = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
        st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------
        # Convert Inputs to Numbers (matches notebook's encoding exactly)
        # ------------------------------
        gender_val = 0 if gender == "Male" else 1
        suicidal_val = 0 if suicidal == "No" else 1
        family_val = 0 if family == "No" else 1

        dietary_map = {"Healthy": 0, "Unhealthy": 1, "Moderate": 2}
        dietary_val = dietary_map[dietary]

        sleep_map = {
            "Less than 5 Hours": 0,
            "5-6 Hours": 1,
            "7-8 Hours": 2,
            "More than 8 Hours": 3
        }
        sleep_val = sleep_map[sleep_duration]

        degree_map = {"Graduated": 0, "Post Graduated": 1, "Higher Secondary": 2}
        degree_val = degree_map[degree_level]

        if st.button("Run Assessment", use_container_width=True):

            row = {
                'Gender': gender_val,
                'Age': age,
                'Academic Pressure': academic_pressure,
                'CGPA': cgpa,
                'Study Satisfaction': study_satisfaction,
                'Sleep Duration': sleep_val,
                'Dietary Habits': dietary_val,
                'Have you ever had suicidal thoughts ?': suicidal_val,
                'Work/Study Hours': study_hours,
                'Financial Stress': financial,
                'Family History of Mental Illness': family_val,
                'New_Degree': degree_val,
            }
            for c in CITY_OPTIONS:
                row[f'City_{c}'] = 1 if c == city else 0

            input_data = np.array([[row[col] for col in FEATURE_COLUMNS]])

            with st.spinner("Analyzing your responses..."):
                probability = model.predict_proba(input_data)[0][1]
                prediction = int(probability >= final_threshold)

            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Step 2 of 2 — Result</div>', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top:6px;'>Assessment Summary</h3>", unsafe_allow_html=True)

            if probability < 0.40:
                st.markdown('<div class="tier-banner tier-safe">🟢 Your responses indicate a lower likelihood of depression.</div>', unsafe_allow_html=True)
            elif probability < 0.70:
                st.markdown('<div class="tier-banner tier-caution">🟡 Your responses suggest some signs that may benefit from attention.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tier-banner tier-alert">🔴 Your responses indicate that additional mental health support may be beneficial.</div>', unsafe_allow_html=True)

            gcol1, gcol2 = st.columns([1, 1.3])
            with gcol1:
                st.markdown(f"<div class='card'>{render_gauge(probability, final_threshold)}</div>", unsafe_allow_html=True)
            with gcol2:
                st.markdown(f"""
                <div class="card">
                <div class="meta-label">Decision threshold</div>
                <div class="meta-value stat-readout">{final_threshold*100:.1f}%</div>
                <div class="meta-label">Classification</div>
                <div class="meta-value stat-readout">
                    {"Elevated risk indicators present" if prediction == 1 else "No elevated risk indicators"}
                </div>
                <div class="meta-label">Note</div>
                <div class="meta-value" style="font-size:0.82rem;">
                    This score reflects patterns learned from historical student
                    data. It is a screening signal, not a diagnosis.
                </div>
                </div>
                """, unsafe_allow_html=True)

            similar = df[(df["Age"] >= age - 2) & (df["Age"] <= age + 2)]

            if len(similar) > 0:
                avg_pressure = similar["Academic Pressure"].mean()
                avg_satisfaction = similar["Study Satisfaction"].mean()
                avg_hours = similar["Work/Study Hours"].mean()
                avg_financial = similar["Financial Stress"].mean()
                depression_percent = similar["Depression"].mean() * 100

                st.markdown("<div class='eyebrow' style='margin-top:8px;'>Comparative Insights</div>", unsafe_allow_html=True)
                st.caption(f"Based on {len(similar)} student records aged {age-2}–{age+2} in the reference dataset.")

                compare_df = pd.DataFrame({
                    "Metric": ["Academic Pressure", "Study Satisfaction", "Work/Study Hours", "Financial Stress"],
                    "You": [academic_pressure, study_satisfaction, study_hours, financial],
                    "Peer Average": [avg_pressure, avg_satisfaction, avg_hours, avg_financial],
                }).set_index("Metric")
                st.bar_chart(compare_df, color=["#173E38", "#C08A3E"])

                st.markdown(f"""
                <div class="card">
                <div class="meta-label">Depression prevalence in this peer group</div>
                <p style="margin:6px 0;">
                    <span class="stat-readout" style="font-size:1.3rem;">{depression_percent:.1f}%</span>
                    of similarly aged students in the reference dataset were classified as having depression;
                    <span class="stat-readout" style="font-size:1.3rem;">{100-depression_percent:.1f}%</span>
                    were not.
                </p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            if st.button("Start New Assessment"):
                st.session_state.start_assessment = False
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="support-card">
    <h3>Need Immediate Support?</h3>
    <p style="color:var(--ink-soft);">
    This application is intended for educational and research purposes only
    and <b>does not replace professional medical advice or diagnosis</b>.
    </p>
    <div class="meta-label">🇮🇳 Tele-MANAS — 24×7 National Mental Health Helpline</div>
    <p class="support-number">14416&nbsp;&nbsp;|&nbsp;&nbsp;1800-891-4416</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.caption("Developed by Iqra Fatima Umang · MCA · Aligarh Muslim University")
