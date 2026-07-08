import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

st.set_page_config(
    page_title="Mental Well-being Assessment",
    page_icon="🧠",
    layout="wide"
)

def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("images/bg.png")
brain = get_base64("images/brain.png")

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{bg}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load comparative-insights dataset
# ------------------------------
# NOTE: use a relative path so this works on any machine, not just yours.
# Place encoded_dataset.csv in the same folder as this app.py.
df = pd.read_csv("encoded_dataset.csv")

if "start_assessment" not in st.session_state:
    st.session_state.start_assessment = False

# ------------------------------
# Load Model + Threshold
# ------------------------------
# These come from Cell 67 of the notebook:
#   joblib.dump(best_model, 'final_random_forest_model.pkl')
#   joblib.dump(final_threshold, 'final_threshold.pkl')
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

# ------------------------------
# Main Heading
# ------------------------------
st.markdown("""
<style>
div.stButton > button {
    background-color: #DFF6DD !important;
    height: 80px !important;
    border-radius: 20px !important;
    border: none !important;
}
div.stButton > button span {
    font-size: 50px !important;
    font-weight: 700 !important;
    color: #333333 !important;
}
div.stButton > button:hover {
    background-color: #C8E6C9 !important;
}
.hero{
background:#E9FFDB;
padding:35px;
border-radius:25px;
box-shadow:0px 5px 18px rgba(0,0,0,.08);
text-align:center;
margin-bottom:25px;
}
div.stButton > button{
transition:.3s;
}
div.stButton > button:hover{
transform:translateY(-3px);
box-shadow:0 10px 25px rgba(0,0,0,.15);
}
.card{
background:#F8FFF8;
padding:20px;
border-radius:18px;
margin-bottom:20px;
box-shadow:0 4px 12px rgba(0,0,0,.05);
}
.card{
animation:fade .7s ease;
}
@keyframes fade{
from{
opacity:0;
transform:translateY(20px);
}
to{
opacity:1;
transform:translateY(0);
}
}
.stSlider{
accent-color:#2E7D32;
}


/* Filled part */
.stSlider [data-baseweb="slider"] div[role="slider"] + div {
    background: #2E8B57 !important;
}

/* Empty track */
.stSlider [data-baseweb="slider"] > div > div {
    background: white !important;
}



/* Number input box */
div[data-testid="stNumberInput"] input{
    border:2px solid #6BCB77 !important;
    border-radius:10px !important;
}
div[data-testid="stNumberInput"] input:focus{
    border:2px solid #2E7D32 !important;
    box-shadow:0 0 0 1px #2E7D32 !important;
}

/* Text input box (in case you add any) */
div[data-testid="stTextInput"] input{
    border:2px solid #6BCB77 !important;
    border-radius:10px !important;
}
div[data-testid="stTextInput"] input:focus{
    border:2px solid #2E7D32 !important;
    box-shadow:0 0 0 1px #2E7D32 !important;
}

/* Selectbox border */
div[data-testid="stSelectbox"] > div > div{
    border:2px solid #6BCB77 !important;
    border-radius:10px !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within{
    border:2px solid #2E7D32 !important;
    box-shadow:0 0 0 1px #2E7D32 !important;
}


@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{
font-family:'Poppins',sans-serif;
}
hr{
border:none;
height:2px;
background:#DDEEDC;
margin:30px 0;
}
.sidebar-card{
background:#FCFFFC;
padding:25px;
border-radius:24px;
box-shadow:0 8px 18px rgba(0,0,0,.10);
margin-bottom:20px;
position:relative;
overflow:hidden;
}
.quote-card{
background:#EAF8EA;
padding:18px;
border-radius:18px;
box-shadow:0 5px 12px rgba(0,0,0,.08);
text-align:center;
}
.green{
color:#2E7D32;
font-weight:700;
}
</style>
""", unsafe_allow_html=True)
#######################################################

left, right = st.columns([1.4, 3], gap="large")

with left:
    st.markdown(f"""
<div class="sidebar-card" style="position:relative;overflow:hidden;">
<img src="data:image/png;base64,{brain}"
style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:240px;opacity:0.08;z-index:0;">
<div style="position:relative;z-index:2;">
<h2 style="text-align:center; color:darkgreen;">About</h2>
<div style="
    width: 60%;
    height: 2px;
    margin: 10px auto 18px auto;
    background: linear-gradient(to right, transparent, #2E7D32, transparent);
    border-radius: 10px;
"></div>
<h3 style="text-align:center;">
Mental Health Assessment<br>
Using Machine Learning
</h3>
<hr>
<b>Machine Learning Model</b><br><br>
Random Forest Classifier (Tuned, Threshold-Optimized)
<hr>
<b>Developed By</b><br><br>
<span class="green">Iqra Fatima Umang</span><br>
Master of Computer Science and Applications<br>
Aligarh Muslim University
<br><br>
<div class="quote-card">
💚<br><br>
<i>
"Taking care of your mind is just as important as taking care of your body."
</i>
</div>
</div>
</div>
""", unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class='hero'>
    <h1>Mental Well-being Assessment</h1>
    <p>
    Machine Learning based depression screening
    using academic and lifestyle factors.
    </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.start_assessment:
        st.divider()
        if st.button("▶Start Assessment", use_container_width=True):
            st.session_state.start_assessment = True
            st.rerun()

    if st.session_state.start_assessment:

        st.write("Fill in the details below to predict the likelihood of depression.")
        st.divider()
        st.subheader("📝 Enter Student Details")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])

            age = st.number_input("Age", min_value=11, max_value=34, value=22, step=1)

            city = st.selectbox("City", sorted(CITY_OPTIONS))

            degree_level = st.selectbox(
                "Education Level",
                ["Higher Secondary", "Graduated", "Post Graduated"]
            )

            cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.01)

            academic_pressure = st.slider("Academic Pressure", 1, 5, 3)

            study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)

        with col2:
            sleep_duration = st.selectbox(
                "Sleep Duration",
                ["Less than 5 Hours", "5-6 Hours", "7-8 Hours", "More than 8 Hours"]
            )

            dietary = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

            suicidal = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])

            study_hours = st.slider("Work/Study Hours", 0, 12, 6)

            financial = st.slider("Financial Stress", 1, 5, 3)

            family = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

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

        st.divider()

        if st.button("💙 Check Your Mental Well-being", use_container_width=True):

            # Build the base 12 features in exact training order
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

            # One-hot encode the selected city — all zero except the chosen one
            for c in CITY_OPTIONS:
                row[f'City_{c}'] = 1 if c == city else 0

            # Assemble in the exact column order the model expects
            input_data = np.array([[row[col] for col in FEATURE_COLUMNS]])

            with st.spinner("Analyzing your responses..."):
                probability = model.predict_proba(input_data)[0][1]
                prediction = int(probability >= final_threshold)

            st.markdown("""
            <div class="card">
            <h2>Assessment Summary</h2>
            </div>
            """, unsafe_allow_html=True)

            if probability < 0.40:
                st.success("🟢 Your responses indicate a lower likelihood of depression.")
            elif probability < 0.70:
                st.warning("🟡 Your responses suggest some signs that may benefit from attention.")
            else:
                st.error("🔴 Your responses indicate that additional mental health support may be beneficial.")

            st.metric("Prediction Confidence", f"{probability*100:.2f}%")
            st.progress(float(probability))

            similar = df[(df["Age"] >= age - 2) & (df["Age"] <= age + 2)]

            if len(similar) > 0:
                avg_pressure = similar["Academic Pressure"].mean()
                avg_satisfaction = similar["Study Satisfaction"].mean()
                avg_hours = similar["Work/Study Hours"].mean()
                avg_financial = similar["Financial Stress"].mean()
                depression_percent = similar["Depression"].mean() * 100

                st.markdown(f"""
                    <div class="card">
                <h3 style="margin-bottom:18px;">📊 Comparative Insights</h3>
                <h5 style="margin-bottom:15px; color:#4b5563; font-weight:500;">
                    Based on observed patterns in student records who share your age ({age}):
                </h5>
                <p style="margin:8px 0;">
                    <b>📚 Average Academic Pressure:</b> {avg_pressure:.1f}/5
                </p>
                <p style="margin:8px 0;">
                    <b>😊 Average Study Satisfaction:</b> {avg_satisfaction:.1f}/5
                </p>
                <p style="margin:8px 0;">
                    <b>⏳ Average Study Hours:</b> {avg_hours:.1f} hrs/day
                </p>
                <p style="margin:8px 0 18px 0;">
                    <b>💰 Average Financial Stress:</b> {avg_financial:.1f}/5
                </p>
                <hr style="margin:18px 0; border:0.5px solid #d9d9d9;">
                <h4 style="margin-bottom:12px;">🧠 Depression Statistics</h4>
                <p style="margin:8px 0;">
                    Approximately <b style="color:#d97706;">{depression_percent:.1f}%</b>
                    of these students were classified as having depression.
                </p>
                <p style="margin:8px 0;">
                    The remaining <b style="color:#16a34a;">{100-depression_percent:.1f}%</b>
                    were not.
                </p>
                </div>
            """, unsafe_allow_html=True)

            st.divider()
            if st.button("Start New Assessment"):
                st.session_state.start_assessment = False
                st.rerun()

    st.divider()

    st.warning("""
    ## Need Immediate Support?

    This application is intended for educational and research purposes only and **does not replace professional medical advice or diagnosis**.

    ### 🇮🇳 India
    **Tele-MANAS (24×7 National Mental Health Helpline)**

    📞 **14416**

    📞 **1800-891-4416**
    """)

    st.markdown("---")

    st.caption("Developed by Iqra Fatima Umang | MCA | Aligarh Muslim University")