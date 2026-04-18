import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    clf = joblib.load("classifier_model.pkl")
    reg = joblib.load("regressor_model.pkl")
    return clf, reg

classifier_model, regressor_model = load_models()

st.set_page_config(page_title="Student Placement Predictor", layout="centered")

st.title("🎓 Student Placement & Salary Prediction")
st.write("Isi data berikut untuk memprediksi placement dan estimasi gaji.")

# ==============================
# INPUT FEATURES (100% MATCH)
# ==============================
st.subheader("📥 Academic Performance")

ssc = st.number_input("SSC Percentage", 0.0, 100.0, 70.0)
hsc = st.number_input("HSC Percentage", 0.0, 100.0, 70.0)
degree = st.number_input("Degree Percentage", 0.0, 100.0, 75.0)
cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
entrance = st.number_input("Entrance Exam Score", 0.0, 100.0, 60.0)

st.subheader("💻 Skills")

technical = st.number_input("Technical Skill Score", 0.0, 100.0, 70.0)
soft = st.number_input("Soft Skill Score", 0.0, 100.0, 70.0)

st.subheader("📊 Experience")

internship = st.number_input("Internship Count", 0, 10, 1)
projects = st.number_input("Live Projects", 0, 10, 1)
work_exp = st.number_input("Work Experience (Months)", 0, 60, 0)
certifications = st.number_input("Certifications", 0, 20, 2)

st.subheader("📅 Activity & Attendance")

attendance = st.number_input("Attendance Percentage", 0.0, 100.0, 75.0)

st.subheader("👤 Personal Info")

gender = st.selectbox("Gender", ["Male", "Female"])

# ⚠️ treat as categorical (as per your model)
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict"):
    
    input_df = pd.DataFrame([{
        "ssc_percentage": ssc,
        "hsc_percentage": hsc,
        "degree_percentage": degree,
        "cgpa": cgpa,
        "entrance_exam_score": entrance,
        "technical_skill_score": technical,
        "soft_skill_score": soft,
        "internship_count": internship,
        "live_projects": projects,
        "work_experience_months": work_exp,
        "certifications": certifications,
        "attendance_percentage": attendance,
        "gender": gender,
        "extracurricular_activities": extracurricular
    }])

    try:
        # Classification
        placement = classifier_model.predict(input_df)[0]

        st.subheader("📊 Result")

        if placement == 1:
            st.success("✅ Likely to be PLACED")

            # Regression only if placed
            salary = regressor_model.predict(input_df)[0]
            st.info(f"💰 Estimated Salary: {salary:.2f} LPA")

        else:
            st.error("❌ Likely NOT placed")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ==============================
# OPTIONAL DEBUG
# ==============================
with st.expander("🔧 Debug Info"):
    st.write("Model expects these features:")
    st.write([
        "ssc_percentage", "hsc_percentage", "degree_percentage", "cgpa",
        "entrance_exam_score", "technical_skill_score", "soft_skill_score",
        "internship_count", "live_projects", "work_experience_months",
        "certifications", "attendance_percentage", "gender",
        "extracurricular_activities"
    ])
