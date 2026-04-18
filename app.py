import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Konfigurasi Header
st.set_page_config(page_title="Placement & Salary Predictor", layout="wide")
st.title("🎓 Student Placement Prediction System")
st.markdown("Aplikasi ini memprediksi status penempatan kerja dan estimasi gaji mahasiswa.")

# Load Models
@st.cache_resource
def load_models():
    classifier = joblib.load("models/classifier_model.pkl")
    regressor = joblib.load("models/regressor_model.pkl")
    return classifier, regressor

try:
    clf_model, reg_model = load_models()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .pkl ada di folder 'models'. Error: {e}")
    st.stop()

# --- SIDEBAR / INPUT FORM ---
st.sidebar.header("Input Data Mahasiswa")

def user_input_features():
    # Sesuaikan input di bawah ini dengan fitur yang ada di X_train kamu
    cgpa = st.sidebar.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
    internships = st.sidebar.number_input("Internships", min_value=0, max_value=10, value=1)
    projects = st.sidebar.number_input("Projects", min_value=0, max_value=20, value=2)
    work_exp = st.sidebar.selectbox("Work Experience", ("Yes", "No"))
    ssc_p = st.sidebar.number_input("SSC Percentage", 0.0, 100.0, 70.0)
    hsc_p = st.sidebar.number_input("HSC Percentage", 0.0, 100.0, 70.0)
    
    # Bungkus ke DataFrame (Pastikan nama kolom SAMA PERSIS dengan saat training)
    data = {
        'cgpa': cgpa,
        'internships': internships,
        'projects': projects,
        'work_experience': work_exp,
        'ssc_p': ssc_p,
        'hsc_p': hsc_p,
        # Tambahkan kolom lain yang kamu gunakan di preprocessing
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- PREDICTION LOGIC ---
if st.button("Predict Status"):
    # 1. Klasifikasi Status
    prediction = clf_model.predict(input_df)[0]
    
    if prediction == 1:
        st.success("### Status: PLACED 🎉")
        
        # 2. Jika Placed, Prediksi Gaji
        salary = reg_model.predict(input_df)[0]
        st.metric(label="Estimated Salary Package (LPA)", value=f"₹ {salary:.2f}")
    else:
        st.error("### Status: NOT PLACED ❌")
        st.info("Saran: Tingkatkan pengalaman magang atau nilai CGPA kamu.")