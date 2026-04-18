import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_models():
    clf = joblib.load("classifier_model.pkl")
    reg = joblib.load("regressor_model.pkl")
    return clf, reg

classifier_model, regressor_model = load_models()

st.title("🎓 Student Placement & Salary Prediction")

st.write("Masukkan data mahasiswa:")

# ==============================
# INPUT (WAJIB SESUAI DATA TRAINING)
# ==============================
age = st.number_input("Age", 18, 40, 22)
cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
internships = st.number_input("Internships", 0, 10, 1)
projects = st.number_input("Projects", 0, 20, 2)

# categorical (contoh, sesuaikan!)
gender = st.selectbox("Gender", ["Male", "Female"])
stream = st.selectbox("Stream", ["CS", "IT", "ECE", "MECH"])
hostel = st.selectbox("Hostel", ["Yes", "No"])

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict"):
    input_df = pd.DataFrame({
        "age": [age],
        "cgpa": [cgpa],
        "internships": [internships],
        "projects": [projects],
        "gender": [gender],
        "stream": [stream],
        "hostel": [hostel]
    })

    try:
        # Classification
        placement = classifier_model.predict(input_df)[0]

        # Regression (hanya jika placed)
        salary = regressor_model.predict(input_df)[0]

        st.subheader("📊 Hasil")

        if placement == 1:
            st.success("✅ Mahasiswa kemungkinan TER-PLACED")
            st.info(f"💰 Prediksi Gaji: {salary:.2f} LPA")
        else:
            st.error("❌ Mahasiswa kemungkinan TIDAK ter-placed")

    except Exception as e:
        st.error(f"❌ Error: {e}")
