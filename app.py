import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi Header
st.set_page_config(page_title="Placement & Salary Predictor", layout="wide")
st.title("🎓 Student Placement Prediction System")
st.markdown("Aplikasi ini memprediksi status penempatan kerja dan estimasi gaji mahasiswa.")

# Load Models
@st.cache_resource
def load_models():
    # Gunakan path relatif yang aman untuk Streamlit Cloud
    clf_path = "models/classifier_model.pkl"
    reg_path = "models/regressor_model.pkl"
    
    if not os.path.exists(clf_path):
        st.error(f"File {clf_path} tidak ditemukan!")
        st.stop()
        
    classifier = joblib.load(clf_path)
    regressor = joblib.load(reg_path)
    return classifier, regressor

try:
    clf_model, reg_model = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- SIDEBAR / INPUT FORM ---
st.sidebar.header("Input Data Mahasiswa")

def user_input_features():
    # ⚠️ SESUAIKAN: Pastikan list di bawah ini SAMA PERSIS dengan kolom di X_train.csv kamu!
    # Jika di X_train ada 'aptitude_test_score' atau 'gender', kamu HARUS menambahkannya di sini.
    
    cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, 7.5)
    internships = st.sidebar.number_input("Internships", 0, 10, 1)
    projects = st.sidebar.number_input("Projects", 0, 20, 2)
    work_exp = st.sidebar.selectbox("Work Experience", ("Yes", "No"))
    ssc_p = st.sidebar.number_input("SSC Percentage", 0.0, 100.0, 70.0)
    hsc_p = st.sidebar.number_input("HSC Percentage", 0.0, 100.0, 70.0)
    # soft_skills_score = st.sidebar.number_input("Soft Skills Score", 0.0, 100.0, 75.0) # Contoh jika ada

    data = {
        'cgpa': cgpa,
        'internships': internships,
        'projects': projects,
        'work_experience': work_exp,
        'ssc_p': ssc_p,
        'hsc_p': hsc_p,
        # 'soft_skills_score': soft_skills_score # Tambahkan pasangannya di sini
    }
    
    df = pd.DataFrame([data])
    
    # 🚨 FIX ERROR: Urutan kolom harus sama dengan saat model di-fit
    # Ambil urutan kolom asli dari model (jika menggunakan sklearn Pipeline)
    try:
        # Mencoba mengambil nama fitur dari preprocessor di dalam pipeline
        expected_columns = clf_model.feature_names_in_
        df = df[expected_columns]
    except AttributeError:
        # Jika tidak bisa, kamu harus menulis urutannya secara manual di sini
        # col_order = ['cgpa', 'internships', ...] 
        # df = df[col_order]
        pass
        
    return df

input_df = user_input_features()

# Tampilkan input untuk verifikasi (Bisa dihapus jika sudah aman)
with st.expander("Lihat Data Input"):
    st.write(input_df)

# --- PREDICTION LOGIC ---
if st.button("Predict Status"):
    try:
        # 1. Klasifikasi Status
        prediction = clf_model.predict(input_df)[0]
        
        if prediction == 1:
            st.success("### Status: PLACED 🎉")
            
            # 2. Jika Placed, Prediksi Gaji
            # Kadang regressor butuh input yang sama
            salary = reg_model.predict(input_df)[0]
            st.metric(label="Estimated Salary Package (LPA)", value=f"₹ {salary:.2f}")
        else:
            st.error("### Status: NOT PLACED ❌")
            st.info("Saran: Tingkatkan pengalaman magang atau nilai CGPA kamu.")
            
    except ValueError as v:
        st.error(f"🚨 Error Fitur: {v}")
        st.info("Pastikan jumlah kolom di app.py sama dengan saat training.")
