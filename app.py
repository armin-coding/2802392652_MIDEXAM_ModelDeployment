import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Placement & Salary Predictor", layout="wide")

st.title("🎓 Student Placement Prediction System")
st.markdown("Aplikasi ini memprediksi status penempatan kerja dan estimasi gaji mahasiswa.")

# --- 2. LOAD MODELS (Langsung dari Root Repository) ---
@st.cache_resource
def load_models():
    # Karena file .pkl berjejer dengan app.py di GitHub
    clf_path = "classifier_model.pkl"
    reg_path = "regressor_model.pkl"
    
    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        st.error(f"❌ File Model (.pkl) tidak ditemukan di repository!")
        st.info("Pastikan nama file di GitHub adalah 'classifier_model.pkl' dan 'regressor_model.pkl'")
        st.stop()
        
    classifier = joblib.load(clf_path)
    regressor = joblib.load(reg_path)
    return classifier, regressor

try:
    clf_model, reg_model = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- 3. SIDEBAR / INPUT FORM ---
st.sidebar.header("Input Data Mahasiswa")

def user_input_features():
    # Input dari User
    cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, 7.5)
    internships = st.sidebar.number_input("Internships", 0, 10, 1)
    projects = st.sidebar.number_input("Projects", 0, 20, 2)
    work_exp_raw = st.sidebar.selectbox("Work Experience", ("Yes", "No"))
    ssc_p = st.sidebar.number_input("SSC Percentage", 0.0, 100.0, 70.0)
    hsc_p = st.sidebar.number_input("HSC Percentage", 0.0, 100.0, 70.0)

    # Konversi "Yes"/"No" ke 1/0 (Sesuaikan dengan data trainingmu)
    work_exp = 1 if work_exp_raw == "Yes" else 0

    # Dictionary data (PASTIKAN NAMA KEY SAMA DENGAN KOLOM SAAT TRAINING)
    data = {
        'cgpa': cgpa,
        'internships': internships,
        'projects': projects,
        'work_experience': work_exp,
        'ssc_p': ssc_p,
        'hsc_p': hsc_p
    }
    
    df = pd.DataFrame([data])
    
    # --- PROSES VALIDASI KOLOM (Kunci biar gak KeyError) ---
    try:
        # Ambil urutan kolom yang diharapkan model
        expected_columns = clf_model.feature_names_in_
        
        # Cek apakah ada kolom yang kurang di dictionary 'data'
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            st.warning(f"⚠️ Kolom berikut ada di model tapi belum kamu input: {missing_cols}")
            # Opsional: Tambahkan kolom kosong agar tidak crash
            for col in missing_cols:
                df[col] = 0 
        
        # Urutkan kolom sesuai keinginan model
        df = df[expected_columns]
        
    except AttributeError:
        st.info("💡 Model tidak menyimpan nama fitur (feature_names_in_). Pastikan urutan kolom manual sudah benar.")
        
    return df

# Ambil data input
input_df = user_input_features()

# Tampilkan data yang akan diprediksi (untuk verifikasi)
with st.expander("🔍 Lihat Detail Data Input"):
    st.write(input_df)

# --- 4. LOGIC PREDIKSI ---
st.markdown("---")
if st.button("🚀 Run Prediction"):
    try:
        # 1. Prediksi Status (Lulus/Tidak)
        prediction = clf_model.predict(input_df)[0]
        
        if prediction == 1:
            st.success("### STATUS: PLACED 🎉")
            st.balloons()
            
            # 2. Prediksi Gaji (Hanya jika Status = Placed)
            salary = reg_model.predict(input_df)[0]
            st.metric(label="Estimated Salary Package (LPA)", value=f"₹ {salary:.2f}")
            
        else:
            st.error("### STATUS: NOT PLACED ❌")
            st.info("Saran: Coba tingkatkan nilai CGPA atau perbanyak proyek dan pengalaman magang.")
            
    except Exception as e:
        st.error(f"🚨 Terjadi kesalahan saat prediksi: {e}")
        
        # DEBUG MODE: Muncul kalau error fitur
        if "feature_names_in_" in dir(clf_model):
            st.subheader("Bantuan Debugging:")
            st.write("Kolom yang diminta model:", clf_model.feature_names_in_.tolist())
            st.write("Kolom yang kamu kirim:", input_df.columns.tolist())
