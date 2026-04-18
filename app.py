import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==============================
# PATH CONFIG
# ==============================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "classifier_model.pkl"
DATA_SAMPLE_PATH = BASE_DIR / "artifacts" / "classification" / "X_train.csv"

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_sample():
    return pd.read_csv(DATA_SAMPLE_PATH)

model = load_model()
df_sample = load_sample()

# ==============================
# UI
# ==============================
st.title("🎓 Student Placement Prediction App")
st.write("Masukkan data mahasiswa untuk memprediksi apakah akan ter-placed atau tidak.")

st.subheader("📥 Input Data")

input_data = {}

# Generate input otomatis berdasarkan dataset
for col in df_sample.columns:
    if df_sample[col].dtype == "object":
        input_data[col] = st.selectbox(
            f"{col}",
            options=df_sample[col].dropna().unique()
        )
    else:
        input_data[col] = st.number_input(
            f"{col}",
            value=float(df_sample[col].mean())
        )

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Prediksi"):
    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("📊 Hasil Prediksi")

        if prediction == 1:
            st.success("✅ Mahasiswa kemungkinan TER-PLACED")
        else:
            st.error("❌ Mahasiswa kemungkinan TIDAK ter-placed")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

# ==============================
# DEBUG INFO (optional)
# ==============================
with st.expander("🔧 Debug Info"):
    st.write("Kolom yang digunakan model:")
    st.write(df_sample.columns.tolist())

    st.write("Contoh data:")
    st.dataframe(df_sample.head())
