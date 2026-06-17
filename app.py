import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc

# Page Configuration
st.set_page_config(
    page_title="Dashboard Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide"
)

# Load Model (No caching to prevent stale model state)
def load_model():
    return joblib.load("best_heart_disease_model.pkl")

model = load_model()

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f9fafd;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #c0392b;
        margin-bottom: 20px;
    }
    .header-style {
        background: linear-gradient(135deg, #c0392b, #8e44ad);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #c0392b, #922b21) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(192, 57, 43, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-style">
    <h1 style='color:white; text-align:center; margin:0; font-family: "Outfit", sans-serif; font-weight: 700;'>🫀 Prediksi Risiko Penyakit Jantung</h1>
    <p style='color:#f5b7b1; text-align:center; margin:8px 0 0; font-size: 1.1rem;'>Berbasis XGBoost Classifier — CRISP-DM</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Telkom_University_Logo.svg/1200px-Telkom_University_Logo.svg.png", width=160)
    st.markdown("---")
    st.markdown("### 📋 Panduan Penggunaan")
    st.markdown("""
    1. Pilih tab **🔮 Prediksi Risiko Mandiri** untuk memasukkan data pasien baru.
    2. Pilih tab **📊 Evaluasi & Metrik Model** untuk melihat performa model pengujian.
    """)
    st.markdown("---")
    st.markdown("### 🧬 Pasien Uji Demo")
    demo_profile = st.selectbox(
        "Pilih profil pasien uji:",
        options=["Input Manual", "Profil RISIKO RENDAH (Normal)", "Profil RISIKO TINGGI (Sakit)"]
    )
    st.markdown("---")
    st.markdown("**Metodologi:** CRISP-DM")
    st.markdown("**Model Champion:** XGBoost")

# Define default form values depending on selected demo profile
if demo_profile == "Profil RISIKO RENDAH (Normal)":
    d_age, d_gender, d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal = (
        37, "Wanita", "Non-Anginal Pain", 120, 215, "Tidak", "Abnormalitas Gelombang ST-T", 170, "Tidak", 0.0, "Upsloping", 0, "Normal"
    )
elif demo_profile == "Profil RISIKO TINGGI (Sakit)":
    d_age, d_gender, d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal = (
        58, "Pria", "Asymptomatic", 128, 259, "Tidak", "Normal", 130, "Ya", 3.0, "Flat", 2, "Reversible Defect"
    )
else: # Input Manual
    d_age, d_gender, d_cp, d_trestbps, d_chol, d_fbs, d_restecg, d_thalach, d_exang, d_oldpeak, d_slope, d_ca, d_thal = (
        54, "Pria", "Atypical Angina", 130, 240, "Tidak", "Normal", 150, "Tidak", 1.0, "Flat", 0, "Normal"
    )

# Tabs Setup
tab1, tab2 = st.tabs(["🔮 Prediksi Risiko Mandiri", "📊 Evaluasi & Metrik Model"])

with tab1:
    col_input, col_result = st.columns([1.2, 1])
    with col_input:
        st.markdown("## 📝 Data Klinis Pasien")
        with st.form("prediction_form"):
            with st.expander("🧑 Data Demografis", expanded=True):
                gender = st.selectbox("Jenis Kelamin (Sex)", options=["Wanita", "Pria"], index=["Wanita", "Pria"].index(d_gender))
                age = st.slider("Usia (Age)", min_value=10, max_value=100, value=d_age)
            with st.expander("❤️ Data Jantung & Tekanan Darah", expanded=True):
                cp_label = st.selectbox(
                    "Tipe Nyeri Dada (Chest Pain Type)", 
                    options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
                    index=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(d_cp)
                )
                trestbps = st.number_input("Tekanan Darah Istirahat (Resting Blood Pressure - mmHg)", min_value=0, value=int(d_trestbps))
                thalach = st.number_input("Detak Jantung Maksimum (Max Heart Rate)", min_value=0, value=int(d_thalach))
                exang_label = st.selectbox("Angina Akibat Olahraga (Exercise Induced Angina)", options=["Tidak", "Ya"], index=["Tidak", "Ya"].index(d_exang))
                oldpeak = st.slider("Depresi ST Akibat Olahraga (Oldpeak)", min_value=0.0, max_value=6.0, value=float(d_oldpeak), step=0.1)
                slope_label = st.selectbox(
                    "Kemiringan Segmen ST Puncak (Slope)", 
                    options=["Downsloping", "Flat", "Upsloping"],
                    index=["Downsloping", "Flat", "Upsloping"].index(d_slope)
                )
            with st.expander("🧪 Data Lab & Diagnostik", expanded=True):
                chol = st.number_input("Kolesterol Serum (Cholesterol - mg/dl)", min_value=0, value=int(d_chol))
                fbs_label = st.selectbox("Gula Darah Puasa > 120 mg/dl (Fasting Blood Sugar)", options=["Tidak", "Ya"], index=["Tidak", "Ya"].index(d_fbs))
                restecg_label = st.selectbox(
                    "Hasil EKG Istirahat (Resting ECG)", 
                    options=["Normal", "Abnormalitas Gelombang ST-T", "Hipertrofi Ventrikel Kiri"],
                    index=["Normal", "Abnormalitas Gelombang ST-T", "Hipertrofi Ventrikel Kiri"].index(d_restecg)
                )
                ca = st.selectbox("Jumlah Pembuluh Utama Terwarnai Fluoroskopi (CA)", options=[0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(d_ca))
                thal_label = st.selectbox("Hasil Tes Thalassemia (Thal)", options=["Normal", "Fixed Defect", "Reversible Defect", "Null/Other"], index=["Normal", "Fixed Defect", "Reversible Defect", "Null/Other"].index(d_thal if d_thal in ["Normal", "Fixed Defect", "Reversible Defect"] else "Null/Other"))
            submit = st.form_submit_button("Prediksi Risiko")

    with col_result:
        st.markdown("## 🔍 Hasil Analisis Prediksi")
        if submit:
            gender_map = {"Wanita": 0, "Pria": 1}
            cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
            fbs_map = {"Tidak": 0, "Ya": 1}
            restecg_map = {"Normal": 0, "Abnormalitas Gelombang ST-T": 1, "Hipertrofi Ventrikel Kiri": 2}
            exang_map = {"Tidak": 0, "Ya": 1}
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            thal_map = {"Normal": 2, "Fixed Defect": 1, "Reversible Defect": 3, "Null/Other": 0}

            input_df = pd.DataFrame([{
                "age": age,
                "sex": gender_map[gender],
                "cp": cp_map[cp_label],
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs_map[fbs_label],
                "restecg": restecg_map[restecg_label],
                "thalach": thalach,
                "exang": exang_map[exang_label],
                "oldpeak": oldpeak,
                "slope": slope_map[slope_label],
                "ca": ca,
                "thal": thal_map[thal_label]
            }])

            # Predict
            probability = model.predict_proba(input_df)[0]
            prob_sehat = probability[0] * 100  # Target 0 = Sehat (Normal)
            prob_sakit = probability[1] * 100  # Target 1 = Penyakit Jantung (Sakit)
            is_sakit = model.predict(input_df)[0] == 1

            if is_sakit:
                st.error(f"## 🚨 RISIKO TINGGI\n**Pasien diprediksi BERISIKO Penyakit Jantung**")
                st.metric("Probabilitas Berisiko", f"{prob_sakit:.1f}%")
                st.warning("⚠️ Segera lakukan pemeriksaan lanjutan ke spesialis jantung.")
            else:
                st.success(f"## ✅ RISIKO RENDAH\n**Pasien diprediksi TIDAK BERISIKO Penyakit Jantung**")
                st.metric("Probabilitas Sehat", f"{prob_sehat:.1f}%")
                st.info("ℹ️ Tetap jaga pola hidup sehat dan lakukan pemeriksaan berkala.")

            st.markdown("---")
            st.markdown("#### 📊 Distribusi Probabilitas")
            st.progress(int(prob_sakit))
            col_a, col_b = st.columns(2)
            col_a.metric("Tidak Berisiko (Sehat)", f"{prob_sehat:.1f}%")
            col_b.metric("Berisiko (Penyakit)", f"{prob_sakit:.1f}%")
        else:
            st.info("ℹ️ Silakan isi data pasien di form sebelah kiri, atau pilih **Pasien Uji Demo** di sidebar.")

with tab2:
    st.markdown("## 📊 Evaluasi Performa Model pada Data Test")
    st.write("Metrik di bawah menunjukkan hasil pengujian model champion **XGBoost** terbaik pada data uji (test set).")
    st.markdown("---")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""
        <div class="metric-card">
            <h3 style='margin:0; color:#555;'>Akurasi</h3>
            <h1 style='margin:5px 0; color:#c0392b;'>83.61%</h1>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class="metric-card">
            <h3 style='margin:0; color:#555;'>Sensitivitas (Recall)</h3>
            <h1 style='margin:5px 0; color:#c0392b;'>85.71%</h1>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div class="metric-card">
            <h3 style='margin:0; color:#555;'>F1-Score</h3>
            <h1 style='margin:5px 0; color:#c0392b;'>82.76%</h1>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
        <div class="metric-card">
            <h3 style='margin:0; color:#555;'>ROC AUC Score</h3>
            <h1 style='margin:5px 0; color:#c0392b;'>0.8647</h1>
        </div>
        """, unsafe_allow_html=True)
    st.success("✅ Model memiliki sensitivitas yang sangat baik (85.71%) untuk meminimalkan False Negative.")

    st.markdown("---")

