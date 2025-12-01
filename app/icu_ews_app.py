import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

MODEL_PATH = "src/models/random_forest_sepsis.pkl"
FEATURE_COLS_PATH = "src/models/feature_cols.pkl"
FEATURE_MEANS_PATH = "src/models/feature_means.json"

st.set_page_config(page_title="ICU Sepsis Early Warning System", layout="centered")

st.title("🩺 ICU Sepsis Early Warning System")
st.markdown(
    """
This app uses a **Random Forest** model trained on ICU data  
to estimate the **risk of sepsis** based on vital signs and context.

> ⚠️ **Disclaimer:** Educational/demo only – not for real clinical use.
"""
)

# -------------------- LOAD MODEL & METADATA --------------------
@st.cache_resource
def load_model_and_metadata():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(f"Feature columns file not found at: {FEATURE_COLS_PATH}")
    if not os.path.exists(FEATURE_MEANS_PATH):
        raise FileNotFoundError(f"Feature means file not found at: {FEATURE_MEANS_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    with open(FEATURE_MEANS_PATH, "r") as f:
        feature_means = json.load(f)

    return model, feature_cols, feature_means

try:
    model, feature_cols, feature_means = load_model_and_metadata()
except Exception as e:
    st.error(f"❌ Could not load model or metadata: {e}")
    st.info("Make sure you have saved the model and feature files in src/models/.")
    st.stop()

st.success("✅ Model loaded successfully.")

st.markdown("---")

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("Patient Inputs")

age = st.sidebar.number_input("Age (years)", min_value=0, max_value=110, value=65)
gender_label = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender = 1.0 if gender_label == "Male" else 0.0

iculos = st.sidebar.number_input("ICU Length of Stay (ICULOS, hours)", min_value=0, max_value=500, value=24)
hour = st.sidebar.number_input("Hour (since measurement start)", min_value=0, max_value=72, value=12)
hosp_adm_time = st.sidebar.number_input("Time since hospital admission (hours, can be negative)", 
                                        min_value=-300, max_value=100, value=-12)

st.sidebar.markdown("### Vital Signs")

hr = st.sidebar.number_input("Heart Rate (HR)", min_value=30, max_value=220, value=95)
o2sat = st.sidebar.number_input("Oxygen Saturation (O2Sat %)", min_value=50, max_value=100, value=96)
sbp = st.sidebar.number_input("Systolic BP (SBP mmHg)", min_value=60, max_value=220, value=120)
map_val = st.sidebar.number_input("Mean Arterial Pressure (MAP mmHg)", min_value=40, max_value=150, value=80)
dbp = st.sidebar.number_input("Diastolic BP (DBP mmHg)", min_value=30, max_value=140, value=70)
resp = st.sidebar.number_input("Respiratory Rate (Resp /min)", min_value=5, max_value=60, value=18)

# Optional extra features some datasets have
temp = st.sidebar.number_input("Temperature (°C)", min_value=30.0, max_value=43.0, value=37.0)
fio2 = st.sidebar.number_input("FiO2 (%)", min_value=0.0, max_value=100.0, value=30.0)
etco2 = st.sidebar.number_input("EtCO2 (mmHg)", min_value=0.0, max_value=80.0, value=32.0)

# -------------------- BUILD INPUT ROW --------------------
def build_feature_row():
    # Start with training means for all features
    row = dict(feature_means)

    overrides = {
        "Age": age,
        "Gender": gender,
        "ICULOS": iculos,
        "Hour": hour,
        "HospAdmTime": hosp_adm_time,
        "HR": hr,
        "O2Sat": o2sat,
        "SBP": sbp,
        "MAP": map_val,
        "DBP": dbp,
        "Resp": resp,
        "Temp": temp,
        "FiO2": fio2,
        "EtCO2": etco2,
    }

    for k, v in overrides.items():
        if k in row:
            row[k] = float(v)

    ordered_values = [row.get(col, 0.0) for col in feature_cols]
    return pd.DataFrame([ordered_values], columns=feature_cols)

# -------------------- PREDICTION --------------------
st.markdown("### Predict Sepsis Risk")

if st.button("🔍 Run Sepsis Risk Prediction"):
    input_df = build_feature_row()
    prob = model.predict_proba(input_df)[0, 1]
    risk_pct = prob * 100

    if prob < 0.15:
        risk_label = "Low"
        color = "🟢"
    elif prob < 0.40:
        risk_label = "Moderate"
        color = "🟠"
    else:
        risk_label = "High"
        color = "🔴"

    st.subheader("Predicted Sepsis Risk")
    st.markdown(f"**Estimated risk: `{risk_pct:.1f}%`**")
    st.markdown(f"**Risk category: {color} {risk_label}**")

    with st.expander("Show model input features"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))

else:
    st.info("Adjust parameters in the sidebar and click **'Run Sepsis Risk Prediction'**.")
