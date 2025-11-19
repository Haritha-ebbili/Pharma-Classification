import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model, scaler, and label encoder
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, label_encoder

# Attempt to load from root or /models folder

import os

def safe_load(path_options):
    for p in path_options:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(f"None of the paths exist: {path_options}")

model = safe_load(['random_forest_model.pkl', 'models/random_forest_model.pkl'])
scaler = safe_load(['scaler.pkl', 'models/scaler.pkl'])
label_encoder = safe_load(['label_encoder.pkl', 'models/label_encoder.pkl'])
()

st.set_page_config(page_title="Liver Disease Prediction", layout="wide")
st.title("Liver Disease Prediction App")

st.write("Enter patient details below to predict liver disease.")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    tot_bil = st.number_input("Total Bilirubin", min_value=0.0)
    direct_bil = st.number_input("Direct Bilirubin", min_value=0.0)
    alk_phos = st.number_input("Alkaline Phosphotase", min_value=0.0)
    sgot = st.number_input("SGOT", min_value=0.0)
    sgpt = st.number_input("SGPT", min_value=0.0)
    tot_prot = st.number_input("Total Protiens", min_value=0.0)
    alb = st.number_input("Albumin", min_value=0.0)
    ag_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0)
    sex = st.selectbox("Sex", ["Male", "Female"])

    submitted = st.form_submit_button("Predict")

if submitted:
    sex_val = 1 if sex == "Male" else 0

    input_data = np.array([[age, tot_bil, direct_bil, alk_phos, sgot, sgpt, tot_prot, alb, ag_ratio, sex_val]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]

    st.subheader("Prediction Result")
    st.write(f"**Predicted Category:** {pred_label}")
