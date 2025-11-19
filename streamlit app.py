import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Disease Prediction", layout="wide")
st.title("Liver Disease Prediction App (No .pkl Files)")

st.write("Upload your Liver Dataset CSV to train the model directly in the app.")

# ------------------------------
# 1. UPLOAD DATASET
# ------------------------------
uploaded_file = st.file_uploader("Upload Liver_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";").dropna()

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # 2. PREPROCESSING
    # ------------------------------
    df["sex"] = df["sex"].map({"m": 1, "f": 0})

    label_encoder = LabelEncoder()
    df["category"] = label_encoder.fit_transform(df["category"])

    X = df.drop("category", axis=1)
    y = df["category"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ------------------------------
    # 3. TRAIN MODEL
    # ------------------------------
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train_scaled, y_train)

    st.success("Model trained successfully!")

    # ------------------------------
    # 4. USER INPUT FOR PREDICTION
    # ------------------------------
    st.subheader("🔍 Predict Liver Disease")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120)
        tot_bil = st.number_input("Total Bilirubin", 0.0)
        direct_bil = st.number_input("Direct Bilirubin", 0.0)
        alk_phos = st.number_input("Alkaline Phosphotase", 0.0)
        sgot = st.number_input("SGOT", 0.0)

    with col2:
        sgpt = st.number_input("SGPT", 0.0)
        tot_prot = st.number_input("Total Proteins", 0.0)
        alb = st.number_input("Albumin", 0.0)
        ag_ratio = st.number_input("A/G Ratio", 0.0)
        sex = st.selectbox("Sex", ["Male", "Female"])

    if st.button("Predict"):
        sex_val = 1 if sex == "Male" else 0

        input_data = np.array(
            [[age, tot_bil, direct_bil, alk_phos, sgot, sgpt, tot_prot, alb, ag_ratio, sex_val]]
        )

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]

        st.subheader("🚨 Prediction Result")
        st.write(f"**Predicted Category:** {pred_label}")
else:
    st.info("Please upload your Liver_data.csv file to begin.")
