import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Liver Disease Prediction (Random Forest — No PKL Files)")

@st.cache_data
def load_data():
    df = pd.read_csv("Liver_data.csv", sep=';')

    # Strip column names
    df.columns = df.columns.str.strip()

    # Convert all numeric columns (except category, sex)
    numeric_cols = df.columns.drop(["category", "sex"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Fix "sex" column
    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"m": 1, "male": 1, "f": 0, "female": 0})
    )

    # ---------- FIX CATEGORY COLUMN ----------
    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"disease": 1, "no_disease": 0})
    )

    # Remove rows where category mapping failed
    df = df.dropna(subset=["category"])
    df["category"] = df["category"].astype(int)
    # ------------------------------------------

    df = df.dropna()
    return df


df = load_data()

# Split data
X = df.drop("category", axis=1)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# Streamlit User Input Section
# --------------------------------------------------

st.header("Enter Patient Details")

# default values (median of each column)
defaults = X.median()

age = st.number_input("Age", value=float(defaults["age"]))
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

total_bilirubin = st.number_input("Total Bilirubin", value=float(defaults["total_bilirubin"]))
direct_bilirubin = st.number_input("Direct Bilirubin", value=float(defaults["direct_bilirubin"]))
alk_phosphate = st.number_input("Alkaline Phosphate", value=float(defaults["alk_phosphate"]))
sgpt = st.number_input("SGPT", value=float(defaults["sgpt"]))
sgot = st.number_input("SGOT", value=float(defaults["sgot"]))
total_proteins = st.number_input("Total Proteins", value=float(defaults["total_proteins"]))
albumin = st.number_input("Albumin", value=float(defaults["albumin"]))
ag_ratio = st.number_input("A/G Ratio", value=float(defaults["ag_ratio"]))

# Collect input into DF
input_data = pd.DataFrame([[
    age, sex, total_bilirubin, direct_bilirubin,
    alk_phosphate, sgpt, sgot, total_proteins,
    albumin, ag_ratio
]], columns=X.columns)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]

if st.button("Predict"):
    if prediction == 1:
        st.error("⚠️ The model predicts: **Disease**")
    else:
        st.success("✅ The model predicts: **No Disease**")
