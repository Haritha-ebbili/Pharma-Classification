import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Liver Disease Prediction using Random Forest ")

@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("Liver_data.csv", sep=None, engine="python")

    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()

    # Fix last two weird columns
    df.rename(columns={
        "gamma_glutamyl_transferase": "gamma_glutamyl_transferase",
        "protein": "protein"
    }, inplace=True)

    # Convert numeric columns automatically
    numeric_cols = df.columns.drop(["category", "sex"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Map sex
    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"m": 1, "male": 1, "f": 0, "female": 0})
    )

    # Ensure valid category values (A = disease=1, no_disease=0)
    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"disease": 1, "no_disease": 0})
    )

    df = df.dropna()   # Drop invalid rows
    df["category"] = df["category"].astype(int)

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

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# --------------------------
# USER INPUT SECTION
# --------------------------

st.header("Patient Inputs")

defaults = X.median()

# All fields EXACTLY match your dataset
age = st.number_input("Age", value=float(defaults["age"]))
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
albumin = st.number_input("Albumin", value=float(defaults["albumin"]))
alk_phos = st.number_input("Alkaline Phosphatase", value=float(defaults["alkaline_phosphatase"]))
alt = st.number_input("Alanine Aminotransferase", value=float(defaults["alanine_aminotransferase"]))
ast = st.number_input("Aspartate Aminotransferase", value=float(defaults["aspartate_aminotransferase"]))
bilirubin = st.number_input("Bilirubin", value=float(defaults["bilirubin"]))
cholinesterase = st.number_input("Cholinesterase", value=float(defaults["cholinesterase"]))
cholesterol = st.number_input("Cholesterol", value=float(defaults["cholesterol"]))
creatinina = st.number_input("Creatinine", value=float(defaults["creatinina"]))
g_gt = st.number_input("Gamma Glutamyl Transferase", value=float(defaults["gamma_glutamyl_transferase"]))
protein = st.number_input("Protein", value=float(defaults["protein"]))

# Construct input row
input_data = pd.DataFrame([[
    age, sex, albumin, alk_phos, alt, ast, bilirubin,
    cholinesterase, cholesterol, creatinina, g_gt, protein
]], columns=X.columns)

# Scale
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]

if st.button("Predict"):
    if prediction == 1:
        st.error("⚠️ Model Prediction: LIVER DISEASE")
    else:
        st.success("✅ Model Prediction: NO DISEASE")
