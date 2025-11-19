import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Liver_data.csv", sep=';')

    # Clean column names (strip spaces)
    df.columns = df.columns.str.strip()

    # Convert numeric columns, except category + sex
    numeric_cols = df.columns.drop(["category", "sex"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Clean 'sex'
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["sex"] = df["sex"].map({"m": 1, "male": 1, "f": 0, "female": 0})

    # Drop rows with NaN after conversion
    df = df.dropna()

    return df


df = load_data()

st.title("Liver Disease Prediction")

# -------------------------------------------------------------
# 2. Prepare data
# -------------------------------------------------------------
FEATURES = [
    'age',
    'sex',
    'albumin',
    'alkaline_phosphatase',
    'alanine_aminotransferase',
    'aspartate_aminotransferase',
    'bilirubin',
    'cholinesterase',
    'cholesterol',
    'creatinina',
    'gamma_glutamyl_transferase',
    'protein'
]

X = df[FEATURES]
y = df["category"].astype(int)

# Encode target if needed
if y.dtype != np.int64 and y.dtype != np.int32:
    y = y.astype('category').cat.codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------
# 3. Train Random Forest
# -------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Accuracy
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.write(f"### 🔍 Model Accuracy: **{acc:.2f}**")

# -------------------------------------------------------------
# 4. User Inputs
# -------------------------------------------------------------
st.header("Enter Patient Details")

def input_number(label, col_name):
    """Safe number input with fallbacks."""
    try:
        median_val = float(df[col_name].median())
    except Exception:
        median_val = 0.0
    return st.number_input(label, value=median_val)

age = input_number("Age", "age")
sex = st.selectbox("Sex", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 0

albumin = input_number("Albumin", "albumin")
alk_phos = input_number("Alkaline Phosphatase", "alkaline_phosphatase")
alt = input_number("Alanine Aminotransferase", "alanine_aminotransferase")
ast = input_number("Aspartate Aminotransferase", "aspartate_aminotransferase")
bilirubin = input_number("Bilirubin", "bilirubin")
cholinesterase = input_number("Cholinesterase", "cholinesterase")
cholesterol = input_number("Cholesterol", "cholesterol")
creatinina = input_number("Creatinina", "creatinina")
ggt = input_number("Gamma Glutamyl Transferase", "gamma_glutamyl_transferase")
protein = input_number("Protein", "protein")

# -------------------------------------------------------------
# 5. Prediction
# -------------------------------------------------------------
if st.button("Predict"):
    user_data = np.array([
        age,
        sex_val,
        albumin,
        alk_phos,
        alt,
        ast,
        bilirubin,
        cholinesterase,
        cholesterol,
        creatinina,
        ggt,
        protein
    ]).reshape(1, -1)

    user_scaled = scaler.transform(user_data)
    pred = model.predict(user_scaled)[0]

    result = "⚠️ Liver Disease Detected" if pred == 1 else "✅ No Liver Disease"
    st.subheader("Prediction Result")
    st.write(f"### {result}")
