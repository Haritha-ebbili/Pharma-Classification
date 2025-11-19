import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Liver Disease Prediction using Random Forest Model")

@st.cache_data
def load_data():
    df = pd.read_csv("Liver_data.csv", sep=None, engine="python")

    df.columns = df.columns.str.strip()

    # Convert numeric columns automatically
    numeric_cols = df.columns.drop(["category", "sex"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Sex mapping
    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"m": 1, "male": 1, "f": 0, "female": 0})
    )

    # FIXED LABEL MAPPING
    # "no_disease" = 0
    # Any other label (like "A", "B", "C") = disease = 1
    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda x: 0 if x == "no_disease" else 1)
    )

    df = df.dropna()
    df["category"] = df["category"].astype(int)

    return df

df = load_data()

# Split
X = df.drop("category", axis=1)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale correctly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # FIX ADDED

# Train RF model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# -------------------------------
# USER INPUTS
# -------------------------------

st.header("Patient Inputs")
defaults = X.median()

user_values = []
for col in X.columns:
    if col == "sex":
        user_values.append(st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x==1 else "Female"))
    else:
        user_values.append(st.number_input(col, value=float(defaults[col])))

input_df = pd.DataFrame([user_values], columns=X.columns)

# Scale input
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error("⚠️ Model Prediction: LIVER DISEASE")
    else:
        st.success("✅ Model Prediction: NO DISEASE")
