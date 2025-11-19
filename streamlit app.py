import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Liver Disease Detection Using Random Forest Model")

# ---------------------------------------------------------
# LOAD AND CLEAN DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Liver_data.csv", sep=None, engine="python")

    # Strip spaces
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # FIX sex column
    df["sex"] = df["sex"].str.lower().map({
        "m": 1, "male": 1,
        "f": 0, "female": 0
    })

    # Convert numeric columns
    numeric_cols = df.columns.drop("category")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Map disease categories
    df["category"] = df["category"].str.lower().map({
        "no_disease": 0,
        "suspect_disease": 1,
        "hepatitis": 2,
        "fibrosis": 3,
        "cirrhosis": 4
    })

    df = df.dropna()

    return df


df = load_data()

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
X = df.drop("category", axis=1)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# USER INPUT FORM
# ---------------------------------------------------------
st.header("Enter Patient Values")
defaults = X.median()

inputs = {}
for col in X.columns:
    if col == "sex":
        inputs[col] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    else:
        inputs[col] = st.number_input(col, value=float(defaults[col]))

# Convert to dataframe
input_data = pd.DataFrame([inputs])
input_scaled = scaler.transform(input_data)

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]

    # Map back to names
    category_map = {
        0: "No Disease",
        1: "Suspect Disease",
        2: "Hepatitis",
        3: "Fibrosis",
        4: "Cirrhosis"
    }

    severity_map = {
        0: "None",
        1: "Mild",
        2: "Moderate",
        3: "High",
        4: "Severe"
    }

    disease = category_map[pred]
    severity = severity_map[pred]

    # Display results
    if pred == 0:
        st.success(f"🟢 Prediction: {disease}")
        st.info("No signs of liver disease detected.")
    else:
        st.error(f"🔴 Prediction: {disease}")
        st.warning(f"Severity Level: **{severity}**")

        st.write("### What this means:")
        if pred == 1:
            st.write("Early signs present. Recommend medical observation.")
        elif pred == 2:
            st.write("Inflammation of liver tissue detected.")
        elif pred == 3:
            st.write("Scarring (fibrosis) occurring in liver.")
        elif pred == 4:
            st.write("Advanced liver damage (cirrhosis). Immediate care needed!")
