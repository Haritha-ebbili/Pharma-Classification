import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Disease Prediction", page_icon="🧬")

st.title("🧬 Liver Disease Prediction")
st.write("This app trains a Random Forest model directly from the dataset .")

# -----------------------------
# LOAD AND PREPARE DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Liver_data.csv", sep=";")

    # Strip column and label spaces
    df.columns = df.columns.str.strip()
    df["category"] = df["category"].str.strip()

    # Convert to binary (disease vs no_disease)
    df["disease"] = df["category"].apply(lambda x: 0 if x == "no_disease" else 1)

    # Drop original label
    df = df.drop(columns=["category"])

    return df

df = load_data()


# -----------------------------
# TRAIN MODEL ON THE FLY
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["disease"])
    y = df["disease"]

    # Store feature names
    feature_names = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, scaler, feature_names

model, scaler, feature_names = train_model(df)

st.success("Model trained successfully using Random Forest!")

# -----------------------------
# USER INPUT FORM
# -----------------------------
st.header("Enter Patient Values")

inputs = {}

for col in feature_names:
    # use median as default
    default_val = float(df[col].median())
    inputs[col] = st.number_input(col, value=default_val)

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# Scale
input_scaled = scaler.transform(input_df)

# -----------------------------
# PREDICT
# -----------------------------
if st.button("Predict Disease Status"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🔴 Disease Likely — Probability: {prob*100:.2f}%")
    else:
        st.success(f"🟢 No Disease — Probability: {prob*100:.2f}%")

    st.subheader("Entered Values")
    st.write(input_df)
