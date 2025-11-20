import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Disease Detection", layout="wide")

st.title("🩺 Liver Disease Detection System")
st.markdown("""
Upload your **Liver_data.csv** file to begin. Once uploaded, the app will allow you to input patient values using sliders, with automatic color indicators (🔴 Red = abnormal, 🟢 Green = normal).
""")

# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload Liver_data.csv", type=["csv"])

if uploaded_file:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df["sex"] = df["sex"].str.lower().map({"m": 1, "male": 1, "f": 0, "female": 0})

        numeric_cols = df.columns.drop("category")
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        df["category"] = df["category"].str.lower().map({
            "no_disease": 0,
            "suspect_disease": 1,
            "hepatitis": 2,
            "fibrosis": 3,
            "cirrhosis": 4
        })

        df = df.dropna()
        return df

    df = load_data(uploaded_file)
    st.success("Dataset uploaded successfully ✔️")

    # ---------------------------------------------------------
    # MODEL TRAINING
    # ---------------------------------------------------------
    X = df.drop("category", axis=1)
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # ---------------------------------------------------------
    # HEALTHY RANGES + SLIDERS
    # ---------------------------------------------------------
    st.header("🧍 Patient Input Values")

    healthy_ranges = {
        "age": (18, 80),
        "total_bilirubin": (0.1, 1.2),
        "direct_bilirubin": (0.0, 0.3),
        "alkphos": (44, 147),
        "sgpt": (7, 56),
        "sgot": (5, 40),
        "total_proteins": (6.0, 8.3),
        "albumin": (3.5, 5.0),
        "ag_ratio": (1.0, 2.5)
    }

    inputs = {}
    for col in X.columns:
        if col == "sex":
            inputs[col] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        else:
            low, high = healthy_ranges.get(col, (df[col].min(), df[col].max()))
            default_val = float(df[col].median())

            block = st.container()
            value = block.slider(col, float(df[col].min()), float(df[col].max()), default_val)
            inputs[col] = value

            if col in healthy_ranges:
                if not (low <= value <= high):
                    block.markdown(f"<span style='color:red; font-weight:bold;'>🔴 Outside healthy range: {low} - {high}</span>", unsafe_allow_html=True)
                else:
                    block.markdown(f"<span style='color:green; font-weight:bold;'>🟢 Within healthy range</span>", unsafe_allow_html=True)

    input_data = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_data)

    # ---------------------------------------------------------
    # STAGE INFO
    # ---------------------------------------------------------
    stage_info = {
        0: ("No Disease", "No abnormalities detected."),
        1: ("Suspect Disease", "Some irregularities found. Monitoring advised."),
        2: ("Hepatitis", "Inflammation detected, often due to infection or toxins."),
        3: ("Fibrosis", "Scarring forming in liver tissues."),
        4: ("Cirrhosis", "Severe irreversible liver damage.")
    }

    suggestions = {
        0: "Maintain healthy diet and lifestyle.",
        1: "Avoid alcohol, hydrate well, repeat tests in a few weeks.",
        2: "Consult a hepatologist. Follow prescribed antiviral/inflammatory treatment.",
        3: "Liver-supportive medicines and lifestyle management needed.",
        4: "Immediate medical care. Possible need for transplantation evaluation."
    }

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------
    if st.button("🔍 Predict Stage", use_container_width=True):
        pred = model.predict(input_scaled)[0]
        disease, desc = stage_info[pred]
        advice = suggestions[pred]

        if pred == 0:
            st.success(f"🟢 Prediction: **{disease}**")
        else:
            st.error(f"🔴 Prediction: **{disease}**")

        st.info(desc)
        st.warning(f"### 💡 Recommendation: {advice}")

        st.markdown("""
        ### 🧬 Liver Disease Stages
        1. **No Disease** – Normal liver  
        2. **Suspect** – Early warning  
        3. **Hepatitis** – Liver inflammation  
        4. **Fibrosis** – Scar tissue formation  
        5. **Cirrhosis** – Advanced damage
        """)
