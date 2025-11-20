import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Disease Detection", layout="wide")

st.title("🩺 Liver Disease Detection System")
st.markdown("""
This tool helps detect potential liver disease stages based on medical data.  
Please upload your **Liver_data.csv** file to begin.
""")

# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload Liver_data.csv", type=["csv"], help="Upload the dataset to proceed")

if uploaded_file:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, sep=None, engine="python")

        df.columns = df.columns.str.strip()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        df["sex"] = df["sex"].str.lower().map({
            "m": 1, "male": 1,
            "f": 0, "female": 0
        })

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

    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())

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
    st.header("🧍 Enter Patient Values")
    defaults = X.median()

    # Define healthy ranges (example ranges — adjust based on your dataset)
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
    column_alerts = []

    for col in X.columns:
        if col == "sex":
            inputs[col] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        else:
            block = st.container()
            val = block.number_input(col, value=float(defaults[col]), key=col)
            inputs[col] = val

            if col in healthy_ranges:
                low, high = healthy_ranges[col]
                if not (low <= val <= high):
                    block.markdown(f"<span style='color:red; font-weight:bold;'>Outside healthy range: {low} - {high}</span>", unsafe_allow_html=True)
                    column_alerts.append(f"⚠️ **{col}** is outside the healthy range ({low} - {high}).")
                else:
                    block.markdown(f"<span style='color:green; font-weight:600;'>Within healthy range ✔️</span>", unsafe_allow_html=True)

    # Show alerts and disease indication per feature
    if column_alerts:
        st.warning("### ⚠️ Range Alerts Detected — Possible Signs of Liver Disease")
        for alert in column_alerts:
            st.write(alert)
        st.info("Values outside the healthy range can indicate potential liver abnormalities.")
    if column_alerts:
        st.warning("### ⚠️ Range Alerts Detected")
        for alert in column_alerts:
            st.write(alert)

    input_data = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_data)

    # ---------------------------------------------------------
    # STAGE DESCRIPTIONS
    # ---------------------------------------------------------
    stage_info = {
        0: ("No Disease", "No liver abnormalities detected."),
        1: ("Suspect Disease", "Early warning signs. Further monitoring advised."),
        2: ("Hepatitis", "Liver inflammation detected due to infection or toxins."),
        3: ("Fibrosis", "Liver scarring present but potentially reversible."),
        4: ("Cirrhosis", "Severe liver damage. Immediate medical care needed.")
    }

    suggestions = {
        0: "Maintain a healthy lifestyle, regular exercise, avoid alcohol excess.",
        1: "Get regular checkups, maintain hydration, avoid alcohol and fatty foods.",
        2: "Consult a hepatologist. Use antiviral or anti-inflammatory medication if prescribed.",
        3: "Start liver‑supportive treatment. Control diabetes, lose weight, avoid alcohol.",
        4: "Emergency medical intervention required. May need transplantation evaluation."
    }

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------
    if st.button("🔍 Predict Disease Stage", use_container_width=True):
        pred = model.predict(input_scaled)[0]

        disease, description = stage_info[pred]
        advice = suggestions[pred]

        # Highlight prediction
        if pred == 0:
            st.success(f"🟢 Prediction: **{disease}**")
        else:
            st.error(f"🔴 Prediction: **{disease}**")

        st.markdown(f"### 📌 Stage Explanation")
        st.info(description)

        st.markdown(f"### 💡 Medical Suggestion")
        st.warning(advice)

        st.markdown("### 🧬 Liver Disease Stages Overview")
        st.write("""
        1️⃣ **No Disease** – Healthy liver  
2️⃣ **Suspect Disease** – Early abnormalities  
3️⃣ **Hepatitis** – Inflammation of liver  
4️⃣ **Fibrosis** – Scar tissue formation  
5️⃣ **Cirrhosis** – Advanced irreversible damage
        """)
