import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Disease Detection", layout="wide")

# ---------------------------
# Helper: load & clean data
# ---------------------------
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

# ---------------------------
# Session state defaults
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "reports" not in st.session_state:
    st.session_state.reports = []

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Reports"])

# ---------------------------
# Home page
# ---------------------------
if page == "Home":
    st.title("🩺 Liver Disease Detection — Home")
    st.markdown(
        "Upload a CSV (Liver_data.csv) containing the columns used by the model (including a 'category' column). (Liver_data.csv) containing the columns used by the model (including a 'category' column)."
 
        "After upload, go to Prediction to enter patient values and get a score + stage prediction."
    )

    uploaded_file = st.file_uploader("📤 Upload Liver_data.csv", type=["csv"]) 
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.success("Dataset uploaded and parsed successfully ✔️")
            st.write("**Preview:**")
            st.dataframe(df.head())

            # Train model immediately and store
            X = df.drop("category", axis=1)
            y = df["category"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
            model.fit(X_train_scaled, y_train)

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.info("Model trained on uploaded data and ready to use in Prediction page.")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

# ---------------------------
# Prediction page
# ---------------------------
elif page == "Prediction":
    st.title("🔍 Prediction & Real-time Liver Health Score")

    if st.session_state.df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        df = st.session_state.df
        model = st.session_state.model
        scaler = st.session_state.scaler

        # Define healthy ranges (adjust as needed)
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

        st.markdown("Enter patient values below.")

        # Input collection without sliders or color feedback
        inputs = {}
        numeric_cols = [c for c in df.drop("category", axis=1).columns]

        cols = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with cols[i % 2]:
                if col == "sex":
                    val = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
                    inputs[col] = val
                else:
                    default_val = float(df[col].median())
                    val = st.number_input(col, value=default_val)
                    inputs[col] = val("category", axis=1).columns

        cols = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with cols[i % 2]:
                if col == "sex":
                    val = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
                    inputs[col] = val
                    abnormal_flags[col] = False
                else:
                    col_min = float(df[col].min())
                    col_max = float(df[col].max())
                    default_val = float(df[col].median())
                    low, high = healthy_ranges.get(col, (col_min, col_max))

                    val = st.slider(col, min_value=col_min, max_value=col_max, value=default_val, step=(col_max-col_min)/100)
                    inputs[col] = val

                    if col in healthy_ranges and not (low <= val <= high):
                        st.markdown(f"<div style='color:red; font-weight:700;'>🔴 Outside healthy range: {low} - {high}</div>", unsafe_allow_html=True)
                        abnormal_flags[col] = True
                    else:
                        st.markdown(f"<div style='color:green; font-weight:600;'>🟢 Within healthy range</div>", unsafe_allow_html=True)
                        abnormal_flags[col] = False

        # Real-time liver health score calculation
        def compute_health_score(inputs, healthy_ranges):
            scores = []
            for col, val in inputs.items():
                if col == "sex":
                    continue
                if col in healthy_ranges:
                    low, high = healthy_ranges[col]
                else:
                    # If no defined healthy range, assume dataset median +- 25%
                    series = df[col]
                    med = float(series.median())
                    span = abs(series.max() - series.min()) or med or 1.0
                    low, high = med - 0.25 * span, med + 0.25 * span

                center = (low + high) / 2.0
                half = (high - low) / 2.0 if (high - low) != 0 else 1.0
                # If within range, full score for this feature
                if low <= val <= high:
                    scores.append(1.0)
                else:
                    # how far outside (normalized): cap at 1.0
                    dist = abs(val - center)
                    outside_amount = max(0.0, (dist - half) / (half if half !=0 else 1.0))
                    # reduce score proportionally (more outside -> lower score)
                    score = max(0.0, 1.0 - min(outside_amount, 1.0))
                    scores.append(score)
            # average across features
            if len(scores) == 0:
                return 100
            avg = sum(scores) / len(scores)
            return int(avg * 100)

        health_score = compute_health_score(inputs, healthy_ranges)

        # Visualize score
        st.subheader("Overall Liver Health Score")
        st.metric("Health Score (0 = worst, 100 = best)", f"{health_score}/100")
        st.progress(health_score / 100)

        # Predict button
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)

        if st.button("🔍 Predict Stage and Save Report"):
            pred = model.predict(input_scaled)[0]
            category_map = {0: "No Disease", 1: "Suspect Disease", 2: "Hepatitis", 3: "Fibrosis", 4: "Cirrhosis"}
            severity_map = {0: "None", 1: "Mild", 2: "Moderate", 3: "High", 4: "Severe"}
            disease = category_map[pred]
            severity = severity_map[pred]

            if pred == 0:
                st.success(f"🟢 Prediction: {disease} — {severity}")
            else:
                st.error(f"🔴 Prediction: {disease} — {severity}")

            stage_info = {
                0: ("No Disease", "No abnormalities detected."),
                1: ("Suspect Disease", "Some irregularities found. Monitoring advised."),
                2: ("Hepatitis", "Inflammation detected."),
                3: ("Fibrosis", "Scarring present."),
                4: ("Cirrhosis", "Severe irreversible damage.")
            }
            suggestions = {
                0: "Maintain healthy diet and lifestyle.",
                1: "Avoid alcohol, hydrate well, repeat tests in a few weeks.",
                2: "Consult a hepatologist. Follow prescribed treatment.",
                3: "Liver-supportive medicines and lifestyle management needed.",
                4: "Immediate medical care. Possible transplantation evaluation."
            }

            st.info(stage_info[pred][1])
            st.warning(suggestions[pred])

            # Save report to session_state
            report = inputs.copy()
            report.update({"predicted_stage": disease, "severity": severity, "health_score": health_score})
            st.session_state.reports.append(report)
            st.success("Report saved — view it on the Reports page.")

# ---------------------------
# Reports page
# ---------------------------
elif page == "Reports":
    st.title("📄 Reports")
    if len(st.session_state.reports) == 0:
        st.info("No reports saved yet. Run a prediction and save a report on the Prediction page.")
    else:
        reports_df = pd.DataFrame(st.session_state.reports)
        st.dataframe(reports_df)

        csv = reports_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Reports CSV", csv, "liver_reports.csv", "text/csv")

        st.markdown("Use this page to review past predictions and export them for clinical records.")
