# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Liver Disease - Random Forest", layout="wide")
st.title("Liver Disease Prediction")

st.markdown(
    """
Upload your `Liver_data.csv` (semicolon-separated) and the app will:
- clean column names,
- train a Random Forest on `category`,
- show metrics and feature importance,
- let you make single-sample predictions.
"""
)

uploaded = st.file_uploader("Upload Liver_data.csv (CSV, uses `;` separator)", type=["csv"])

def clean_and_assert(df: pd.DataFrame) -> pd.DataFrame:
    # Trim column names and strip trailing spaces
    df.columns = [c.strip() for c in df.columns]
    required = [
        'category', 'age', 'sex', 'albumin', 'alkaline_phosphatase',
        'alanine_aminotransferase', 'aspartate_aminotransferase',
        'bilirubin', 'cholinesterase', 'cholesterol', 'creatinina',
        'gamma_glutamyl_transferase', 'protein'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[required]

@st.cache_data(show_spinner=False)
def train_model(df: pd.DataFrame, random_state: int = 42):
    # Preprocess
    df = df.copy()
    # map sex
    df['sex'] = df['sex'].astype(str).str.lower().map({'m':1, 'male':1, 'f':0, 'female':0})
    if df['sex'].isnull().any():
        # if mapping produced NaNs, try numeric cast fallback
        df['sex'] = pd.to_numeric(df['sex'], errors='coerce').fillna(0).astype(int)
    # encode target
    le = LabelEncoder()
    y = le.fit_transform(df['category'].astype(str))
    X = df.drop(columns=['category'])
    # Keep feature order consistent
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # metrics
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf = confusion_matrix(y_test, y_pred)
    # roc-auc if binary and model supports predict_proba
    roc_auc = None
    try:
        if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_scaled)[:, 1]
            roc_auc = float(roc_auc_score(y_test, probs))
    except Exception:
        roc_auc = None

    # feature importance
    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        fi_series = pd.Series(fi, index=feature_names).sort_values(ascending=False)
    else:
        fi_series = pd.Series([], dtype=float)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": conf,
        "roc_auc": roc_auc,
        "feature_importances": fi_series
    }

if uploaded:
    try:
        # use semicolon separator as dataset uses ;
        df = pd.read_csv(uploaded, sep=';').dropna()
        df = clean_and_assert(df)
    except Exception as e:
        st.error(f"Failed to read/validate CSV: {e}")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    with st.spinner("Training Random Forest..."):
        results = train_model(df)

    st.success("Training completed!")

    # Show metrics
    st.subheader("Model performance on test split")
    st.write(f"Accuracy: **{results['accuracy']:.4f}**")
    if results['roc_auc'] is not None:
        st.write(f"ROC-AUC (binary): **{results['roc_auc']:.4f}**")
    st.write("Classification report:")
    st.dataframe(pd.DataFrame(results['report']).transpose(), use_container_width=True)

    st.write("Confusion matrix:")
    cm = results['confusion_matrix']
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center', color='black')
    st.pyplot(fig)

    # Feature importance
    if not results['feature_importances'].empty:
        st.subheader("Feature importances")
        st.bar_chart(results['feature_importances'])

    # Single-sample prediction form (fields in exact dataset order)
    st.subheader("Single sample prediction")
    cols = st.columns(2)
    with cols[0]:
        age = st.number_input("age", value=float(df['age'].median()))
        sex = st.selectbox("sex", options=["m","f"], index=0)
        albumin = st.number_input("albumin", value=float(df['albumin'].median()))
        alkaline_phosphatase = st.number_input("alkaline_phosphatase", value=float(df['alkaline_phosphatase'].median()))
        alanine_aminotransferase = st.number_input("alanine_aminotransferase", value=float(df['alanine_aminotransferase'].median()))
        aspartate_aminotransferase = st.number_input("aspartate_aminotransferase", value=float(df['aspartate_aminotransferase'].median()))
    with cols[1]:
        bilirubin = st.number_input("bilirubin", value=float(df['bilirubin'].median()))
        cholinesterase = st.number_input("cholinesterase", value=float(df['cholinesterase'].median()))
        cholesterol = st.number_input("cholesterol", value=float(df['cholesterol'].median()))
        creatinina = st.number_input("creatinina", value=float(df['creatinina'].median()))
        gamma_glutamyl_transferase = st.number_input("gamma_glutamyl_transferase", value=float(df['gamma_glutamyl_transferase'].median()))
        protein = st.number_input("protein", value=float(df['protein'].median()))

    if st.button("Predict"):
        # build sample in same order
        sex_val = 1 if str(sex).lower().startswith('m') else 0
        sample = pd.DataFrame([{
            'age': age,
            'sex': sex_val,
            'albumin': albumin,
            'alkaline_phosphatase': alkaline_phosphatase,
            'alanine_aminotransferase': alanine_aminotransferase,
            'aspartate_aminotransferase': aspartate_aminotransferase,
            'bilirubin': bilirubin,
            'cholinesterase': cholinesterase,
            'cholesterol': cholesterol,
            'creatinina': creatinina,
            'gamma_glutamyl_transferase': gamma_glutamyl_transferase,
            'protein': protein
        }])

        # scale with trained scaler
        sample_scaled = results['scaler'].transform(sample[results['feature_names']])
        pred = results['model'].predict(sample_scaled)[0]
        pred_label = results['label_encoder'].inverse_transform([pred])[0]
        prob = None
        if hasattr(results['model'], "predict_proba") and len(results['label_encoder'].classes_) == 2:
            prob = results['model'].predict_proba(sample_scaled)[0][1]

        st.markdown("### Prediction")
        if prob is not None:
            st.write(f"**Predicted category:** {pred_label} (probability of positive class: {prob:.3f})")
        else:
            st.write(f"**Predicted category:** {pred_label}")

    st.info("Model is trained in-app on the uploaded CSV. No external .pkl files required.")
else:
    st.info("Upload the semicolon-separated `Liver_data.csv` to train the Random Forest model.")
