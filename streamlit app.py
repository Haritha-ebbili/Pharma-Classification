#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Liver Disease — 5 Models + GridSearch + Save Best", layout="wide")
st.title("🩺 Liver Disease Classification — 5 Models with GridSearchCV & Save Best Model")

# ----------------------------
# Helper: read CSV (comma or semicolon)
# ----------------------------
def smart_read_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file)
        # if whole file ended up in single column, try semicolon
        if df.shape[1] == 1 and df.columns[0].count(';') > 0:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';')
    # normalize column names
    df.columns = df.columns.astype(str).str.strip()
    return df

uploaded_file = st.file_uploader("Upload Liver_data.csv (CSV)", type=["csv"], accept_multiple_files=False)

if not uploaded_file:
    st.info("Please upload your CSV file (e.g. Liver_data.csv). The app will auto-detect comma/semicolon separators.")
    st.stop()

# Load dataframe
df = smart_read_csv(uploaded_file)
st.write("### Preview of uploaded data")
st.dataframe(df.head())

# ----------------------------
# Target column - user specified 'category'
# ----------------------------
desired_target = "category"
# find a column matching desired_target (case-insensitive, stripped)
target_col = None
for c in df.columns:
    if c.strip().lower() == desired_target.lower():
        target_col = c
        break

if target_col is None:
    st.error(f"❌ Could not find target column named '{desired_target}'. Available columns: {list(df.columns)}")
    st.stop()

st.success(f"✔ Target column detected: `{target_col}`")

# ----------------------------
# Preprocessing
# ----------------------------
# 1) Drop rows with all-NA
df = df.dropna(how="all").reset_index(drop=True)

# 2) Separate X and y
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# 3) Strip column names
X.columns = X.columns.str.strip()

# 4) Convert obvious numeric columns to numeric (coerce invalid -> NaN)
for col in X.columns:
    X[col] = pd.to_numeric(X[col].astype(str).str.replace(',','.').str.strip(), errors='coerce')

# 5) Identify categorical columns (those that are still non-numeric)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# 6) If categorical columns exist, one-hot encode them
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 7) After encoding: ensure numeric columns list is updated
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# 8) Fill remaining NaNs in numeric columns with median (safe)
if len(numeric_cols) > 0:
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
else:
    # If no numeric columns remain (unlikely), drop rows with NaN
    X = X.fillna(0)

st.write(f"Features: {X.shape[1]} columns after encoding. Numeric columns: {len(numeric_cols)}")

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y if len(np.unique(y))>1 else None, random_state=42
)

# ----------------------------
# Safe scaling: scale numeric cols only
# ----------------------------
scaler = StandardScaler()
if numeric_cols:
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
else:
    # no numeric columns: keep copies
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

# ----------------------------
# Determine CV folds safely using smallest class size
# ----------------------------
if len(np.unique(y_train)) > 1:
    min_class_count = int(y_train.value_counts().min())
    cv_folds = min(5, min_class_count)
    if cv_folds < 2:
        st.error("❌ Not enough samples per class to perform cross-validation (need at least 2).")
        st.stop()
else:
    # single-class target - cannot train classifiers
    st.error("❌ The target column contains only one class. Classification cannot proceed.")
    st.stop()

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
st.write(f"Using StratifiedKFold with n_splits={cv_folds} for GridSearchCV.")

# ----------------------------
# Models and parameter grids
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

params = {
    "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["liblinear"]},
    "KNN": {"n_neighbors": [3,5,7], "weights": ["uniform","distance"]},
    "SVM": {"C": [0.1,1,10], "kernel": ["linear","rbf"], "gamma": ["scale","auto"]},
    "Random Forest": {"n_estimators": [50,100], "max_depth": [5, 10, None]},
    "Decision Tree": {"max_depth": [3,5,10,None], "criterion": ["gini","entropy"]}
}

# ----------------------------
# Train with GridSearchCV and evaluate
# ----------------------------
st.subheader("🔍 Training models with GridSearchCV")
results = []
trained_models = {}
best_model_name = None
best_model_obj = None
best_test_acc = -1.0

for name, model in models.items():
    st.write(f"Training **{name}**...")
    grid = GridSearchCV(estimator=model, param_grid=params[name], cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best = grid.best_estimator_
    trained_models[name] = best

    # Predictions
    y_pred = best.predict(X_test_scaled)

    # Choose metric averaging depending on binary/multiclass
    if len(np.unique(y)) == 2:
        avg = "binary"
    else:
        avg = "weighted"

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

    # Try ROC-AUC if possible (binary)
    roc_auc = None
    try:
        if len(np.unique(y)) == 2 and hasattr(best, "predict_proba"):
            probs = best.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, probs)
    except Exception:
        roc_auc = None

    results.append({
        "Model": name,
        "Test Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1": f1,
        "CV Mean Accuracy": grid.best_score_,
        "ROC-AUC": roc_auc
    })

    # track best by Test Accuracy
    if acc > best_test_acc:
        best_test_acc = acc
        best_model_name = name
        best_model_obj = best

# ----------------------------
# Results display
# ----------------------------
results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)
st.subheader("📊 Model Comparison (sorted by Test Accuracy)")
st.dataframe(results_df.style.format({"Test Accuracy":"{:.4f}", "Precision":"{:.4f}", "Recall":"{:.4f}", "F1":"{:.4f}", "CV Mean Accuracy":"{:.4f}"}))

st.info(f"🏆 Best model by test accuracy: **{best_model_name}** (Test Accuracy = {best_test_acc:.4f})")

# ----------------------------
# Confusion matrix and ROC for best model
# ----------------------------
st.subheader(f"📈 Detailed evaluation — Best model: {best_model_name}")

best = best_model_obj
y_pred_best = best.predict(X_test_scaled)

fig_cm = None
try:
    cm = confusion_matrix(y_test, y_pred_best)
    import matplotlib.pyplot as plt
    fig_cm, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    st.pyplot(fig_cm)
except Exception as e:
    st.write("Could not plot confusion matrix:", e)

# ROC (binary only)
if len(np.unique(y)) == 2 and hasattr(best, "predict_proba"):
    try:
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_estimator(best, X_test_scaled, y_test, ax=ax_roc)
        st.pyplot(fig_roc)
    except Exception as e:
        st.write("Could not plot ROC curve:", e)

# ----------------------------
# Save best model + scaler together
# ----------------------------
st.subheader("💾 Download best model (pickle)")

if best_model_obj is not None:
    package = {
        "model_name": best_model_name,
        "model": best_model_obj,
        "scaler": scaler,
        "feature_columns": X.columns.tolist()
    }
    bytes_obj = pickle.dumps(package)

    st.download_button(
        label=f"⬇ Download `{best_model_name}` as best_model.pkl",
        data=bytes_obj,
        file_name="best_model.pkl",
        mime="application/octet-stream"
    )
else:
    st.write("No trained model available to download.")
