#!/usr/bin/env python
import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
st.title("🩺 Liver Disease Classification with GridSearchCV ")

# -------------------------
# Helpers
# -------------------------
def smart_read_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file)
        # If read into 1 column (likely semicolon delimited), try semicolon
        if df.shape[1] == 1 and ';' in df.columns[0]:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';')
    df.columns = df.columns.astype(str).str.strip()
    return df

def safe_metric(v, digits=4):
    if v is None:
        return "N/A"
    try:
        return f"{v:.{digits}f}"
    except Exception:
        return str(v)

# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("Upload Liver_data.csv (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV (comma or semicolon separated).")
    st.stop()

df = smart_read_csv(uploaded_file)
st.write("### Dataset preview")
st.dataframe(df.head())
st.write("Columns:", df.columns.tolist())

# -------------------------
# Target detection: use 'Category' (case-insensitive)
# -------------------------
desired_target = "Category"
target_col = None
for c in df.columns:
    if c.strip().lower() == desired_target.lower():
        target_col = c
        break

if target_col is None:
    st.error(f"❌ Could not find a column named '{desired_target}' (case-insensitive). Available columns: {list(df.columns)}")
    st.stop()

st.success(f"✔ Target column detected: `{target_col}`")

# -------------------------
# Preprocessing
# -------------------------
# Drop fully-empty rows
df = df.dropna(how="all").reset_index(drop=True)

# Split features and target
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Normalize column names
X.columns = X.columns.astype(str).str.strip()

# Convert numeric-like strings -> numeric where possible
for col in X.columns:
    # allow comma as decimal separator, strip whitespace
    X[col] = X[col].astype(str).str.replace(",", ".", regex=False).str.strip()
    # attempt numeric conversion, leave strings as-is if not numeric
    X[col] = pd.to_numeric(X[col], errors="ignore")

# Identify numeric vs categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

st.write(f"Detected numeric columns: {numeric_cols}")
st.write(f"Detected categorical columns: {categorical_cols}")

# One-hot encode only if categorical columns exist
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    st.write(f"After encoding, feature count: {X.shape[1]}")

# Update numeric columns after encoding
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Impute numeric NaNs with median
if numeric_cols:
    X[numeric_cols] = X[numeric_cols].apply(lambda col: col.fillna(col.median()))
# Final safety fill (any remaining NaN -> 0)
X = X.fillna(0)

st.write(f"Final features shape: {X.shape}")

# -------------------------
# Validate target/classes
# -------------------------
if len(np.unique(y)) < 2:
    st.error("❌ The target column contains only one class. Cannot train classifiers.")
    st.stop()

# -------------------------
# Train-test split (stratify)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------
# Scale numeric features only
# -------------------------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
if numeric_cols:
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# -------------------------
# CV folds safe selection based on min class count
# -------------------------
min_class_count = int(y_train.value_counts().min())
cv_folds = min(5, min_class_count)
if cv_folds < 2:
    st.error("❌ Not enough samples in the smallest class to perform cross-validation (need at least 2).")
    st.stop()

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
st.write(f"Using StratifiedKFold with n_splits={cv_folds}")

# -------------------------
# Models and hyperparameter grids
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

params = {
    "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["liblinear"]},
    "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    "Decision Tree": {"max_depth": [3, 5, 10, None], "criterion": ["gini", "entropy"]}
}

# -------------------------
# Train with GridSearchCV
# -------------------------
st.subheader("🔍 Training models with GridSearchCV (may take a few minutes)")
results = []
trained_models = {}
best_model = None
best_acc = -1.0
best_name = None

for name, model in models.items():
    st.write(f"Training **{name}** …")
    grid = GridSearchCV(estimator=model, param_grid=params[name], cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_est = grid.best_estimator_
    trained_models[name] = best_est

    # test predictions
    y_pred = best_est.predict(X_test_scaled)

    # choose averaging for metrics
    avg = "binary" if len(np.unique(y)) == 2 else "weighted"

    acc = accuracy_score(y_test, y_pred)
    try:
        pre = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
    except Exception:
        pre = rec = f1 = None

    # roc auc if binary
    roc_auc = None
    if len(np.unique(y)) == 2 and hasattr(best_est, "predict_proba"):
        try:
            probs = best_est.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, probs)
        except Exception:
            roc_auc = None

    results.append({
        "Model": name,
        "Test Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1 Score": f1,
        "CV Mean Accuracy": grid.best_score_,
        "ROC-AUC": roc_auc
    })

    if acc > best_acc:
        best_acc = acc
        best_model = best_est
        best_name = name

# -------------------------
# Results table
# -------------------------
results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)
st.subheader("📊 Model Comparison")
st.dataframe(results_df.style.format({
    "Test Accuracy": "{:.4f}",
    "Precision": lambda v: safe_metric(v),
    "Recall": lambda v: safe_metric(v),
    "F1 Score": lambda v: safe_metric(v),
    "CV Mean Accuracy": "{:.4f}",
    "ROC-AUC": lambda v: safe_metric(v)
}))

st.success(f"🏆 Best model by Test Accuracy: **{best_name}** (Test Accuracy = {safe_metric(best_acc)})")

# -------------------------
# Detailed eval of best model
# -------------------------
if best_model is not None:
    st.subheader(f"🔎 Detailed evaluation — Best model: {best_name}")

    y_pred_best = best_model.predict(X_test_scaled)

    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred_best)
        import matplotlib.pyplot as plt
        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
    except Exception as e:
        st.write("Could not plot confusion matrix:", e)

    # ROC curve (binary & supports predict_proba)
    if len(np.unique(y)) == 2 and hasattr(best_model, "predict_proba"):
        try:
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test, ax=ax_roc)
            st.pyplot(fig_roc)
        except Exception as e:
            st.write("Could not plot ROC curve:", e)

# -------------------------
# Download best model + scaler + feature list
# -------------------------
st.subheader("💾 Download best model (pickle)")

package = {
    "model_name": best_name,
    "model": best_model,
    "scaler": scaler,
    "feature_columns": X.columns.tolist()
}

bytes_out = pickle.dumps(package)
st.download_button(
    label=f"⬇ Download `{best_name}` + scaler as best_model_package.pkl",
    data=bytes_out,
    file_name="best_model_package.pkl",
    mime="application/octet-stream"
)
