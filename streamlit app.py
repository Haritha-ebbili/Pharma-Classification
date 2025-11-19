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

st.set_page_config(page_title="Liver Disease Classification", layout="wide")
st.title("🩺 Liver Disease Classification — 5 Models + GridSearch + Save Best")

# -------------------------------
# Helper: Read CSV smartly
# -------------------------------

def smart_read_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file)
        # If the file reads into just one wide column containing semicolons, try semicolon sep
        if df.shape[1] == 1 and ';' in df.columns[0]:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';')
    df.columns = df.columns.astype(str).str.strip()
    return df

uploaded_file = st.file_uploader("Upload Liver_data.csv", type=["csv"])
if not uploaded_file:
    st.info("Please upload the dataset (CSV).")
    st.stop()

df = smart_read_csv(uploaded_file)
st.write("### Dataset Preview", df.head())
st.write("Columns:", df.columns.tolist())

# -------------------------------
# Target Column = "category"
# -------------------------------
target_name = "category"
target_col = None
for c in df.columns:
    if c.strip().lower() == target_name.lower():
        target_col = c
        break

if target_col is None:
    st.error(f"❌ Could not find target column named '{target_name}'. Available columns: {list(df.columns)}")
    st.stop()

st.success(f"✔ Target column detected: `{target_col}`")

# -------------------------------
# Preprocessing
# -------------------------------

# Drop completely empty rows
df = df.dropna(how="all").reset_index(drop=True)

# Split X and y
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Strip whitespace from column names
X.columns = X.columns.str.strip()

# Convert numeric-like columns to numeric
for col in X.columns:
    # Replace commas with dots, strip spaces
    X[col] = X[col].astype(str).str.replace(',', '.', regex=False).str.strip()
    # Try convert to numeric; if fail, leave as is
    X[col] = pd.to_numeric(X[col], errors='ignore')

# Detect numeric vs categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# One-hot encode categorical columns if any
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Update numeric_cols after encoding
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Impute missing numeric values with median
if numeric_cols:
    X[numeric_cols] = X[numeric_cols].apply(lambda col: col.fillna(col.median()))

# Final safety: fill any remaining NaN (if any) with zero
X = X.fillna(0)

st.write(f"After preprocessing: {X.shape[0]} rows, {X.shape[1]} features.")
st.write(f"Numeric features: {numeric_cols}")

# -------------------------------
# Split data
# -------------------------------
if len(np.unique(y)) < 2:
    st.error("❌ The target column has only one class. Cannot perform classification.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# Scale numeric features only
# -------------------------------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if numeric_cols:
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# -------------------------------
# Decide CV folds based on class balance
# -------------------------------
min_class_count = int(y_train.value_counts().min())
cv_folds = min(5, min_class_count)
if cv_folds < 2:
    st.error("❌ Not enough samples in smallest class for cross-validation.")
    st.stop()

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
st.write(f"Using StratifiedKFold with n_splits = {cv_folds}")

# -------------------------------
# Models + Hyperparams
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

params = {
    "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["liblinear"]},
    "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    "Decision Tree": {"max_depth": [3, 5, 10, None], "criterion": ["gini", "entropy"]},
}

# -------------------------------
# Training with GridSearchCV
# -------------------------------
st.subheader("🔍 Training Models with GridSearchCV")

results = []
trained_models = {}
best_model = None
best_acc = -1
best_name = None

for name, model in models.items():
    st.write(f"Training **{name}** …")
    grid = GridSearchCV(estimator=model, param_grid=params[name], cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_est = grid.best_estimator_
    trained_models[name] = best_est

    # Predict on test
    y_pred = best_est.predict(X_test_scaled)

    # Determine average for precision/recall/f1
    if len(np.unique(y)) == 2:
        avg = "binary"
    else:
        avg = "weighted"

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

    # ROC-AUC if binary
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
        "ROC-AUC": roc_auc,
    })

    # Track best
    if acc > best_acc:
        best_acc = acc
        best_model = best_est
        best_name = name

# -------------------------------
# Display results
# -------------------------------
results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)
st.subheader("📊 Model Comparison")
st.dataframe(
    results_df.style.format({
        "Test Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "CV Mean Accuracy": "{:.4f}",
        "ROC-AUC": "{:.4f}"
    })
)

st.success(f"🏆 Best model: **{best_name}** with Test Accuracy = {best_acc:.4f}")

# -------------------------------
# Show Confusion Matrix & ROC
# -------------------------------
st.subheader(f"🔎 Evaluation of Best Model: {best_name}")
y_pred_best = best_model.predict(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
import matplotlib.pyplot as plt
fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# ROC curve
if len(np.unique(y)) == 2 and hasattr(best_model, "predict_proba"):
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test, ax=ax_roc)
    st.pyplot(fig_roc)

# -------------------------------
# Save model + scaler + feature list
# -------------------------------
st.subheader("💾 Download Best Model (Pickle)")

package = {
    "model_name": best_name,
    "model": best_model,
    "scaler": scaler,
    "feature_columns": X.columns.tolist()
}

bytes_out = pickle.dumps(package)
st.download_button(
    label=f"⬇ Download `{best_name}` + scaler",
    data=bytes_out,
    file_name="best_model_package.pkl",
    mime="application/octet-stream"
)
