#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------
# STREAMLIT CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("🚀 ML Model Deployment – All Models (with GridSearchCV)")

# ----------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if target:
        X = df.drop(target, axis=1)
        y = df[target]

        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale numeric features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.subheader("🔍 Train Models (with Hyperparameter Tuning)")
        model_scores = []

        if st.button("Run All Models"):

            # ----------------------------------------------------
            # Helper function for safe ROC-AUC
            # ----------------------------------------------------
            def safe_auc(y_true, y_prob):
                try:
                    return roc_auc_score(y_true, y_prob)
                except:
                    return None

            # ----------------------------------------------------
            # 1️⃣ Logistic Regression
            # ----------------------------------------------------
            st.write("### Logistic Regression")
            log_params = {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}

            log_reg = GridSearchCV(LogisticRegression(max_iter=2000), log_params,
                                   cv=5, scoring="accuracy")
            log_reg.fit(X_train_scaled, y_train)

            y_pred = log_reg.predict(X_test_scaled)
            y_prob = log_reg.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None

            model_scores.append([
                "Logistic Regression",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='weighted'),
                recall_score(y_test, y_pred, average='weighted'),
                f1_score(y_test, y_pred, average='weighted'),
                safe_auc(y_test, y_prob) if y_prob is not None else "N/A"
            ])

            joblib.dump(log_reg.best_estimator_, "logistic_regression.pkl")

            # ----------------------------------------------------
            # 2️⃣ Random Forest
            # ----------------------------------------------------
            st.write("### Random Forest")
            rf_params = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}

            rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring="accuracy")
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

            model_scores.append([
                "Random Forest",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='weighted'),
                recall_score(y_test, y_pred, average='weighted'),
                f1_score(y_test, y_pred, average='weighted'),
                safe_auc(y_test, y_prob) if y_prob is not None else "N/A"
            ])

            joblib.dump(rf.best_estimator_, "random_forest.pkl")

            # ----------------------------------------------------
            # 3️⃣ Support Vector Machine (SVM)
            # ----------------------------------------------------
            st.write("### SVM")
            svm_params = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            }

            svm = GridSearchCV(SVC(probability=True), svm_params, cv=5, scoring="accuracy")
            svm.fit(X_train_scaled, y_train)

            y_pred = svm.predict(X_test_scaled)
            y_prob = svm.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None

            model_scores.append([
                "SVM",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='weighted'),
                recall_score(y_test, y_pred, average='weighted'),
                f1_score(y_test, y_pred, average='weighted'),
                safe_auc(y_test, y_prob) if y_prob is not None else "N/A"
            ])

            joblib.dump(svm.best_estimator_, "svm.pkl")

            # ----------------------------------------------------
            # 4️⃣ KNN
            # ----------------------------------------------------
            st.write("### KNN")
            knn_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

            knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring="accuracy")
            knn.fit(X_train_scaled, y_train)

            y_pred = knn.predict(X_test_scaled)
            y_prob = knn.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None

            model_scores.append([
                "KNN",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='weighted'),
                recall_score(y_test, y_pred, average='weighted'),
                f1_score(y_test, y_pred, average='weighted'),
                safe_auc(y_test, y_prob) if y_prob is not None else "N/A"
            ])

            joblib.dump(knn.best_estimator_, "knn.pkl")

            # ----------------------------------------------------
            # 5️⃣ XGBoost
            # ----------------------------------------------------
            st.write("### XGBoost")
            xgb_params = {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }

            xgb = GridSearchCV(XGBClassifier(eval_metric="logloss"), xgb_params,
                               cv=5, scoring="accuracy")
            xgb.fit(X_train, y_train)

            y_pred = xgb.predict(X_test)
            y_prob = xgb.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

            model_scores.append([
                "XGBoost",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='weighted'),
                recall_score(y_test, y_pred, average='weighted'),
                f1_score(y_test, y_pred, average='weighted'),
                safe_auc(y_test, y_prob) if y_prob is not None else "N/A"
            ])

            joblib.dump(xgb.best_estimator_, "xgboost.pkl")

            # ----------------------------------------------------
            # FINAL COMPARISON TABLE
            # ----------------------------------------------------
            st.subheader("📊 Final Model Comparison")

            results = pd.DataFrame(
                model_scores,
                columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            )

            st.dataframe(results.style.highlight_max(axis=0, color="lightgreen"))
        import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

st.subheader("📊 Visualizations")

# -----------------------------  
# 1️⃣ Confusion Matrix  
# -----------------------------
st.write("### 🔹 Confusion Matrix (Best Model)")

# Select model for visualization
selected_model_name = st.selectbox(
    "Select a model for confusion matrix visualization:",
    ["Logistic Regression", "Random Forest", "SVM", "KNN", "XGBoost"]
)

# Map name to model object
model_map = {
    "Logistic Regression": log_reg.best_estimator_,
    "Random Forest": rf.best_estimator_,
    "SVM": svm.best_estimator_,
    "KNN": knn.best_estimator_,
    "XGBoost": xgb.best_estimator_,
}

best_model = model_map[selected_model_name]

# Predictions for selected model
if selected_model_name in ["Random Forest", "XGBoost"]:
    y_pred_viz = best_model.predict(X_test)
else:
    y_pred_viz = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_viz)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title(f"Confusion Matrix - {selected_model_name}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -----------------------------  
# 2️⃣ ROC Curve  
# -----------------------------
if len(np.unique(y)) == 2:

    st.write("### 🔹 ROC Curve (Binary Classification Only)")

    fig2, ax2 = plt.subplots()

    for name, model, scaled in [
        ("Logistic Regression", log_reg.best_estimator_, True),
        ("Random Forest", rf.best_estimator_, False),
        ("SVM", svm.best_estimator_, True),
        ("KNN", knn.best_estimator_, True),
        ("XGBoost", xgb.best_estimator_, False),
    ]:

        if scaled:
            X_used = X_test_scaled
        else:
            X_used = X_test

        try:
            y_prob_curve = model.predict_proba(X_used)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob_curve)
            ax2.plot(fpr, tpr, label=name)
        except:
            continue

    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.warning("ROC Curve not supported for multi-class datasets.")

# -----------------------------  
# 3️⃣ Feature Importance (RF & XGBoost Only)  
# -----------------------------
st.write("### 🔹 Feature Importance Plot")

model_choice = st.selectbox(
    "Select model for feature importance:",
    ["Random Forest", "XGBoost"]
)

if model_choice == "Random Forest":
    model = rf.best_estimator_
else:
    model = xgb.best_estimator_

importance = model.feature_importances_

fig3, ax3 = plt.subplots()
sns.barplot(x=importance, y=X.columns, ax=ax3)
ax3.set_title(f"Feature Importance - {model_choice}")
ax3.set_xlabel("Importance Score")
ax3.set_ylabel("Features")
st.pyplot(fig3)

# -----------------------------  
# 4️⃣ Model Performance Bar Chart  
# -----------------------------
st.write("### 🔹 Performance Comparison Bar Chart")

fig4, ax4 = plt.subplots(figsize=(10, 4))
sns.barplot(data=results, x="Model", y="Accuracy", ax=ax4)
ax4.set_title("Model Accuracy Comparison")
ax4.set_ylabel("Accuracy Score")
ax4.set_xticklabels(results["Model"], rotation=45)
st.pyplot(fig4)



else:
    st.info("👆 Upload a CSV file to begin")

