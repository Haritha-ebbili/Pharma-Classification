#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, RocCurveDisplay
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("📊 Pharma Classification App")
st.write("Upload a dataset to compare ML models and make predictions.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # --------------------------
    # TARGET COLUMN SELECTION
    # --------------------------
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # --------------------------
    # CLEANING NUMERIC COLUMNS
    # --------------------------
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # --------------------------
    # SPLIT DATA
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # --------------------------
    # MODELS + GRID SEARCH
    # --------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
    }

    params = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
        "Decision Tree": {"max_depth": [3, 5, 7, None]},
        "Random Forest": {"n_estimators": [50, 100, 200]},
        "KNN": {"n_neighbors": [3, 5, 7]},
        "SVM": {"C": [0.1, 1, 10]},
    }

    results = []
    trained_models = {}

    st.subheader("🔍 Training Models...")

    for name, model in models.items():
        st.write(f"Training **{name}** ...")

        grid = GridSearchCV(
            estimator=model,
            param_grid=params[name],
            cv=5,
            scoring="accuracy"
        )

        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        trained_models[name] = best_model

        y_pred = best_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append([name, acc, pre, rec, f1])

    # --------------------------
    # RESULTS TABLE
    # --------------------------
    results_df = pd.DataFrame(
        results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    )

    st.subheader("📌 Model Comparison Table")
    st.dataframe(results_df)

    # --------------------------
    # ACCURACY PLOT
    # --------------------------
    st.subheader("📊 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results_df["Model"], results_df["Accuracy"], marker="o")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    st.pyplot(fig)

    # --------------------------
    # BEST MODEL
    # --------------------------
    best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    st.subheader(f"🔥 Best Model: **{best_model_name}**")

    # --------------------------
    # CONFUSION MATRIX
    # --------------------------
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)

    st.write("### Confusion Matrix")
    fig2, ax2 = plt.subplots()
    ax2.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    st.pyplot(fig2)

    # --------------------------
    # ROC CURVE (binary only)
    # --------------------------
    if len(np.unique(y)) == 2:
        st.write("### ROC Curve")
        fig3, ax3 = plt.subplots()
        RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test, ax=ax3)
        st.pyplot(fig3)

    # --------------------------
    # PREDICTION SECTION
    # --------------------------
    st.subheader("🧪 Predict with Best Model")

    predict_values = []

    for col in X.columns:
        default_value = float(X[col].mean()) if col in numeric_cols else 0.0
        val = st.number_input(f"Enter {col}", value=default_value)
        predict_values.append(val)

    if st.button("Predict"):
        input_df = pd.DataFrame([predict_values], columns=X.columns)

        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        pred = best_model.predict(input_df)[0]
        st.success(f"### Predicted Output: **{pred}**")
