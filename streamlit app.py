#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, RocCurveDisplay

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# --------------------------------------------
# STREAMLIT UI
# --------------------------------------------
st.title("📊 Pharma Classification")
st.write("Upload your dataset and get predictions + model comparison + visualizations")


# --------------------------------------------
# FILE UPLOAD
# --------------------------------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target = st.selectbox("Select Target Variable", df.columns)

    # Split feature & target
    X = df.drop(columns=[target])
    y = df[target]

    # FIX: HANDLE CATEGORICAL + NUMERIC COLUMNS
    # --------------------------------------------

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
    
    # Handle categorical variables safely
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        X = X.copy()

    # Split after encoding
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

    # Scale numerical columns only
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


    # --------------------------------------------
    # MODEL DEFINITIONS + GRID SEARCH
    # --------------------------------------------

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True)
    }

    params = {
        "Logistic Regression": {"C": [0.1, 1, 10]},
        "Decision Tree": {"max_depth": [3, 5, 7, None]},
        "Random Forest": {"n_estimators": [50, 100, 200]},
        "KNN": {"n_neighbors": [3,5,7]},
        "SVM": {"C": [0.1,1,10]}
    }

    results = []

    st.subheader("🔍 Training All Models...")

    trained_models = {}

    for model_name, model in models.items():
        st.write(f"Training **{model_name}** ...")
        
        grid = GridSearchCV(model, params[model_name], cv=5, scoring='accuracy')
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        trained_models[model_name] = best_model

        # Predictions
        y_pred = best_model.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append([model_name, acc, pre, rec, f1])


    # --------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

    st.subheader("📌 Model Comparison Table")
    st.dataframe(results_df.style.highlight_max(axis=0))


    # --------------------------------------------
    # VISUALIZATION
    # --------------------------------------------
    st.subheader("📊 Model Performance Visualization")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(results_df["Model"], results_df["Accuracy"], marker='o')
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy Score")
    plt.title("Model Accuracy Comparison")
    st.pyplot(fig)


    # --------------------------------------------
    # CONFUSION MATRIX FOR BEST MODEL
    # --------------------------------------------
    best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    st.subheader(f"🔥 Best Model: {best_model_name}")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)

    fig2, ax2 = plt.subplots()
    ax2.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    st.pyplot(fig2)

    # ROC Curve
    if len(np.unique(y)) == 2:  
        st.write("### ROC Curve")

        fig3, ax3 = plt.subplots()
        RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test, ax=ax3)
        st.pyplot(fig3)

    # --------------------------------------------
    # PREDICTION SECTION
    # --------------------------------------------
    st.subheader("🧪 Predict With Best Model")

    input_data = []
    for col in X.columns:
        val = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))
        input_data.append(val)

    if st.button("Predict"):
        final_input = scaler.transform([input_data])
        pred = best_model.predict(final_input)[0]
        st.success(f"### Predicted Output: **{pred}**")


