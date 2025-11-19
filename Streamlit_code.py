#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# --------------------- Streamlit Page Config ---------------------
st.set_page_config(page_title="ML Model Comparison App", layout="wide")

st.title("📊 Machine Learning Model Performance Comparison")
st.write("Upload a dataset and compare ML models easily!")


# --------------------- File Upload ---------------------
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.info("Training models... Please wait.")

        # Store results
        results = []

        # ------------------------------------------------------------
        # 1️⃣ KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        y_pred_knn = knn.predict(X_test_scaled)

        results.append([
            "K-Nearest Neighbors",
            accuracy_score(y_test, y_pred_knn),
            precision_score(y_test, y_pred_knn, average="macro"),
            recall_score(y_test, y_pred_knn, average="macro"),
            f1_score(y_test, y_pred_knn, average="macro"),
            roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_knn), average="macro"),
            cross_val_score(knn, X_train_scaled, y_train, cv=5).mean()
        ])

        # ------------------------------------------------------------
        # 2️⃣ SVM
        svm = SVC(probability=True)
        svm.fit(X_train_scaled, y_train)
        y_pred_svm = svm.predict(X_test_scaled)

        results.append([
            "Support Vector Machine",
            accuracy_score(y_test, y_pred_svm),
            precision_score(y_test, y_pred_svm, average="macro"),
            recall_score(y_test, y_pred_svm, average="macro"),
            f1_score(y_test, y_pred_svm, average="macro"),
            roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_svm), average="macro"),
            cross_val_score(svm, X_train_scaled, y_train, cv=5).mean()
        ])

        # ------------------------------------------------------------
        # 3️⃣ Logistic Regression
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)

        results.append([
            "Logistic Regression",
            accuracy_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_lr, average="macro"),
            recall_score(y_test, y_pred_lr, average="macro"),
            f1_score(y_test, y_pred_lr, average="macro"),
            roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_lr), average="macro"),
            cross_val_score(lr, X_train_scaled, y_train, cv=5).mean()
        ])

        # ------------------------------------------------------------
        # 4️⃣ Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        results.append([
            "Random Forest",
            accuracy_score(y_test, y_pred_rf),
            precision_score(y_test, y_pred_rf, average="macro"),
            recall_score(y_test, y_pred_rf, average="macro"),
            f1_score(y_test, y_pred_rf, average="macro"),
            roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_rf), average="macro"),
            cross_val_score(rf, X_train, y_train, cv=5).mean()
        ])

        # ------------------------------------------------------------
        # 5️⃣ Decision Tree
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)

        results.append([
            "Decision Tree",
            accuracy_score(y_test, y_pred_dt),
            precision_score(y_test, y_pred_dt, average="macro"),
            recall_score(y_test, y_pred_dt, average="macro"),
            f1_score(y_test, y_pred_dt, average="macro"),
            roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_dt), average="macro"),
            cross_val_score(dt, X_train, y_train, cv=5).mean()
        ])

        # ------------------------------------------------------------
        # 📌 Final Comparison Table
        st.subheader("📊 Model Performance Comparison Table")

        metrics_df = pd.DataFrame(
            results,
            columns=[
                "Model", "Accuracy", "Precision", "Recall",
                "F1 Score", "ROC-AUC", "CV Mean Accuracy"
            ]
        )

        st.dataframe(metrics_df)

        # ------------------------------------------------------------
        # 📈 Accuracy Visualization
        st.subheader("📈 Accuracy Comparison")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(metrics_df["Model"], metrics_df["Accuracy"])
        plt.xticks(rotation=45)
        plt.ylabel("Accuracy Score")
        plt.title("Model Accuracy Comparison")
        st.pyplot(fig)

