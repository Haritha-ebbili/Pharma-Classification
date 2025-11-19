import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

st.title("🩺 Liver Disease Classification – Save Best Model (Pickle)")

# -------------------------
# 1. UPLOAD FILE
# -------------------------
uploaded_file = st.file_uploader("Upload Liver_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📌 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # 2. AUTO-DETECT TARGET COLUMN
    # -------------------------
    target_col = df.columns[-1]   # Last column automatically
    st.success(f"✔ Target column detected automatically: `{target_col}`")

    # -------------------------
    # 3. SPLIT FEATURES & TARGET
    # -------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Auto encode categorical
    X = pd.get_dummies(X, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # 4. MODELS + PARAMETERS
    # -------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
    }

    params = {
        "Logistic Regression": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear"]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None]
        },
        "Decision Tree": {
            "max_depth": [3, 5, 10, None]
        }
    }

    # -------------------------
    # 5. TRAIN MODELS
    # -------------------------
    st.write("### ⚙ Training Models with GridSearchCV...")

    best_model_object = None
    best_accuracy = -1
    best_model_name = ""
    results = []

    for model_name, model in models.items():
        st.write(f"### Training {model_name}...")

        grid = GridSearchCV(model, params[model_name], cv=5, scoring="accuracy")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append([model_name, acc, grid.best_score_])

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_object = best_model
            best_model_name = model_name

    # -------------------------
    # 6. SHOW RESULTS TABLE
    # -------------------------
    results_df = pd.DataFrame(
        results,
        columns=["Model", "Test Accuracy", "CV Mean Accuracy"]
    )

    st.write("## 📊 Model Comparison")
    st.dataframe(results_df)

    # -------------------------
    # 7. SAVE BEST MODEL
    # -------------------------
    st.write("## 💾 Save Best Model")

    st.success(f"🏆 Best Model: **{best_model_name}** (Accuracy: {best_accuracy:.4f})")

    pickle_model = pickle.dumps(best_model_object)

    st.download_button(
        label="⬇ Download Best Model (best_model.pkl)",
        data=pickle_model,
        file_name="best_model.pkl",
        mime="application/octet-stream"
    )
