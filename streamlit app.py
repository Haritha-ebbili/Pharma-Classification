import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("🩺 Liver Disease Classification – Multi-Model Comparison")

# -------------------------
# 1. LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Liver_data.csv", sep=";")  # FIXED: semicolon separator

df = load_data()
st.write("### Dataset Preview", df.head())

# -------------------------
# 2. PREPROCESSING
# -------------------------
target_col = "Dataset"

X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical + numeric separation
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

# One-hot encoding only if needed
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling numeric features
scaler = StandardScaler()
if len(numeric_cols) > 0:
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# -------------------------
# 3. MODELS + PARAMETERS
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

params = {
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear"]
    },

    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, None]
    },

    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "criterion": ["gini", "entropy"]
    },

    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },

    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    }
}

cv = 5  # k-fold cross validation

# -------------------------
# 4. TRAIN + EVALUATE
# -------------------------
results = []
st.write("## 🔍 Training Models...")

for name, model in models.items():
    st.write(f"### Training {name}...")

    grid = GridSearchCV(model, params[name], cv=cv, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([
        name, acc, pre, rec, f1, grid.best_score_
    ])

# -------------------------
# 5. COMPARISON TABLE
# -------------------------
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score", "CV Mean Accuracy"
])

st.write("## 📊 Model Comparison Table")
st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "CV Mean Accuracy": "{:.4f}"
    })
)
