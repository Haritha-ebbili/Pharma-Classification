import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import os

# --- Configuration ---
st.set_page_config(page_title="Multi-Model Liver Disease Prediction App", layout="wide")
FILE_NAME = "Liver_data.csv"

# --- Custom Preprocessing Class ---
# Custom encoder for 'sex' since ColumnTransformer needs numerical output
class CustomCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes 'sex' column ('m'->1, 'f'->0)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_encoded = X.copy()
        # Map 'm' to 1 and 'f' to 0
        X_encoded.loc[:, 'sex'] = X_encoded['sex'].map({'m': 1, 'f': 0}).fillna(0)
        return X_encoded.values # Return as a NumPy array for compatibility

# --- Data Loading and Multi-Model Training Function ---
@st.cache_resource
def load_data_and_train_models(file_path):
    """Loads data, preprocesses it, and trains multiple classification models."""
    try:
        # 1. Load Data
        df = pd.read_csv(file_path, sep=';')
        df = df.replace('NA', np.nan)
        for col in df.columns:
            if col not in ['category', 'sex']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Target Variable Encoding (y)
        # 'cirrhosis' -> 1 (Disease Present), 'no_disease' -> 0 (No Disease)
        le = LabelEncoder()
        df['category'] = le.fit_transform(df['category'])
        y = df['category']
        X = df.drop('category', axis=1)

        # 3. Feature Preparation
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = ['sex']
        feature_names = numerical_features + categorical_features

        # 4. Define Preprocessing Pipeline
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', CustomCategoricalEncoder(), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # 5. Define Models to Train
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest Classifier': RandomForestClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        
        trained_pipelines = {}
        
        # 6. Train Each Model
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipeline.fit(X, y)
            trained_pipelines[name] = pipeline

        return trained_pipelines, feature_names

    except Exception as e:
        st.error(f"Error during data loading or model training: {e}")
        return None, None

# --- Main Application Logic ---

if not os.path.exists(FILE_NAME):
    st.error(f"Error: The file '{FILE_NAME}' was not found. Please ensure it is in the same directory.")
else:
    # Load and train all models
    trained_models, feature_names = load_data_and_train_models(FILE_NAME)

    if trained_models:
        st.title("🩺 Multi-Model Liver Disease Prediction App")
        st.markdown("---")
        
        # 1. Model Selection
        st.sidebar.subheader("Model Configuration")
        model_choice = st.sidebar.selectbox(
            "Select Classification Model:",
            list(trained_models.keys())
        )
        
        # Get the selected model pipeline
        selected_model = trained_models[model_choice]

        # 2. Input Section
        st.subheader("Input Patient Data")
        
        col1, col2, col3 = st.columns(3)
        input_data = {}
        
        # Helper dictionary for all feature labels and typical ranges
        # Keys are the original column names from the CSV
        feature_details = {
            'age': ('Age (years)', col1, st.slider, (18, 90, 45)),
            'sex': ('Sex', col1, st.radio, (['m', 'f'], 'm')),
            'albumin': ('Albumin (g/L)', col2, st.number_input, (20.0, 60.0, 40.0, 0.1)),
            'alkaline_phosphatase': ('Alkaline Phosphatase (U/L)', col2, st.number_input, (0.0, 1000.0, 80.0, 1.0)),
            'alanine_aminotransferase': ('Alanine Aminotransferase (U/L)', col2, st.number_input, (0.0, 500.0, 30.0, 1.0)),
            'aspartate_aminotransferase': ('Aspartate Aminotransferase (U/L)', col3, st.number_input, (0.0, 500.0, 30.0, 1.0)),
            'bilirubin': ('Bilirubin ($\mu$mol/L)', col3, st.number_input, (0.0, 30.0, 10.0, 0.1)),
            'cholinesterase': ('Cholinesterase (kU/L)', col3, st.number_input, (0.0, 20.0, 8.0, 0.01)),
            'cholesterol': ('Cholesterol (mmol/L)', col1, st.number_input, (1.0, 10.0, 4.0, 0.01)),
            'creatinina': ('Creatinina ($\mu$mol/L)', col2, st.number_input, (50.0, 200.0, 90.0, 1.0)),
            'gamma_glutamyl_transferase ': ('Gamma-Glutamyl Transferase (U/L)', col3, st.number_input, (10.0, 500.0, 50.0, 1.0)),
            'protein': ('Protein (g/L)', col1, st.number_input, (50.0, 100.0, 70.0, 0.1)),
        }

        # Dynamically create input widgets
        for feature, (label, col, widget_func, args) in feature_details.items():
            with col:
                # The unpack operator (*) is used to pass the tuple of arguments to the widget function
                input_data[feature] = widget_func(label, *args)


        st.markdown("---")

        if st.button(f'Predict Liver Disease using {model_choice}'):
            # 3. Create a DataFrame from the user input
            input_df = pd.DataFrame([input_data])
            
            # Ensure columns are in the correct order for the model
            input_df = input_df[feature_names]
            
            # 4. Make Prediction
            # The selected_model pipeline handles all the preprocessing and prediction
            prediction = selected_model.predict(input_df)[0]
            prediction_proba = selected_model.predict_proba(input_df)[0]

            st.subheader("Prediction Result")
            st.markdown(f"**Model Used:** `{model_choice}`")

            if prediction == 1:
                st.error(f"**Prediction: Cirrhosis Detected**")
                st.write(f"This indicates a high probability of liver disease.")
            else:
                st.success(f"**Prediction: No Disease Detected**")
                st.write(f"This indicates a low probability of liver disease.")
            
            st.markdown(f"**Probability of Cirrhosis (Disease Present):** `{prediction_proba[1]:.2%}`")
            st.markdown(f"**Probability of No Disease:** `{prediction_proba[0]:.2%}`")
            st.markdown("---")
            st.caption("Disclaimer: This is a classification result based on machine learning models and should not be used as medical advice.")
