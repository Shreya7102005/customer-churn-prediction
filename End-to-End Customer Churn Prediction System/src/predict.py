import joblib
import pandas as pd
import os

# Absolute project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "model_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")

# Load model files
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Safe optional import for LIME
HAS_LIME = False
explainer = None

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except Exception:
    HAS_LIME = False

if HAS_LIME and os.path.exists(DATA_PATH):
    try:
        train_df = pd.read_csv(DATA_PATH)
        train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
        train_df['TotalCharges'] = train_df['TotalCharges'].fillna(train_df['TotalCharges'].median())
        if 'customerID' in train_df.columns:
            train_df.drop('customerID', axis=1, inplace=True)
        if 'Churn' in train_df.columns:
            train_df['Churn'] = train_df['Churn'].map({'Yes': 1, 'No': 0})
            X_train = train_df.drop('Churn', axis=1)
        else:
            X_train = train_df.copy()

        for col in binary_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].map({'Yes': 1, 'No': 0})

        categorical_cols = X_train.select_dtypes(include='object').columns
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_train = X_train.reindex(columns=model_columns, fill_value=0)

        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['No Churn', 'Churn'],
            mode='classification'
        )
    except Exception:
        HAS_LIME = False


def predict_churn(customer_data):
    df = pd.DataFrame([customer_data])

    # Binary encoding
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # Prediction & Probability
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]

    explanation_list = []
    
    if HAS_LIME and explainer is not None:
        try:
            explanation = explainer.explain_instance(
                df.iloc[0].values,
                model.predict_proba
            )
            explanation_list = explanation.as_list()
        except Exception:
            explanation_list = []

    # Fallback to feature importance if LIME is unavailable
    if not explanation_list:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            row_vals = df.iloc[0].values
            feature_impacts = [(col, float(val * imp)) for col, val, imp in zip(model_columns, row_vals, importances)]
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            explanation_list = feature_impacts[:8]

    return prediction[0], probability, explanation_list