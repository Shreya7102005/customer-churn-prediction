import joblib
import pandas as pd
import os
from lime.lime_tabular import LimeTabularExplainer

# Absolute project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "model_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")

# Load model files
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# Load training data
train_df = pd.read_csv(DATA_PATH)

# Preprocessing
train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
train_df['TotalCharges'] = train_df['TotalCharges'].fillna(train_df['TotalCharges'].median())
train_df.drop('customerID', axis=1, inplace=True)
train_df['Churn'] = train_df['Churn'].map({'Yes': 1, 'No': 0})

X_train = train_df.drop('Churn', axis=1)

# Binary encoding
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

for col in binary_cols:
    X_train[col] = X_train[col].map({'Yes': 1, 'No': 0})

# One-hot encoding
categorical_cols = X_train.select_dtypes(include='object').columns
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)

# Match columns
X_train = X_train.reindex(columns=model_columns, fill_value=0)

# LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['No Churn', 'Churn'],
    mode='classification'
)


def predict_churn(customer_data):
    df = pd.DataFrame([customer_data])

    # Binary encoding
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)

    # Probability
    probability = model.predict_proba(df)[0][1]

    # LIME explanation
    explanation = explainer.explain_instance(
        df.iloc[0].values,
        model.predict_proba
    )

    explanation_list = explanation.as_list()

    return prediction[0], probability, explanation_list