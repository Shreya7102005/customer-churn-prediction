import joblib
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# Load saved model and column structure
model = joblib.load("models/churn_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# Load training data for LIME context
X_train = pd.read_csv("data/Telco-Customer-Churn.csv")
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['No Churn', 'Churn'],
    mode='classification'
)

def predict_churn(customer_data):
    df = pd.DataFrame([customer_data])

    # Binary encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)

    return prediction[0]


# Sample test customer
sample_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.5,
    'TotalCharges': 400.0
}

result = predict_churn(sample_customer)

print("Prediction:", result)