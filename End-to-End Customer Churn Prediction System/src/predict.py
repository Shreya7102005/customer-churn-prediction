import joblib
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# Load saved model and column structure
model = joblib.load("models/churn_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# Load original training data
train_df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Same preprocessing as training
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

# Match exact training columns
X_train = X_train.reindex(columns=model_columns, fill_value=0)

# Create LIME explainer
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

    # Match model columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)

    # LIME explanation
    explanation = explainer.explain_instance(
        df.iloc[0].values,
        model.predict_proba
    )

    print("\nWhy this prediction happened:")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight}")

    return prediction[0]


# Sample customer
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

print("\nPrediction:", result)