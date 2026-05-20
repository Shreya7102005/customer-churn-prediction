import pandas as pd

data_path = "data/Telco-Customer-Churn.csv"


def load_data():
    df = pd.read_csv(data_path)
    return df


def clean_data(df):
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Drop useless column
    df.drop('customerID', axis=1, inplace=True)

    # Convert target column
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df