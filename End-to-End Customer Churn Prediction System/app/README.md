# ChurnGuard 🛡️

## AI-Powered Customer Churn Prediction & Retention System

ChurnGuard is an end-to-end Machine Learning application that predicts whether a telecom customer is likely to churn or stay.

It uses customer demographics, service details, contract information, and billing data to generate a churn prediction and probability.

## 🚀 Features

- Customer churn prediction
- Churn probability estimation
- Random Forest classification
- Data preprocessing and feature encoding
- Explainable AI using LIME
- Interactive Streamlit web application
- Model persistence using Joblib
- Cloud deployment

## 🧠 Machine Learning

**Model:** Random Forest Classifier

The model is trained on the IBM Telco Customer Churn dataset containing 7,000+ customer records.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- LIME
- Joblib
- Streamlit
- Git & GitHub

## 🔄 Workflow

```text
Customer Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Random Forest Model
      ↓
Churn Prediction
      ↓
Churn Probability
      ↓
Prediction Explanation
      ↓
Streamlit Application

End-to-End Customer Churn Prediction System/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│   ├── churn_model.pkl
│   └── model_columns.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── requirements.txt
└── runtime.txt