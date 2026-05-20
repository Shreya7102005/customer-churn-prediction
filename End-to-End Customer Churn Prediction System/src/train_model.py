from data_preprocessing import load_data, clean_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


df = load_data()
df = clean_data(df)

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Binary encoding
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

for col in binary_cols:
    X[col] = X[col].map({'Yes': 1, 'No': 0})

# One-hot encoding
categorical_cols = X.select_dtypes(include='object').columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/churn_model.pkl")

# Save column structure for prediction
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("Model training completed!")