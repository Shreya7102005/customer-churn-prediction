import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

data_path="data/Telco-Customer-Churn.csv"

def load_data():
    df=pd.read_csv(data_path)
    return df

def clean_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    df.drop('customerID', axis=1, inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

if __name__=="__main__":
    df=load_data()
    df=clean_data(df)
    #print(df.head())


    #train aur test data m split hua h
    X=df.drop('Churn',axis=1)
    y=df['Churn']
    #print("Features shape:",X.shape)
    #print("Target shape:",y.shape)

    

    #identify which columns have categorical value
    #print(X.dtypes)

    #koi koi categorical value wale output ko 0/1 ke form m convert krna h
    binaer_cols=['Partner','Dependents','PhoneService','PaperlessBilling']
    for col in binaer_cols:
        X[col]=X[col].map({'Yes':1 ,'No':0})
    #print(X[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']].head())

    categorical_cols = X.select_dtypes(include='object').columns
    #print("Categorical columns:", categorical_cols)
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    #print("\nAfter Encoding:")
    #print(X.head())
    #print("\nShape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #print("Train shape:", X_train.shape)
    #print("Test shape:", X_test.shape)

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #print("Sample Predictions:", y_pred[:10])

    #print("\nAccuracy:", accuracy_score(y_test, y_pred))

    #print("\nClassification Report:\n")
    #print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/churn_model.pkl")
    #print("Model saved successfully!")

    # Load the saved model
    loaded_model = joblib.load("models/churn_model.pkl")

    # Take one sample from test data
    sample = X_test.iloc[0:1]

    # Make prediction
    prediction = loaded_model.predict(sample)

    print("\nSample Prediction:", prediction)