import pandas as pd 
data_path="data/Telco-Customer-Churn.csv"

def load_data():
    df=pd.read_csv(data_path)
    return df

def clean_data(df):
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True)
    df.drop('customerID',axis=1,inplace=True)
    df['Churn']=df['Churn'].map({'Yes':1,'No':0})
    return df 

if __name__=="__main__":
    df=load_data()
    df=clean_data(df)
    #print(df.head())

    

