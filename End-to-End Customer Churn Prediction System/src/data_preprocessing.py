import pandas as pd 
data_path="data/Telco-Customer-Churn.csv"

def load_data():
    df=pd.read_csv(data_path)
    return df

if __name__=="__main__":
    df=load_data()
    #print(df.head())
    