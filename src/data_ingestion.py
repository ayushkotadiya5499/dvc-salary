import pandas as pd
import numpy as np  
import os
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df=pd.read_csv(data_path)
    return df

def train_test_df(df,test_size, random_state):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def save_data(train_df,test_df,path):
    path=os.path.join(path, 'raw')
    os.makedirs(path, exist_ok=True)
    train_df.to_csv(os.path.join(path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(path, 'test.csv'), index=False)

def main():
    data_path = 'C:/Users/ayush/Desktop/dvc/salary_dataset.csv'
    df = load_data(data_path)
    train_df,test_df=train_test_df(df,0.2, 42)
    save_data(train_df, test_df, 'data')

if __name__=='__main__':
    main()
    print("Data ingestion completed successfully.")
    

    
    