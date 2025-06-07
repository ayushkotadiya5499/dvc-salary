import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

def load_data(data_path):
    return pd.read_csv(data_path)

def train_test_df(df, test_size, random_state):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def save_data(train_df, test_df, path):
    os.makedirs(path, exist_ok=True)
    train_df.to_csv(os.path.join(path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(path, 'test.csv'), index=False)

def main():
    # Load params from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    data_path = params["data"]["source_path"]
    save_path = params["data"]["raw_path"]
    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    # Load and split
    path='C:/Users/ayush/Desktop/dvc/salary_dataset.csv'
    df = load_data(path)
    train_df, test_df = train_test_df(df, test_size, random_state)

    # Save
    save_data(train_df, test_df, save_path)
    print("Data ingestion completed successfully.")

if __name__ == '__main__':
    main()
