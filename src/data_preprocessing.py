import pandas as pd
import joblib
import os
import yaml
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    return pd.read_csv(path)

def x_y_split(df, target_col):
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    return x, y

def get_numeric_transformer():
    return Pipeline([("scaler", StandardScaler())])

def get_categorical_transformer():
    return Pipeline([
        ("encoder", OrdinalEncoder(categories=[
            ['Analyst', 'Data Scientist', 'DL Engineer', 'ML Engineer', 'AI Specialist'],
            ["Bachelor's", "Master's", 'PhD'],
            ['Rural', 'Suburban', 'Urban']
        ]))
    ])

def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", get_numeric_transformer(), num_cols),
        ("cat", get_categorical_transformer(), cat_cols)
    ])

def build_model(preprocessor, model_params, random_state):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(**model_params, random_state=random_state))
    ])

def main():
    # Load params
    params = load_params()

    # Extract values
    train_path = params["data"]["train_path"]
    target_col = params["data"]["target"]
    num_cols = params["features"]["num_cols"]
    cat_cols = params["features"]["cat_cols"]
    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]
    model_output_path = params["model"]["output_path"]

    # Optional: dynamically unpack model hyperparameters
    model_params = {k: v for k, v in params["model"].items() if k not in ["output_path"]}

    # Load and split
    df = load_data('C:\\Users\\ayush\\Desktop\\dvc\\data\\raw\\train.csv')
    x, y = x_y_split(df,'salary')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Build pipeline
    preprocessor = build_preprocessor(num_cols, cat_cols)
    model = build_model(preprocessor, model_params, random_state)

    # Train and save
    model.fit(x_train, y_train)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model trained and saved to {model_output_path}")

if __name__ == '__main__':
    main()
