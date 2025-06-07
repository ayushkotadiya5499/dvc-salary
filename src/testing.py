import pandas as pd
import joblib
import json
import yaml
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    return pd.read_csv(path)

def x_y_split(df, target_col):
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    return x, y

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mae": round(mae, 4)
    }

def main():
    params = load_params()
    
    test_data_path = params["data"]["test_path"]
    model_path = params["model"]["output_path"]
    target_col = params["data"]["target"]
    metrics_output_path = params["evaluation"]["metrics_output"]

    test_df = load_data(test_data_path)
    x_test, y_test = x_y_split(test_df, target_col)

    model = joblib.load(model_path)
    metrics = evaluate_model(model, x_test, y_test)

    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Metrics written to", metrics_output_path)

if __name__ == '__main__':
    main()
