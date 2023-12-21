import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_prediction(test_dataset="X_test.csv", model_path="model.joblib"):
    X_test = pd.read_csv(test_dataset)
    X_test = X_test.to_numpy()

    with open(model_path, "rb") as file:
        model = joblib.load(file)
    prediction = model.predict(X_test)
    params = model.get_params()

    return prediction, params


def get_scores(test_target="target.csv"):
    y_test = pd.read_csv(test_target)
    y_pred, params = make_prediction()

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv("predictions.csv", index=False)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    with mlflow.start_run(run_name="House prices") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    return metrics


if __name__ == "__main__":
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    client.create_experiment(name="House prices")
    get_scores()
