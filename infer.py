import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter


def make_prediction(test_dataset="data/test_data.csv", model_path="model.joblib"):
    X_test = pd.read_csv(test_dataset)
    X_test = X_test.to_numpy()

    with open(model_path, "rb") as file:
        model = joblib.load(file)
    prediction = model.predict(X_test)
    params = model.get_params()

    return prediction, params


def get_scores(summary_writer, test_target="data/target.csv"):
    y_test = pd.read_csv(test_target)
    y_pred, params = make_prediction()

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv("predictions.csv", index=False)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics_names = ["mse", "mae", "rmse", "r2_score"]
    meitrics = [mse, mae, rmse, r2]

    for metric, name in zip(meitrics, metrics_names):
        summary_writer.add_scalar(name, metric, global_step=0)

    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2_score": r2}


if __name__ == "__main__":
    summary_writer = SummaryWriter("exp_logs")
    os.system("dvc fetch data/X_test.csv")
    os.system("dvc fetch data/target.csv")
    os.system("dvc fetch data/model.joblib")
    get_scores(summary_writer)
    os.system("dvc add predictions.csv")
    os.system("dvc push predictions.csv.dvc")
