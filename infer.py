import pandas as pd 
import joblib
from sklearn.metrics import mean_squared_error


def make_prediction(test_dataset='X_test.csv', model_path='model.joblib'):
    X_test = pd.read_csv(test_dataset)

    with open(model_path, "rb") as file:
         model = joblib.load(file)
    prediction = model.predict(X_test)

    return prediction

def get_mse_score(test_target='target.csv'):
    y_test = pd.read_csv(test_dataset)
    y_pred = make_prediction()
    mse = mean_squared_error(y_test, y_pred)

    return f'test MSE: {mse}'

if __name__ == "__main__":
    get_mse_score()