import os

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def open_data(path):
    df = pd.read_csv(path)

    return df


def prepare_data(
    encoder, scaler, df, test_size, random_state, encode_columns, y_column
):
    df[encode_columns] = encoder.fit_transform(df[encode_columns])

    features = df.drop(columns=[y_column], axis=1)
    target = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return features, target, X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

    return model


def save_model(path, model):
    with open(path, "wb") as file:
        joblib.dump(model, file, compress=3)

    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def prepare_data_and_model(cfg: DictConfig):
    df = open_data(cfg.data.path)

    encoder = LabelEncoder()
    scaler = StandardScaler()
    X, y, X_train, X_test, y_train, y_test = prepare_data(
        encoder,
        scaler,
        df,
        cfg.model.test_size,
        cfg.model.random_state,
        cfg.model.encode_columns,
        cfg.model.target_column,
    )

    X_test_df = pd.DataFrame(X_test)
    X_test_df.to_csv(cfg.save_names.x, index=False)
    y_df = pd.DataFrame(y_test)
    y_df.to_csv(cfg.save_names.y, index=False)

    model = LinearRegression()
    model = train_model(model, X_train, y_train)
    model = save_model(cfg.model.save_path, model)


if __name__ == "__main__":
    os.system("dvc fetch data/housing_price_dataset.csv")
    os.system("dvc pull")
    prepare_data_and_model()
    os.system("dvc add model.joblib")
    os.system("dvc push model.joblib.dvc")
