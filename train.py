import pandas as pd 
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


def open_data(path="data/housing_price_dataset.csv"):
    df = pd.read_csv(path)

    return df

def prepare_data(encoder, scaler, df, test_size, random_state, encode_columns, y_column):
    df[encode_columns] = encoder.fit_transform(df[encode_columns])
   
    X = df.drop(columns = [y_column], axis = 1)
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, y, X_train, X_test

def train_model(model, X_train):
    model.fit(X_train, y_train)

    return model

def save_model(path, model):
    with open(path, "wb") as file:
        joblib.dump(model, file, compress=3)
    
    return model

def prepare_data_and_model():
    df = open_data()

    encoder = LabelEncoder()
    scaler = StandardScaler()
    X, y, X_train, X_test = prepare_data(encoder, scaler, df, 0.3, 42, 'Neighborhood', 'Price')
    X_test.to_csv ('X_test.csv', index= False)
    y.to_csv ('target.csv', index= False)

    model = LinearRegression()
    model = train_model(model, X_train)
    model = save_model('model.joblib', model)


if __name__ == "__main__":
    prepare_data_and_model()

