import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def build_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_test_split(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train, test = data[:train_size], data[train_size:]
    return train, test

def scale_data(train, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    return scaler, train_scaled, test_scaled

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

def evaluate_forecast(test, predictions):
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

def plot_results(train, test, predictions):
    plt.figure(figsize=(10,6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, predictions, color='red', label='Predicted')
    plt.title('LSTM Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def main():
    # Fetch stock data
    ticker = "TCS.NS"  # Example: Apple Inc.
    start_date = "2022-01-01"
    end_date = "2024-08-31"
    data = fetch_stock_data(ticker, start_date, end_date)

    # Split data into train and test sets
    train, test = train_test_split(data)

    # Scale the data
    scaler, train_scaled, test_scaled = scale_data(train, test)

    # Reshape data for LSTM [samples, time steps, features]
    look_back = 60  # Use 60 days of historical data to predict the next day
    X_train, Y_train = create_dataset(train_scaled, look_back)
    X_test, Y_test = create_dataset(test_scaled, look_back)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build and train the LSTM model
    model = build_lstm_model(look_back)
    model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=0)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions to original scale
    train_predict = inverse_transform(scaler, train_predict)
    Y_train = inverse_transform(scaler, Y_train.reshape(-1, 1))
    test_predict = inverse_transform(scaler, test_predict)
    Y_test = inverse_transform(scaler, Y_test.reshape(-1, 1))

    # Evaluate the forecast
    train_rmse = evaluate_forecast(Y_train, train_predict)
    test_rmse = evaluate_forecast(Y_test, test_predict)
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Align predictions with the correct dates
    train_predict_dates = train.index[look_back:len(train_predict)+look_back]
    test_predict_dates = test.index[look_back:len(test_predict)+look_back]

    # Plot the results
    plt.figure(figsize=(10,6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(train_predict_dates, train_predict, color='green', label='Train Predicted')
    plt.plot(test_predict_dates, test_predict, color='red', label='Test Predicted')
    plt.title('LSTM Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()