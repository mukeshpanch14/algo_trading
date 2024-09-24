import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import yfinance as yf
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']
def train_test_split(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train, test = data[:train_size], data[train_size:]
    return train, test
def arima_forecast(train, test, order):
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    return predictions
def evaluate_forecast(test, predictions):
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse
def plot_results(train, test, predictions):
    plt.figure(figsize=(10,6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, predictions, color='red', label='Predicted')
    plt.title('ARIMA Stock Price Prediction')
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
    # Define ARIMA order (p,d,q)
    order = (1, 1, 1)  # This is just an example, you may need to tune these parameters
    # Forecast using ARIMA
    predictions = arima_forecast(train, test, order)
    # Evaluate the forecast
    rmse = evaluate_forecast(test, predictions)
    print(f"Root Mean Squared Error: {rmse}")
    print(predictions)
    # Plot the results
    plot_results(train, test, predictions)
if __name__ == "__main__":
    main()