import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

import matplotlib.pyplot as plt


def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def prepare_data(df):
    df['Date'] = pd.to_datetime(df.index)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Close']
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def predict_future_prices(model, last_date, days_to_predict):
    future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1)[1:]
    future_days = (future_dates - last_date).days.values.reshape(-1, 1)
    future_prices = model.predict(future_days)
    return pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices})

def plot_data(predicted_df):
    predicted_df.plot(x="Date", y="Predicted_Price", kind="line")
    plt.show()


def main():
    ticker = "TCS.NS"  # Example: Apple Inc.
    start_date = "2022-01-01"
    end_date = "2024-08-31"
    days_to_predict = 30

    # Fetch historical data
    df = fetch_stock_data(ticker, start_date, end_date)

    # Prepare data
    X, y = prepare_data(df)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate model
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

    # Predict future prices
    last_date = df.index[-1]
    future_predictions = predict_future_prices(model, last_date, days_to_predict)
    print("\nPredicted prices for the next 30 days:")
    print(future_predictions)

    plot_data(predicted_df=future_predictions)


if __name__ == "__main__":
    main()
