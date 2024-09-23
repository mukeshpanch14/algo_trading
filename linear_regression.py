import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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