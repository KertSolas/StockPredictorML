import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def predict_stock_prices(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].astype(np.int64) // 10**9  
    X = data[['Date']]
    y = data['Close']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict stock prices on both training and testing sets
    data['Predicted_Close'] = model.predict(X)
    data.loc[X_train.index, 'Predicted_Close_Train'] = model.predict(X_train)
    data.loc[X_test.index, 'Predicted_Close_Test'] = model.predict(X_test)

    # Calculate mean squared error for both training and testing sets
    mse_train = mean_squared_error(y_train, data.loc[X_train.index, 'Predicted_Close_Train'])
    mse_test = mean_squared_error(y_test, data.loc[X_test.index, 'Predicted_Close_Test'])

    st.write(f"Mean Squared Error (Train): {mse_train:.2f}")
    st.write(f"Mean Squared Error (Test): {mse_test:.2f}")

    return data

# Function to display the stock predictor app
def main():
    st.title("Stock Predictor App")

    # User input for stock selection and date range
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Select Start Date:", pd.to_datetime('2022-01-01'))
    end_date = st.date_input("Select End Date:", pd.to_datetime('2023-01-01'))

    # Fetch historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Predict stock prices using linear regression
    predicted_data = predict_stock_prices(stock_data)

    # Display historical stock data
    st.subheader(f"{ticker} Stock Data")
    st.write(stock_data)

    # Display regression plot using Plotly
    st.subheader("Regression Plot")
    fig = px.scatter(predicted_data, x='Date', y=['Close', 'Predicted_Close'], title='Stock Price Prediction')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
