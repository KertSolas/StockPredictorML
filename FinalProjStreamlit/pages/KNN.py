import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def predict_knn(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].astype(np.int64) // 10**9  
    X = data[['Date']]
    y = data['Close']
    return X, y

def main():
    st.title("Stock Predictor App")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Select Start Date:", pd.to_datetime('2022-01-01'))
    end_date = st.date_input("Select End Date:", pd.to_datetime('2023-01-01'))

    stock_data_value = stock_data(ticker, start_date, end_date)
    X, y = predict_knn(stock_data_value)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k = 3
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)

    y_pred = knn_model.predict(X_test_scaled)

    st.subheader(f"{ticker} Stock Data")
    st.write(stock_data_value)

    st.subheader("Regression Plot")
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                     title="Actual vs Predicted Values")
    st.plotly_chart(fig)

    st.subheader("Model Evaluation")
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")

main()
