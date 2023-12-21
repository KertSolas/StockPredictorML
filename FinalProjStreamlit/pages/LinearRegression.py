import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import plotly.express as px


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def create_features(data):
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    return data

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to make predictions using the trained model
def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

# Main function to run the Streamlit app
def main():
    st.title('Stock Price Prediction App')

    # Sidebar for user input
    st.sidebar.header('User Input')
    ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL):', 'AAPL')
    start_date = st.sidebar.text_input('Enter Start Date (YYYY-MM-DD):', '2020-01-01')
    end_date = st.sidebar.text_input('Enter End Date (YYYY-MM-DD):', '2022-01-01')

    # Fetch historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Display the stock data
    st.subheader('Stock Data')
    st.write(stock_data)

    # Create features from stock data
    features_data = create_features(stock_data)

    # Select features and target variable
    X = features_data[['Year', 'Month', 'Day']]
    y = features_data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = train_linear_regression(X_train, y_train)

    # Make predictions on the test set
    predictions = make_predictions(model, X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    st.subheader('Model Evaluation')
    st.write(f'Mean Squared Error: {mse}')

    # Plot the actual vs. predicted stock prices using Plotly
    fig = px.scatter(x=X_test.index, y=[y_test, predictions], labels={'y': 'Stock Price'}, 
                  title='Actual vs. Predicted Prices')
    fig.update_layout(xaxis_title='Date')
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
