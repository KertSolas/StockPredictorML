import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title("Stock Trend Predictor")
today = date.today()
# startYear = today.year
# startYear = 0
# endYear = startYear + 5
year = st.slider("Select a year", 0, 5)
period = year * 365

@st.cache_data
def getStockData(symbol, start_date, end_date):
    data = pd.DataFrame(yf.download(symbol, start=start_date, end=end_date))
    data.reset_index(inplace=True)
    return data

def selectBox():
    tickers = pd.read_csv("nasdaq-listed.csv")
    stocks = tickers["Symbol"].tolist()
    selectionBox = st.selectbox("Select a stock", stocks)
    return selectionBox

def trainModel(data):
    df_train = data[["Date", "Close", "Volume"]]
    df_train = data.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df_train) 
    return model

def forecast(model):
    future = model.make_future_dataframe(periods= period)
    forecast = model.predict(future)

    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

def displayData(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_open"))
    fig.layout.update(xaxis_rangeslider_visible=True)   
 
    st.plotly_chart(fig) 



def main():

    selectionBox = selectBox()
    data = getStockData(selectionBox, "2015-01-01", today)

    st.subheader(selectionBox + " Data")
    st.write(data.tail())
    st.title(f"{selectionBox}")

    displayData(data) #raw data
    train = trainModel(data)
    
    st.subheader("Forecast Data")
    forecastData = forecast(train)
    forecastData

main()








