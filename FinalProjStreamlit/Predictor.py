import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


def getStockData(symbol, start_date, end_date):
    data = pd.DataFrame(yf.download(symbol, start=start_date, end=end_date))
    return data

def displayData():
    return st.line_chart(getStockData('AAPL', '2019-01-01', '2020-01-01')[["Close", "Open"]])

st.write(getStockData('AAPL', '2019-01-01', '2020-01-01'))
displayData()






