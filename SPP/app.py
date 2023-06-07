import numpy as np
import pandas as pd 
import yfinance as yf
import streamlit as st 
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

try : 
    # Title
    st.title('Stock Trend Prediction')

    # get the list of all stocks listed on NSE
    nse = pd.read_csv('ind_nifty50list.csv')
    pd.set_option('display.max_colwidth', 100)

    # get the name of all the stocks
    stock_list = nse['Symbol'].to_list()
    stock_company = nse['Company Name'].to_list()

    # user input
    user_input = st.selectbox('Select Stock', stock_company)
    user_input_start_date = st.date_input('Start Date', value = pd.to_datetime('2019-01-01'))
    user_input_end_date = st.date_input('End Date', value = pd.to_datetime('2020-01-01'))

    # data
    symbol = stock_list[stock_company.index(user_input)] + '.NS'
    ticker = yf.Ticker(symbol)
    stock = ticker.history(start = user_input_start_date, end = user_input_end_date)
    
    # change Date format in stock dataframe
    stock.reset_index(inplace = True)
    stock['Date'] = stock['Date'].dt.strftime('%Y-%m-%d') 

    #Describing Date
    st.subheader('Date from 2010-2019')
    st.dataframe(stock)

    #Visualizaion
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize =(12, 6))
    plt.plot(stock.Close)
    st.pyplot(fig)

    # Moving Average
    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = stock.Close.rolling(100).mean()
    fig = plt.figure(figsize =(12, 6))
    plt.plot(ma100,'g')
    plt.plot(stock.Close)
    st.pyplot(fig)

    # Moving Average
    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = stock.Close.rolling(100).mean()
    ma200 = stock.Close.rolling(200).mean()
    fig = plt.figure(figsize =(12, 6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(stock.Close)
    st.pyplot(fig)

    #splitting data into training and testing
    data_train = pd.DataFrame(stock['Close'][0:int(len(stock)*0.70)])
    data_test = pd.DataFrame(stock['Close'][int(len(stock)*0.70):int(len(stock))])


    # Normalizing the data
    print(data_train.shape)
    print(data_test.shape)

    # Normalizing the data
    scaler = MinMaxScaler(feature_range = (0,1))

    # scaling the data
    data_train_array = scaler.fit_transform(data_train)
    #splitting data into x_train and y_train

    #load my model
    model = load_model('keras_model.h5')

    #testing part
    past_100_days = data_train.tail(100)
    final_stock = pd.concat([past_100_days, data_test]) ['Close']
    final_stock = np.array(final_stock).reshape(-1,1)
    input_data = scaler.fit_transform(final_stock)

    # x_test and y_test
    x_test = []
    y_test = []

    # appending past 100 days
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    # converting x_test and y_test into numpy arrays
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    # scale_factor
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # final Visualization
    st.subheader('Prediction VS Original')
    fig2= plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price ')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

except Exception as e:
    print(e)
    st.error('Error : System is not able to process your request. Please try again later.', icon = '🚨')