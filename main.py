import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import numpy
import yfinance as yf
# from matplotlib import pyplot as pp
import datetime as dt
from calendar import monthrange as mr
from flask import Flask, render_template, request, redirect
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_score
# import pandas as pndx
from curl_cffi import requests

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import math

# from tensorflow.python.keras.utils.generic_utils import to_list


app = Flask(__name__)


@app.route("/")
def start():
    return render_template('homepage.html')


@app.route("/aboutme")
def about():
    return render_template("aboutme.html")


@app.route("/faqs")
def faq():
    return render_template("faqs.html")


# Function to get if a stock exists on Yahoo Finance
def is_real_stock(new):
    session = requests.Session(impersonate="chrome")

    s = yf.Ticker(new, session=session)
    hist = s.history()

    if hist.empty:
        return False
    else:
        return True


def predictprice(num, stocklist, stockname):
    stkc = stocklist["Close"]
    stkclist = stkc.tolist()
    # stkclist.append("0")
    stkdata = numpy.array(stkclist).reshape(-1, 1)
    print(stkdata)
    train_len = math.ceil(len(stkdata) * 0.98)
    print(train_len)

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_data = stkdata[0:train_len, :]
    scaled_dataset = scaler.fit_transform(stkdata)
    scaled_test_dataset = scaler.fit_transform(train_data)

    x_train = []
    y_train = []

    for i in range((len(scaled_test_dataset) - num), len(scaled_test_dataset)):
        x_train.append(scaled_test_dataset[i - num:i, 0])
        y_train.append(scaled_test_dataset[i, 0])

    x_train, y_train = numpy.array(x_train), numpy.array(y_train)

    x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1), dropout=0.4)))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    # model.add(Dropout(0.1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=16, epochs=64, verbose=1, callbacks=[early_stopping])

    test_data = scaled_dataset[train_len - num:, :]
    x_test = []
    y_test = stkdata[train_len:, :]

    for i in range(num, len(test_data)):
        x_test.append(test_data[i - num:i, 0])

    x_test = numpy.array(x_test)
    x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # rmse = numpy.sqrt(numpy.mean(((list(predictions) - y_test) ** 2)))
    # print(rmse)

    train = stkdata[:train_len]
    valid = stkdata[train_len:]
    # valid['Predictions'] = predictions

    predictionfinal = predictions.tolist()

    finalvalue = round(float((predictionfinal[-1])[0]), 2)

    p = f'${finalvalue}'
    return render_template("predict.html", stockPick=stockname, endval=p)


@app.route("/stockinfo")
def get_stock_name():
    # global stock
    stock_name = request.args.get('stockname')
    stock = (stock_name.upper())
    # is_real_stock(stock)

    if stock == "":
        return render_template("error.html", msg="No stock name entered.")

    if is_real_stock(stock):
        stock = stock.upper()
    else:
        returnmsg = str(f"{stock} is not a real stock.")
        return render_template("error.html", msg=returnmsg)


    # global stock
    session = requests.Session(impersonate="chrome")

    stk = yf.Ticker(stock, session=session)
    stk = stk.history(period="max")
    stkml = stk.loc['2024-11-3':].copy()
    print(stkml["Close"])

    stk["Tomorrow"] = stk["Close"].shift(-1)
    stk["Target"] = (stk["Tomorrow"] > stk["Close"]).astype(int)

    stkml["Tomorrow1"] = stkml["Close"].shift(-1)
    stkml["Target1"] = (stkml["Tomorrow1"] > stkml["Close"]).astype(int)


    stk["MA 15 Days"] = stk['Close'].rolling(15).mean()
    stk["Ratio"] = stk["Close"] / stk["MA 15 Days"]

    stkml["MA 7 Days"] = stkml['Close'].rolling(2).mean()
    stk["Pct Change"] = (stk['Close'].pct_change()) * 100

    return predictprice(12, stkml, stock)


if __name__ == '__main__':
    app.run()
