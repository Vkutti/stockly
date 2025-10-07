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
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import math
import gc
import mimetypes
from sklearn.metrics import mean_absolute_error, r2_score

# from tensorflow.python.keras.utils.generic_utils import to_list


app = Flask(__name__)
mimetypes.add_type('image/svg+xml', '.svg')


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
    stocklist['SMA_5'] = stocklist['Close'].rolling(window=5).mean()
    stocklist['SMA_7'] = stocklist['Close'].rolling(window=7).mean()
    stocklist['SMA_10'] = stocklist['Close'].rolling(window=10).mean()
    stocklist['SMA_15'] = stocklist['Close'].rolling(window=15).mean()
    stocklist['SMA_20'] = stocklist['Close'].rolling(window=20).mean()


    # Drop rows with NaN values
    stocklist = stocklist.dropna()
    # Check if we have enough data after dropping NaN
    if len(stocklist) < num + 9:
        return render_template("error.html", msg="Not enough data after calculating rolling averages.")

    features = ["Open", "High", "Close"]
    stkc = stocklist[features]
    close_index = features.index("Close")

    # stkclist = stkc.tolist()
    # stkclist.append("0")

    stkdata = numpy.array(stkc).reshape(-1, 1)
    train_len = math.ceil(len(stkdata) * 0.9)

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_data = stkdata[0:train_len]

    scaled_train_data = scaler.fit_transform(train_data)
    scaled_dataset = scaler.transform(stkdata)


    x_train = []
    y_train = []
    for i in range(num, len(scaled_train_data)):
        print(i)
        x_train.append(scaled_train_data[i - num:i, 0])
        y_train.append(scaled_train_data[i, 0])
        # print(x_train)
        # print(y_train)


    x_train, y_train = numpy.array(x_train), numpy.array(y_train)

    x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    """
    model.add((LSTM(64, return_sequences=False, input_shape=(x_train.shape[1], 1), dropout=0.4)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    """

    model.add((LSTM(64, return_sequences=False, input_shape=(x_train.shape[1], 1), dropout=0.4)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))


    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    model.fit(x_train, y_train, batch_size=8, epochs=16, verbose=0, callbacks=[early_stopping])

    test_data = scaled_dataset[train_len - num:, :]
    x_test = []

    for i in range(num, len(test_data)):
        x_test.append(test_data[i - num:i, 0])

    x_test = numpy.array(x_test)
    x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    predictionfinal = predictions.tolist()

    finalvalue = round(float((predictionfinal[-1])[0]), 2)

    p = f'${finalvalue}'

    actual_prices = stkdata[train_len:, :]  # unscaled true values

    # Ensure equal lengths
    actual_prices = actual_prices[-len(predictions):]

    mae = mean_absolute_error(actual_prices, predictions)
    mape = numpy.mean(numpy.abs((actual_prices - predictions) / actual_prices)) * 100

    print("\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Approx. Accuracy: {100 - mape:.2f}%")

    del model, x_train, y_train, x_test, predictions, scaled_dataset, scaled_train_data, stkdata
    tensorflow.keras.backend.clear_session()

    gc.collect()

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
    stk = stk.history(period="30d")  # or "1y"

    stkml = stk.loc['2025-1-1':].copy()
    print(stkml["Close"])



    return predictprice(2, stkml, stock)


if __name__ == '__main__':
    app.run()
