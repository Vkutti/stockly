import yfinance as yf
# from matplotlib import pyplot as pp
import datetime as dt
from calendar import monthrange as mr
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pnd

ones = []
zeros = []

stock = str()


# Returns the average percent of rises and falls
def returnPercent(new):
    global ones, zeros

    for i in range(len(new)):
        if new[i] == 1:
            ones.append(new[i])
        else:
            zeros.append(new[i])

    print(ones)
    print(zeros)

    prup = float(len(ones) / len(new)) * 100
    prdown = float(len(zeros) / len(new)) * 100

    print(f'The chance of the price going up tomorrow is {prup}%')
    print(f'The chance of the price going down tomorrow is {prdown}%')


# Returns the average of all the volumes
def getAverageVol(n):
    x = 0

    for i in range(len(n) - 1):
        x += int(n[i])

    print(float(x / (len(n))))


# Returns the average return daily
def getAvgReturn(n):
    x = []
    g = 0

    closedata = list(n["Close"])
    opendata = list(n["Open"])

    # print(closeData)
    # print(openData)

    for i in range(len(n - 1)):
        x.append(closedata[i] - opendata[i])
        # print(h)

    for i in range(len(x)):
        g += x[i]

    return g / (len(x) - 1)


app = Flask(__name__)


@app.route("/")
def start():
    return render_template('homepage.html')


# Function to get if a stock exists on Yahoo Finance
def is_real_stock(new):
    st = yf.Ticker(new)
    hist = st.history()

    if hist.empty:
        return False
    else:
        return True


@app.route("/stockinfo")
def get_stock_name():
    global stock
    stock_name = request.args.get('stockname')
    stock = str(stock_name)

    is_real_stock(stock)

    if is_real_stock(stock):
        stock = stock.upper()
        returnmsg = str(f"{stock} is a real stock.")
    else:
        returnmsg = str(f"{stock} is not a real stock.")

    return render_template("homepage.html", stockPick=stock, returnTF=returnmsg)


@app.route("/dateoption")
def dateoption():
    global stock
    date = str(request.args.get('dateopt'))

    option = date.upper()

    date = ''
    ct = dt.datetime.now()

    # print(int(list(mr(ct.year, ct.month))[1]) - 10)

    if option == 'FUTURE PRICE':
        date = f'{ct.year}-{ct.month}-{(ct.day - 10)}'

        if ct.day < 10:
            date = f'{ct.year}-{ct.month - 1}-{int(list(mr(ct.year, ct.month))[1]) - (10 - ct.day)}'

    if option == 'TOMORROWS PRICE':
        date = f'{int(ct.year - 1)}-{ct.month}-{ct.day}'

    if option == 'ALL TIME':
        date = '1980-1-1'

    stk = yf.Ticker(stock)
    stk = stk.history(period="max")
    stkml = stk.loc['2012-1-1':].copy()
    stk = stk.loc[date:].copy()

    del stk["Dividends"]
    del stk["Stock Splits"]
    # stk["High"]
    # stk["Low"]

    stk["Tomorrow"] = stk["Close"].shift(-1)
    stk["Target"] = (stk["Tomorrow"] > stk["Close"]).astype(int)

    stkml["Tomorrow1"] = stkml["Close"].shift(-1)
    stkml["Target1"] = (stkml["Tomorrow1"] > stkml["Close"]).astype(int)

    targ = list(stk["Target"])
    vol = list(stk["Volume"])

    returnPercent(targ)
    print(getAvgReturn(stk))
    getAverageVol(vol)

    stk["MA 15 Days"] = stk['Close'].rolling(15).mean()

    stk["MA 7 Days"] = stk['Close'].rolling(7).mean()

    stk["Pct Change"] = (stk['Close'].pct_change()) * 100

    stkml["MA 15 Days"] = stkml['Close'].rolling(15).mean()

    """

    pp.subplot(221)
    stk["Open"].plot(use_index=True)
    stk["Close"].plot(use_index=True)

    stk["MA 15 Days"].plot(use_index=True)
    stk["MA 7 Days"].plot(use_index=True)

    pp.legend()

    pp.subplot(222)
    stk["Pct Change"].hist(bins=25)

    pp.suptitle("All Values")

    """

    h = 0

    # Final Calculations
    averagePercentChange(h, stk)

    apc = float(averagePercentChange(h, stk))
    print(f'This is the average percent change: {apc}')

    prup = float(len(ones) / len(targ))
    prdown = float(len(zeros) / len(targ))

    changeValue = float(((stk["Close"].iloc[-1]) * apc) / 100)

    # Average Return - Percent Error <= Change Value <= Average Return + Percent Error

    lowbound = float(changeValue - ((getAvgReturn(stk) * prdown) / 100))
    highbound = float(changeValue + ((getAvgReturn(stk) * prup) / 100))

    changeval = f'This is the change value: ${round(changeValue, 2)}'
    lowchangeval = f'This is the lowest change value: ${round(float(changeValue - ((getAvgReturn(stk) * prdown) / 100)), 2)}'
    highchangeval = f'This is the highest change value: ${round(float(changeValue + ((getAvgReturn(stk) * prup) / 100)), 2)}'

    if lowbound <= changeValue <= highbound:
        trend = 'rise'
        pricing = round(float(changeValue), 2)
        if changeValue < 0:
            trend = 'fall'
            pricing = round(float(changeValue), 2)
    else:
        trend = 'fall'
        pricing = round(float(changeValue * - 1), 2)

    finish = ''

    if option == 'TOMORROWS PRICE':
        finish = f'The stock {stock} is expected to {trend} tomorrow by about ${pricing}'

    if option == 'FUTURE PRICE':
        finish = f'The stock {stock} is expected to {trend} in the next few days by about ${pricing}'

    model = RandomForestClassifier(n_estimators=500, min_samples_split=100, random_state=1)

    train = stkml.iloc[:-100]
    test = stkml.iloc[-100:]

    predictors = ["Open", "Close", "High", "Low", "Volume"]
    model.fit(train[predictors], train["Target1"])



    def predict(train1, test1, predictors1, model1):
        model1.fit(train1[predictors1], train1["Target1"])
        predictions = model1.predict_proba(test1[predictors1])[:, 1]
        predictions[predictions >= 0.6] = 1
        predictions[predictions < 0.6] = 0
        predictions = pnd.Series(predictions, index=test1.index, name="Predictions")
        combined = pnd.concat([test1["Target1"], predictions], axis=1)
        print(combined)
        return combined

    def backtest(data2, model2, predictors2, start=3000, step=250):
        all_predictions = []

        for i in range(start, data2.shape[0], step):
            train2 = data2.iloc[0:i].copy()
            test2 = data2.iloc[i:(i+step)].copy()
            spfcpreds = predict(train2, test2, predictors2, model2)
            all_predictions.append(spfcpreds)

        return pnd.concat(all_predictions)


    horizons = [2, 5, 30, 60, 250, 1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = stkml.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        stkml[ratio_column] = stkml["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        stkml[trend_column] = stkml.shift(1).rolling(horizon).sum()["Target1"]

        new_predictors += [ratio_column, trend_column]

    print(new_predictors)

    preds = backtest(stkml, model, new_predictors)
    ps = precision_score(preds["Target1"], preds["Predictions"])
    print(ps)


    return render_template("homepage.html", date=date, cv=changeval, lcv=lowchangeval, hcv=highchangeval,
                           endval=finish)


# Returns the average percent change daily
def averagePercentChange(a, sk):
    for i in range(1, len(list(sk["Pct Change"]))):
        a += (sk["Pct Change"].iloc[i])

    print(((len(list(sk["Pct Change"]))) - 1))
    avgPercentChange = f'{a / ((len(list(sk["Pct Change"]))) - 1)}% Change Daily'

    print(avgPercentChange)
    return a / ((len(list(sk["Pct Change"]))) - 1)


if __name__ == '__main__':
    app.run()

# option = str(input("What do you want today? - "))

# pp.show()
