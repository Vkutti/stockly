import yfinance as yf
from matplotlib import pyplot as pp
import datetime as dt
from calendar import monthrange as mr

ones = []
zeros = []

stock = str(input("What stock are you looking for? - "))

downVal = 0.0
upVal = 0.0


# Function to get if a stock exists on Yahoo Finance
def is_real_stock(new):
    st = yf.Ticker(new)
    hist = st.history()

    if hist.empty:
        return False
    else:
        return True


if is_real_stock(stock):
    stock = stock.upper()
    print(f"{stock} is a real stock.")
else:
    print(f"{stock} is not a real stock.")


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


# Returns the average percent change daily
def averagePercentChange(a):
    for i in range(1, len(list(stk["Pct Change"]))):
        a += (stk["Pct Change"].iloc[i])
        print(a)

    print(((len(list(stk["Pct Change"]))) - 1))
    avgPercentChange = f'{a / ((len(list(stk["Pct Change"]))) - 1)}% Change Daily'

    print(avgPercentChange)
    return a / ((len(list(stk["Pct Change"]))) - 1)


# Returns the average return daily
def getAvgReturn(n):
    x = []
    g = 0

    closedata = list(n["Close"])
    opendata = list(n["Open"])

    # print(closeData)
    # print(openData)

    for i in range(len(n-1)):
        x.append(closedata[i] - opendata[i])
        # print(h)

    for i in range(len(x)):
        g += x[i]

    print(g / (len(x) - 1))
    return g / (len(x) - 1)


option = str(input("What do you want today? - "))

date = ''
ct = dt.datetime.now()

print(int(list(mr(ct.year, ct.month))[1]) - 14)

if option == 'Tomorrows Price':
    date = f'{ct.year}-{ct.month}-{(ct.day - 14)}'

    if ct.day < 14:
        date = f'{ct.year}-{ct.month - 1}-{int(list(mr(ct.year, ct.month))[1]) - 13}'

if option == 'Future Price':
    date = f'{int(ct.year - 1)}-{ct.month}-{ct.day}'

if option == 'All Time':
    date = '1980-1-1'

stk = yf.Ticker(stock)
stk = stk.history(period="max")
stk = stk.loc[date:].copy()

del stk["Dividends"]
del stk["Stock Splits"]
# stk["High"]
# stk["Low"]

stk["Tomorrow"] = stk["Close"].shift(-1)
stk["Target"] = (stk["Tomorrow"] > stk["Close"]).astype(int)

targ = list(stk["Target"])
vol = list(stk["Volume"])

returnPercent(targ)
getAvgReturn(stk)
getAverageVol(vol)


stk["MA 15 Days"] = stk['Close'].rolling(15).mean()

stk["MA 7 Days"] = stk['Close'].rolling(7).mean()

stk["Pct Change"] = (stk['Close'].pct_change()) * 100

pp.subplot(221)
stk["Open"].plot(use_index=True)
stk["Close"].plot(use_index=True)

stk["MA 15 Days"].plot(use_index=True)
stk["MA 7 Days"].plot(use_index=True)

pp.legend()

pp.subplot(222)
stk["Pct Change"].hist(bins=25)
pp.legend()

pp.suptitle("All Values")

h = 0

print(len(list(stk["Pct Change"])))

averagePercentChange(h)

apc = float(averagePercentChange(h))

# Final Calculations

changeValue = float(((stk["Close"].iloc[-1]) * apc) / 100)
print(changeValue)

if changeValue >= getAvgReturn(stk):
    print("Yes")

pp.show()


