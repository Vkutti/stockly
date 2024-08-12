import yfinance as yf
from matplotlib import pyplot as pp
import datetime as dt
from calendar import monthrange as mr



ones = []
zeros = []

stock = str(input("What stock are you looking for? - "))

downVal = 0.0
upVal = 0.0


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


def getAverageVol(n):
    h = 0

    for i in range(len(n) - 1):
        h += int(n[i])

    print(float(h / (len(n))))


option = str(input("What do you want today? - "))

date = ''
ct = dt.datetime.now()

print(int(list(mr(ct.year, ct.month))[1]) - 7)

if option == 'Tomorrows Price':
    date = f'{ct.year}-{ct.month}-{(ct.day - 7)}'

    if ct.day < 5:
        date = f'{ct.year}-{ct.month - 1}-{int(list(mr(ct.year, ct.month))[1]) - 6}'

if option == 'Future Price':
    date = f'{int(ct.year - 1)}-{ct.month}-{ct.day}'

if option == 'All Time':
    date = '1980-1-1'

stk = yf.Ticker(stock)
stk = stk.history(period="max")
stk = stk.loc[date:].copy()

del stk["Dividends"]
del stk["Stock Splits"]
del stk["High"]
del stk["Low"]

stk["Tomorrow"] = stk["Close"].shift(-1)
stk["Target"] = (stk["Tomorrow"] > stk["Close"]).astype(int)


def getAvgReturn(n):
    h = []
    g = 0
    closeData = list(n["Close"])
    openData = list(n["Open"])

    # print(closeData)
    # print(openData)

    for i in range(len(n)):
        h.append(closeData[i] - openData[i])
        # print(h)

    for x in range(len(h)):
        g += h[x]

    print(g / len(h))




a = list(stk["Target"])
vol = list(stk["Volume"])

returnPercent(a)
getAvgReturn(stk)
getAverageVol(vol)

stk["MA 18 Days"] = stk['Close'].rolling(15).mean()
stk["Pct Change"] = stk['Close'].pct_change()

pp.subplot(221)
stk["Open"].plot(use_index=True)
stk["Close"].plot(use_index=True)
stk["MA 18 Days"].plot(use_index=True)
pp.legend()

pp.subplot(222)
stk["Pct Change"].hist(bins=25)
pp.legend()

pp.suptitle("All Values")

h = 0
print(list(stk["Pct Change"]))
print(len(list(stk["Pct Change"])))

for i in range(1, len(list(stk["Pct Change"]))):
    h += ((stk["Pct Change"].iloc[i]) * 100)

print(h)

avgPCTChange = (h / len(list(stk["Pct Change"])))

print(avgPCTChange)

pp.show()



