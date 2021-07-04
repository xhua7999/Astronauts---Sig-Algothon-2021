# An attempt to split the stock data by each stock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Code from eval.py file
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T

# Extract all prices from txt file
pricesFile="./prices250.txt"
prcAll = loadPrices(pricesFile)

prcAll = pd.DataFrame(prcAll).T

# Find the market portfolio - Assume that all stocks have equal market cap
prcAll['avg'] = prcAll.mean(axis=1)
plt.figure("Market Portfolio")
plt.plot(range(len(prcAll['avg'])),prcAll['avg'])
plt.xlabel('Days')
plt.ylabel('Price')

# # Momentum calculation
# def momentum(prices):
#     returns = np.log(prices)
#     x = np.arange(len(returns))
#     slope, _, rvalue, _, _ =linregress(x, returns)
#     return ((1 + slope) ** 252) * (rvalue ** 2)

# # Apply the momentum calculation to all the stocks on a rolling basis of a 30day window
# stockMomentums = prcAll.copy()
# for i in range(100):
#     stockMomentums[i] = prcAll[i].rolling(30).apply(momentum, raw=False)

# plt.figure(figsize=(12, 9))
# plt.xlabel('Market')
# plt.ylabel('Stock Price')

# bests = stockMomentums.max().sort_values(ascending=False).index[:5]
# for best in bests:
#     end = stockMomentums[best].index.get_loc(stockMomentums[best].idxmax())
#     rets = np.log(prcAll[best].iloc[end - 30 : end])
#     x = np.arange(len(rets))
#     slope, intercept, r_value, p_value, std_err = linregress(x, rets)
#     plt.plot(np.arange(60), prcAll[best][end-30:end+30])
#     plt.plot(x, np.e ** (intercept + slope*x))

# plt.show()

# Try another Momentum indicator with a 10 day window
def momentum_difference(prices):
    prices = prices.to_numpy()
    momentum = prices[-1] - prices[0]
    return momentum

simpleMomentums = prcAll.copy()
simpleMomentums = simpleMomentums.drop(simpleMomentums.columns[-1],axis = 1)
for i in range(100):
    simpleMomentums[i] = prcAll[i].rolling(10).apply(momentum_difference, raw=False)

plt.figure("Simple Momentum")
for stock in simpleMomentums:
    plt.plot(range(len(simpleMomentums[stock])),simpleMomentums[stock])
plt.show()

# Calculating the RSI index for each stock 
def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff<0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

RSI = prcAll.copy()
RSI = RSI.drop(RSI.columns[-1],axis = 1)

for i in range(100):
    RSI[i] = computeRSI(prcAll[i], 14)

plt.figure("RSI Chart")
plt.plot(range(len(RSI[17])),RSI[17])
#plt.plot(range(len(prcAll[17])),prcAll[17])
plt.show()