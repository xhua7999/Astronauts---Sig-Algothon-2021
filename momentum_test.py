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
    return df

# Extract all prices from txt file
pricesFile="./prices250.txt"
prcAll = loadPrices(pricesFile)

# Momentum calculation
def momentum(prices):
    returns = np.log(prices)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    return ((1 + slope) ** 252) * (rvalue ** 2)

# Apply the momentum calculation to all the stocks on a rolling basis of a 30day window
stockMomentums = prcAll.copy()
for i in range(100):
    stockMomentums[i] = prcAll[i].rolling(30).apply(momentum, raw=False)

plt.figure(figsize=(12, 9))
plt.xlabel('Days')
plt.ylabel('Stock Price')

bests = stockMomentums.max().sort_values(ascending=False).index[:10]
for best in bests:
    end = stockMomentums[best].index.get_loc(stockMomentums[best].idxmax())
    rets = np.log(prcAll[best].iloc[end - 30 : end])
    x = np.arange(len(rets))
    slope, intercept, r_value, p_value, std_err = linregress(x, rets)
    plt.plot(np.arange(60), prcAll[best][end-30:end+30])
    plt.plot(x, np.e ** (intercept + slope*x))

plt.show()

