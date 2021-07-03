#!/usr/bin/env python

import numpy as np
from scipy.stats import linregress

nInst=100
currentPos = np.zeros(nInst)

# Dummy algorithm to demonstrate function format.
def getMyPosition (prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape
    rpos = np.array([int(x) for x in 1000 * np.random.randn(nins)])
    currentPos += rpos
    # The algorithm must return a vector of integers, indicating the position of each stock.
    # Position = number of shares, and can be positve or negative depending on long/short position.
    return currentPos

# Function to check that the position does not exceed 10k limit per stock
# Input: currentPos, a 1*100 nparray & latest price of each stock
# Output: True (Valid Position) or False (at least one stock exceeding 10k limit)
def isPositionValid(currentPos, prcSoFar):
    limit = 10000
    result = True

    try:
        lastPrice = prcSoFar[-1,:]
    except:
        # Handle Day 1 when prcSoFar is a 1D array
        lastPrice = prcSoFar

    assert lastPrice.shape == currentPos.shape, "The dimension of the current position does not match data."
    
    positionValue = abs(currentPos)*lastPrice

    stockNumber = 0
    for pos in positionValue:
        if pos > limit:
            result = False
            print(f"Stock {stockNumber}'s position exceeds limit")
        stockNumber += 1
        assert stockNumber < 100, "More stocks than expected."

    return result

# Calculate the momentum for one stock given the stocks historical data
# Code from https://teddykoker.com/2019/05/momentum-strategy-from-stocks-on-the-move-in-python/
# Formula of momentum is the annualised exponential regression slope multiplied
# by the R^2 coefficient of the regression calculation
def momentum(prices):
    returns = np.log(prices)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    return ((1 + slope) ** 252) * (rvalue ** 2)

# Apply the momentum calculation to all the stocks on a rolling basis of a 30day window
