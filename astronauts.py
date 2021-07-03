#!/usr/bin/env python

import numpy as np
from scipy.stats import linregress

nInst=100
currentPos = np.zeros(nInst)

# A basic theoretical price function
def getTheoreticalPrices (prcSoFar):
    # an array containing the latest prices of each stockie (I DONT KNOW HOW TRY AND EXCEPT WORK!)
    try:
        lastPrice = prcSoFar[-1,:]
        secondLastPrice = prcSoFar[-2,:]
    except:
        # Handle Day 1 when prcSoFar is a 1D array
        lastPrice = prcSoFar

    # The theoretical price is the last price with a basic mean reversion - 0.5 of the way to the 2nd last price
    theoreticalPrices = lastPrice + (secondLastPrice - lastPrice) * 0.5
    return theoreticalPrices


# Applys the edge model to determine how much we should size up/down for each stockie
def applyEdgeModel (prcSoFar, theoPrices, currentPos): # theoPrices is a 100 length array with the edge for each

    # an array containing the latest prices of each stockie (I DONT KNOW HOW TRY AND EXCEPT WORK!)
    try:
        lastPrice = prcSoFar[-1,:]
    except:
        # Handle Day 1 when prcSoFar is a 1D array
        lastPrice = prcSoFar

    # Our edge is the theoretical price minus the current price expressed as a percentage (positive means buy)
    edgeArray = (theoPrices - lastPrice)/lastPrice
    newPosition = currentPos
    edgeRequired = 0.01 # if we have 1% of edge we will max out long and short based on direction
    for i in edgeArray:
        if edgeArray[i] > edgeRequired:
            newPosition[i] = floor(10000/lastPrice[i]) # We buy as much as we allowed! Everything!
        if edgeArray[i] < -edgeRequired:
            newPosition[i] = ceil(-10000/lastPrice[i]) # We sell as much as we allowed! Everything!
    return newPosition

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
