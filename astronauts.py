#!/usr/bin/env python
import math
import numpy as np
import pandas as pd
from scipy.stats import linregress

nInst=100
currentPos = np.zeros(nInst)

# A basic theoretical price function
def getTheoreticalPrices(prcSoFar):
    # an array containing the latest prices of each stockie (I DONT KNOW HOW TRY AND EXCEPT WORK!)
    try:
        lastPrice = prcSoFar[:,-1]
        secondLastPrice = prcSoFar[:,-2]
    except:
        # Handle Day 1 when prcSoFar is a 1D array
        lastPrice = prcSoFar
        secondLastPrice = prcSoFar
    
    rsi_values = applyRSIModel(prcSoFar)
    weightfactor = 1/3000  
    
    # The theoretical price is the last price with a basic mean reversion - 0.5 of the way to the 2nd last price
    theoreticalPrices = lastPrice + (secondLastPrice - lastPrice) * 0.5 \
    + (50*np.ones(100) - rsi_values) * lastPrice * weightfactor
    return theoreticalPrices


# Applys the edge model to determine how much we should size up/down for each stockie
def applyEdgeModel(prcSoFar, theoPrices, currentPos): # theoPrices is a 100 length array with the edge for each

    # an array containing the latest prices of each stockie (I DONT KNOW HOW TRY AND EXCEPT WORK!)
    try:
        lastPrice = prcSoFar[:,-1]
    except:
        # Handle Day 1 when prcSoFar is a 1D array
        lastPrice = prcSoFar

    # Our edge is the theoretical price minus the current price expressed as a percentage (positive means buy)
    edgeArray = (theoPrices - lastPrice)/lastPrice
    newPosition = currentPos
    edgeRequired = 0.01 # if we have 1% of edge we will max out long and short based on direction
    for i in range(len(edgeArray)):
        edge = edgeArray[i]
        if (edge > edgeRequired).any():
            newPosition[i] = math.floor(10000/lastPrice[i]) # We buy as much as we allowed! Everything!
        if (edge < -edgeRequired).any():
            newPosition[i] = math.ceil(-10000/lastPrice[i]) # We sell as much as we allowed! Everything!
    return newPosition

# Function to change current position based on the results of the RSI Model
# Calculates the Relative Strength Index and uses the last RSI to determine the current stock position
def applyRSIModel(prcSoFar):
    
    # RSI Method
    prcSoFar = pd.DataFrame(prcSoFar).T
    RSI = prcSoFar.copy()
    for i in range(100):
        RSI[i] = computeRSI(prcSoFar[i], 14)
        RSI_copy = RSI.fillna(0).to_numpy()
    assert len(RSI_copy[-1]) == 100, "Position is incorrect"
    return np.transpose(RSI_copy[-1])

# Dummy algorithm to demonstrate function format.
def getMyPosition(prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape
    #rpos = np.array([int(x) for x in 1000 * np.random.randn(nins)])
    #currentPos += rpos

    #currentPos = applyRSIModel(prcSoFar, currentPos)
    
    theoPrices = getTheoreticalPrices(prcSoFar)
    currentPos = applyEdgeModel(prcSoFar, theoPrices, currentPos)
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
