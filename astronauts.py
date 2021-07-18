#!/usr/bin/env python

import numpy as np

nInst=100
currentPos = np.zeros(nInst)

# Polynomial powers for moving average after optimisation
optimised_powers = [2.250000000000001, 4.000000000000001, 6.300000000000002, 0.6000000000000002, 0.1, 0.45000000000000007, 2.0500000000000007, 2.450000000000001, 0.40000000000000013, 6.200000000000002, 1.2500000000000004, 1.6000000000000005, 7.750000000000002, 2.3000000000000007, 0.6500000000000001, 2.0500000000000007, 1.7500000000000007, 2.350000000000001, 0.8500000000000002, 3.2500000000000013, 3.8500000000000014, 1.5500000000000005, 3.550000000000001, 0.8000000000000002, 5.500000000000002, 2.800000000000001, 1.0500000000000003, 5.650000000000001, 3.2500000000000013, 5.850000000000001, 0.9500000000000003, 2.400000000000001, 3.9500000000000015, 5.050000000000002, 3.2500000000000013, 2.700000000000001, 1.4500000000000006, 0.9500000000000003, 0.30000000000000004, 2.0500000000000007, 4.850000000000001, 7.750000000000002, 6.150000000000001, 4.650000000000001, 0.9000000000000002, 1.3500000000000005, 1.0000000000000004, 3.9000000000000012, 0.30000000000000004, 5.600000000000001]
optimised_powers = np.array(optimised_powers).T

# Parameters used in the model
parameters = {
    "weighted_average_window_size": 19,
    "weighted_average_powers": optimised_powers,
    "required_edge": 0.005
}

# Array containing 1's for the stocks which we will trade and 0's for stocks we won't trade
trade_stocks = np.ones((100,))
for i in range(50):
    trade_stocks[i] = 0


def polynomial_moving_average(price_data, window_size, powers):
    """Calculate the polynomial moving average of an array of price data.
    A polynomial moving average is a weighted moving average where the 
    weights of most recent prices are the highest, and successive weights
    of past prices decay polynomially. The polynomial degree is determined
    by the powers argument."""

    num_instruments, num_days = price_data.shape

    # Initialise an array of linearly increasing weights between 0 and 1
    weights = np.linspace(0.01, 1, window_size)

    # Adjust window size if is larger than the number of days of data
    if num_days < window_size:
        weights = weights[window_size - num_days:]
        window_size = min(window_size, num_days)        

    # Duplicate the weights into a matrix such that there is a row of weights
    # for each instrument
    weights = np.tile(weights, (num_instruments,1))

    # Apply powers to the weights to attain polynomially increasing weights
    weights = np.power(weights, powers.reshape((num_instruments,1)))

    # Normalise each instrument's weight to sum to 1
    weights /= np.sum(weights, axis=1).reshape(num_instruments,1)

    # Multiply the prices by the weights for each day
    values = np.zeros((num_instruments, window_size))
    for i in range(window_size):
        values[:,-(1+i)] = price_data[:,-(1+i)] * weights[:,-(1+i)]

    # Sum the results from each day to get the weighted moving average
    moving_averages = np.sum(values, axis=1)

    return moving_averages


def getMyPosition(prcSoFar):
    
    global currentPos

    # Zero out position on first day just in case it hasn't been initialised to 0
    num_instruments, num_days = prcSoFar.shape
    if num_days == 1:
        currentPos = np.zeros(num_instruments)

    # Only use the last 50 stocks, as we don't trade the first 50
    prcSoFar = prcSoFar[50:]

    # Use a weighted average as our theoretical stock value
    weighted_avg = polynomial_moving_average(prcSoFar, parameters["weighted_average_window_size"], parameters["weighted_average_powers"])

    # Calculate the edge of our determined theoretical price against the actual current price
    weighted_edge = (weighted_avg - prcSoFar[:,-1]) / weighted_avg

    # This array represents our current position in discrete states, where each array value
    # can be -1, 0, 1 representing short, zero and long positions.
    # This is set here to match the current position.
    max_min_pos = 2*(currentPos[50:] > 0) - 1 + 1*(currentPos[50:] == 0)
    
    # If the edge in either direction crosses our edge threshold, we switch positions
    for i in range(len(max_min_pos)):
        if weighted_edge[i] > parameters["required_edge"]:
            max_min_pos[i] = 1
        elif weighted_edge[i] < -parameters["required_edge"]:
            max_min_pos[i] = -1      

    # Convert the short/zero/long array to a position for the last 50 stocks
    currentPos[50:] = (10000/prcSoFar[:,-1]) * max_min_pos
    
    # Zero out stocks which we aren't trading
    currentPos *= trade_stocks

    return currentPos
