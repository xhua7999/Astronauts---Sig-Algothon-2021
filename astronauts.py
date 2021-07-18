#!/usr/bin/env python

import numpy as np

nInst=100
currentPos = np.zeros(nInst)

# optimised_wma_window = [30, 41, 18, 42, 20, 47, 56, 44, 34, 7, 53, 49, 57, 32, 56, 53, 51, 44, 35, 36, 35, 52, 31, 59, 25, 49, 51, 24, 36, 24, 53, 46, 48, 26, 34, 42, 59, 56, 40, 33, 58, 36, 26, 28, 52, 58, 37, 15, 51, 32]
# o2 = [19, 19, 19, 10, 20, 19, 19, 19, 19, 9, 19, 19, 19, 13, 19, 19, 15, 19, 19, 19, 19, 21, 19, 19, 19, 23, 14, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 21, 19, 40, 35, 19, 19, 18, 34, 59, 10, 19, 24]

# Polynomial powers for moving average after optimisation
optimised_powers = [2.250000000000001, 4.000000000000001, 6.300000000000002, 0.6000000000000002, 0.1, 0.45000000000000007, 2.0500000000000007, 2.450000000000001, 0.40000000000000013, 6.200000000000002, 1.2500000000000004, 1.6000000000000005, 7.750000000000002, 2.3000000000000007, 0.6500000000000001, 2.0500000000000007, 1.7500000000000007, 2.350000000000001, 0.8500000000000002, 3.2500000000000013, 3.8500000000000014, 1.5500000000000005, 3.550000000000001, 0.8000000000000002, 5.500000000000002, 2.800000000000001, 1.0500000000000003, 5.650000000000001, 3.2500000000000013, 5.850000000000001, 0.9500000000000003, 2.400000000000001, 3.9500000000000015, 5.050000000000002, 3.2500000000000013, 2.700000000000001, 1.4500000000000006, 0.9500000000000003, 0.30000000000000004, 2.0500000000000007, 4.850000000000001, 7.750000000000002, 6.150000000000001, 4.650000000000001, 0.9000000000000002, 1.3500000000000005, 1.0000000000000004, 3.9000000000000012, 0.30000000000000004, 5.600000000000001]
optimised_powers = np.array(optimised_powers).T

# Parameters used in our model and for optimisation
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
    """Calculate the polynomial moving average of an array of price data."""
    num_instruments, num_days = price_data.shape

    weights = np.linspace(0.01, 1, window_size)

    if num_days < window_size:
        weights = weights[window_size - num_days:]
        window_size = min(window_size, num_days)        

    weights = np.tile(weights, (num_instruments,1))
    weights = np.power(weights, powers.reshape((num_instruments,1)))

    weights /= np.sum(weights, axis=1).reshape(num_instruments,1)

    values = np.zeros((num_instruments, window_size))
    for i in range(window_size):
        values[:,-(1+i)] = price_data[:,-(1+i)] * weights[:,-(1+i)]

    return np.sum(values, axis=1)

# def non_vectorised_polynomial_moving_average(price_data, window_size, power):
#     """Calculate the polynomial moving average of an array of price data."""
#     num_instruments, num_days = price_data.shape
    
#     pma_values = np.zeros((num_instruments,1))

#     for i in range(num_instruments):
        
#         window_size = optimised_wma_window[i]
#         weights = np.linspace(0.01, 1, window_size)
#         if num_days < window_size:
#             weights = weights[window_size - num_days:]
#             window_size = min(window_size, num_days)

#         weights = np.power(weights, optimised_powers[i])
#         weights /= np.sum(weights)

#         for j in range(window_size):
#             pma_values[-(1+j)] = price_data[i,-(1+j)] * weights[-(1+j)]


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
