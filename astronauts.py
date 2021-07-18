#!/usr/bin/env python

# RENAME THIS FILE WITH YOUR TEAM NAME.

import numpy as np

nInst=100
currentPos = np.zeros(nInst)

# optimised_wma_window = [16, 10, 13, 21, 45, 56, 26, 30, 22, 8, 16, 13, 9, 17, 22, 18, 21, 22, 38, 18, 14, 59, 18, 47, 17, 16, 15, 17, 18, 22, 19, 19, 10, 15, 18, 14, 19, 19, 29, 21, 16, 13, 16, 41, 20, 28, 13, 27, 23, 14]

# optimised_wma_window2 = [8, 7, 28, 27, 51, 52, 21, 35, 29, 8, 20, 7, 10, 14, 26, 13, 26, 45, 45, 33, 12, 58, 21, 54, 11, 20, 21, 19, 19, 14, 21, 18, 16, 7, 10, 8, 20, 43, 32, 14, 17, 12, 16, 12, 21, 40, 14, 28, 26, 10]

# optimised_powers = [2.250000000000001, 7.500000000000002, 0.3500000000000001, 0.45000000000000007, 0.1, 2.600000000000001, 0.8500000000000002, 0.15000000000000002, 2.950000000000001, 4.200000000000001, 0.8500000000000002, 6.200000000000002, 
# 4.800000000000002, 1.7000000000000006, 0.6500000000000001, 1.2000000000000004, 0.5000000000000001, 0.8500000000000002, 0.6000000000000002, 1.1000000000000005, 2.950000000000001, 2.0500000000000007, 1.2000000000000004, 0.9500000000000003, 3.4000000000000012, 0.8000000000000002, 1.8000000000000007, 1.0500000000000003, 1.2500000000000004, 3.7500000000000013, 1.0500000000000003, 2.3000000000000007, 2.100000000000001, 3.450000000000001, 1.1500000000000004, 5.700000000000001, 0.8500000000000002, 0.9000000000000002, 0.30000000000000004, 2.3000000000000007, 1.6000000000000005, 2.900000000000001, 5.750000000000002, 1.9500000000000006, 1.0000000000000004, 1.3500000000000005, 1.0000000000000004, 0.3500000000000001, 0.25000000000000006, 2.350000000000001]
# optimised_powers = np.array(optimised_powers)

parameters = {
    "weighted_average_window_size": 19,
    "weighted_average_power": 2.109999999999999,
    "required_edge": 0.005
}

# parameters["weighted_average_window_size"] = 19
# parameters["weighted_average_power"] = 3.19
# parameters["momentum_window_size"] = 0
# parameters["required_edge"] = 0.005

trade_stocks = np.ones((100,))
for i in range(50):
    trade_stocks[i] = 0


def weighted_moving_average(price_data, window_size):
    num_instruments, num_days = price_data.shape
    actual_window_size = min(window_size, num_days)
    
    values = np.zeros((num_instruments, actual_window_size))

    weights = np.linspace(0.01, 1, window_size)
    if actual_window_size != window_size:
        weights = weights[window_size-actual_window_size:]

    weights = np.power(weights, parameters["weighted_average_power"])

    weights = weights / np.sum(weights)

    for i in range(actual_window_size):
        values[:,-(1+i)] = price_data[:,-(1+i)] * weights[-(1+i)]

    return np.sum(values, axis=1)



def getMyPosition(prcSoFar):
    global currentPos

    # Zero out position on first day just in case it hasn't been initialised to 0
    num_instruments, num_days = prcSoFar.shape
    if num_days == 1:
        currentPos = np.zeros(num_instruments)

    # Use a weighted average as our theoretical stock value
    weighted_avg = weighted_moving_average(prcSoFar, parameters["weighted_average_window_size"])

    # Calculate the edge of our determined theoretical price against the actual current price
    weighted_edge = (weighted_avg - prcSoFar[:,-1]) / weighted_avg

    max_min_pos = 2*(currentPos > 0) - 1 + 1*(currentPos == 0)

    # max_min_pos = 1*(weighted_edge > required_edge) - 1*(weighted_edge < -required_edge)

    required_edge = parameters["required_edge"]

    for i in range(len(max_min_pos)):
        if weighted_edge[i] > required_edge:
            max_min_pos[i] = 1
        elif weighted_edge[i] < -required_edge:
            max_min_pos[i] = -1

        # if max_min_pos[i] == 0:
        #     max_min_pos[i] = 1 if currentPos[i] >= 0 else -1


    # for i in range(50):
    #     if momentums[i] > parameters["required_edge"]:
    #         max_min_pos[i] = 1
    #     elif momentums[i] < -parameters["required_edge"]:
    #         max_min_pos[i] = -1
            

    currentPos = (10000/prcSoFar[:,-1]) * max_min_pos
    currentPos *= trade_stocks

    

    # The algorithm must return a vector of integers, indicating the position of each stock.
    # Position = number of shares, and can be positve or negative depending on long/short position.
    return currentPos


# def getMyPosition(prcSoFar):
#     global currentPos
#     (nins,nt) = prcSoFar.shape

#     mov_avg1 = moving_average(prcSoFar, 1)
#     mov_avg2 = moving_average(prcSoFar, 5)
#     mov_avg3 = moving_average(prcSoFar, 10)

#     edge1 = (mov_avg1 - prcSoFar[:,-1]) / prcSoFar[:,-1]
#     edge2 = (mov_avg2 - prcSoFar[:,-1]) / prcSoFar[:,-1]
#     edge3 = (mov_avg3 - prcSoFar[:,-1]) / prcSoFar[:,-1]

#     weighted_edge = parameters[0]*edge1 + parameters[1]*edge2 + parameters[2]*edge3
#     weighted_edge /= sum(parameters)

#     required_edge = 0.005

#     max_min_pos = (1*(weighted_edge > required_edge) - 1*(weighted_edge < -required_edge))

#     currentPos = (10000/prcSoFar[:,-1]) * max_min_pos * trade_stocks

#     # The algorithm must return a vector of integers, indicating the position of each stock.
#     # Position = number of shares, and can be positve or negative depending on long/short position.
#     return currentPos


