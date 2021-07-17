#!/usr/bin/env python

# RENAME THIS FILE WITH YOUR TEAM NAME.

import numpy as np

nInst=100
currentPos = np.zeros(nInst)

parameters = [0.1, 0.55, 0.95]

trade_stocks = np.ones(100)

def moving_average(price_data, window_size):
    num_instruments, num_days = price_data.shape
    if num_days < window_size:
        return np.mean(price_data, axis=1)
    else:
        return np.mean(price_data[:,-window_size:], axis=1)


# Dummy algorithm to demonstrate function format.
def getMyPosition(prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape

    mov_avg1 = moving_average(prcSoFar, 1)
    mov_avg2 = moving_average(prcSoFar, 5)
    mov_avg3 = moving_average(prcSoFar, 10)

    # edge1 = (mov_avg1 - prcSoFar[:,-1]) / prcSoFar[:,-1]
    # edge2 = (mov_avg2 - prcSoFar[:,-1]) / prcSoFar[:,-1]
    # edge3 = (mov_avg3 - prcSoFar[:,-1]) / prcSoFar[:,-1]

    edge1 = (mov_avg1 - prcSoFar[:,-1]) / mov_avg1
    edge2 = (mov_avg2 - prcSoFar[:,-1]) / mov_avg2
    edge3 = (mov_avg3 - prcSoFar[:,-1]) / mov_avg3

    weighted_edge = parameters[0]*edge1 + parameters[1]*edge2 + parameters[2]*edge3
    weighted_edge /= sum(parameters)

    required_edge = 0.005

    max_min_pos = 1*(weighted_edge > required_edge) - 1*(weighted_edge < -required_edge)

    currentPos = (10000/prcSoFar[:,-1]) * max_min_pos

    # The algorithm must return a vector of integers, indicating the position of each stock.
    # Position = number of shares, and can be positve or negative depending on long/short position.
    return currentPos

