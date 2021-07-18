#!/usr/bin/env python

# RENAME THIS FILE WITH YOUR TEAM NAME.

import numpy as np

nInst=100
currentPos = np.ones(nInst)

# TODO: ADD REQUIRED EDGE TO PARAM OPTIMISATION


optimised_wma_window = [16, 10, 13, 21, 45, 56, 26, 30, 22, 8, 16, 13, 9, 17, 22, 18, 21, 22, 38, 18, 14, 59, 18, 47, 17, 16, 15, 17, 18, 22, 19, 19, 10, 15, 18, 14, 19, 19, 29, 21, 16, 13, 16, 41, 20, 28, 13, 27, 23, 14]

parameters = [0.1, 0.55, 0.95, 19, 10, 0.005]

# optimsation_parameters = {
#     "weighted_average_window_size": 19,
#     "momentum_window_size": 10,
#     "required_edge": 0.05
# }

trade_stocks = np.ones((100,))
for i in range(50):
    trade_stocks[i] = 0

buy_sell_history = []


def moving_average(price_data, window_size):
    num_instruments, num_days = price_data.shape
    if num_days < window_size:
        return np.mean(price_data, axis=1)
    else:
        return np.mean(price_data[:,-window_size:], axis=1)


last_weights = []

def weighted_moving_average(price_data, window_size):
    num_instruments, num_days = price_data.shape
    actual_window_size = min(window_size, num_days)
    
    values = np.zeros((num_instruments, actual_window_size))

    weights = np.linspace(0.01, 1, window_size)
    if actual_window_size != window_size:
        weights = weights[window_size-actual_window_size:]
    weights = weights / np.sum(weights)

    # global last_weights
    # if len(weights) != len(last_weights) or (weights - last_weights).any():
    #     print(weights)
    # last_weights = weights.copy()

    for i in range(actual_window_size):
        values[:,-(1+i)] = price_data[:,-(1+i)] * weights[-(1+i)]

    return np.sum(values, axis=1)


# momentum_window_size = 10
def get_momentums(price_data):
    num_instruments, num_days = price_data.shape
    actual_window_size = min(parameters[4], num_days-1)
    price_diffs = np.zeros((num_instruments, actual_window_size))
    for i in range(actual_window_size):
        price_diffs[:,-(i+1)] = (price_data[:,-(i+1)] - price_data[:,-(i+2)]) / price_data[:,-(i+2)]

    # weights = np.linspace(0, 1, actual_window_size)
    # weights /= np.sum(weights)

    # for i in range(actual_window_size):
    #     price_diffs[:,i] *= weights[i]

    # momentum = np.sum(price_diffs, axis=1)
    momentum = np.mean(price_diffs, axis=1)

    return momentum




def get_price_change(df):
    new_df = df.copy()
    new_df.iloc[0,:] = np.zeros(df.shape[1])
    for i in range(1, df.shape[0]):
        new_df.iloc[i,:] = df.iloc[i,:] - df.iloc[i-1,:]
    return new_df


def getMyPosition(prcSoFar):
    global currentPos
    (nins,nt) = prcSoFar.shape

    # momentum = get_momentums(prcSoFar)
    momentum = 0

    weighted_avg = (1 + momentum) * weighted_moving_average(prcSoFar, parameters[3])

    weighted_edge = (weighted_avg - prcSoFar[:,-1]) / weighted_avg

    required_edge = 0.005
    required_edge = parameters[5]


    max_min_pos = 1*(weighted_edge > required_edge) - 1*(weighted_edge < -required_edge)

    currentPos = (10000/prcSoFar[:,-1]) * max_min_pos

    currentPos = currentPos * trade_stocks

    # The algorithm must return a vector of integers, indicating the position of each stock.
    # Position = number of shares, and can be positve or negative depending on long/short position.
    return currentPos


def get_momentum(prcSoFar):
    try:
        lastPrice = prcSoFar[:,-1]
        secondLastPrice = prcSoFar[:,-2]
        thirdLastPrice = prcSoFar[:,-3]
        fourthLastPrice = prcSoFar[:,-4]
    except:
        # Handle Day 1 when prcSoFar is a 1D array
        lastPrice = prcSoFar
        secondLastPrice = prcSoFar
        thirdLastPrice = prcSoFar
        fourthLastPrice = prcSoFar
    
    gradientprev1 = (lastPrice - secondLastPrice)/secondLastPrice
    gradientprev2 = (secondLastPrice - thirdLastPrice)/thirdLastPrice
    gradientprev3 = (thirdLastPrice - fourthLastPrice)/fourthLastPrice
    
    momentum = 0.5* gradientprev1 + 0.3* gradientprev2 + 0.2 *gradientprev3

    return momentum