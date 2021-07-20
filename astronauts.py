#!/usr/bin/env python

# from analyse_data import moving_averages
import numpy as np

nInst=100
currentPos = np.zeros(nInst)

# Polynomial powers for moving average after optimisation
optimised_powers = [2.250000000000001, 4.000000000000001, 6.300000000000002, 0.6000000000000002, 0.1, 0.45000000000000007, 2.0500000000000007, 2.450000000000001, 0.40000000000000013, 6.200000000000002, 1.2500000000000004, 1.6000000000000005, 7.750000000000002, 2.3000000000000007, 0.6500000000000001, 2.0500000000000007, 1.7500000000000007, 2.350000000000001, 0.8500000000000002, 3.2500000000000013, 3.8500000000000014, 1.5500000000000005, 3.550000000000001, 0.8000000000000002, 5.500000000000002, 2.800000000000001, 1.0500000000000003, 5.650000000000001, 3.2500000000000013, 5.850000000000001, 0.9500000000000003, 2.400000000000001, 3.9500000000000015, 5.050000000000002, 3.2500000000000013, 2.700000000000001, 1.4500000000000006, 0.9500000000000003, 0.30000000000000004, 2.0500000000000007, 4.850000000000001, 7.750000000000002, 6.150000000000001, 4.650000000000001, 0.9000000000000002, 1.3500000000000005, 1.0000000000000004, 3.9000000000000012, 0.30000000000000004, 5.600000000000001]
optimised_powers = np.array(optimised_powers).T
# Parameters used in the model

mov_avgs = [[45, 70], [13, 70], [3, 98], [13, 66], [49, 70], [17, 56], [39, 50], [47, 90], [27, 36], [27, 58], [39, 76], [29, 86], [33, 98], [5, 22], [31, 92], [13, 48], [41, 90], [5, 22], [19, 28], [9, 32], [33, 66], [49, 82], [15, 82], [13, 36], [29, 74], [43, 52], [19, 44], [47, 98], [27, 50], [27, 90], [27, 64], [17, 36], [31, 40], [7, 98], [19, 52], [11, 28], [13, 84], [19, 98], [17, 56], [37, 50], [45, 92], [5, 34], [29, 60], [33, 70], [31, 92], [33, 98], [3, 58], [7, 40], [13, 48], [43, 94]]
skip = {0,1,2,4,6,10,11,12,14,17,18,19,21,23,24,26,27,28,30,33,35,36,37,39,40,45,46,47}

parameters = {
    "weighted_average_window_size": 19,
    "weighted_average_powers": optimised_powers,
    "required_edge": 0.005
}

# Array containing 1's for the stocks which we will trade and 0's for stocks we won't trade
trade_stocks = np.ones((100,))
for i in range(50):
    if i in skip:
        trade_stocks[i] = 0
# for i in range(50,100):
#     trade_stocks[i] = 0


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


def moving_average(price_data, window_size):
    num_instruments, num_days = price_data.shape
    if num_days < window_size:
        return np.mean(price_data, axis=1)
    else:
        return np.mean(price_data[:,-window_size:], axis=1)


def getMyPosition(prcSoFar):
    global currentPos

    # Zero out position on first day just in case it hasn't been initialised to 0
    num_instruments, num_days = prcSoFar.shape
    if num_days == 1:
        currentPos = np.zeros(num_instruments)


    # First 50 stocks
    first50 = prcSoFar[:50]
    first50_max_min = 2*(currentPos[:50] > 0) - 1 + 1*(currentPos[:50] == 0)


    for i in range(50):
        if i not in skip:
            avg1 = moving_average(first50[i,:].reshape((1,num_days)), mov_avgs[i][0])
            avg2 = moving_average(first50[i,:].reshape((1,num_days)), mov_avgs[i][1])

            if avg1 > avg2:
                first50_max_min[i] = 1
            elif avg2 > avg1:
                first50_max_min[i] = -1

    currentPos[:50] = (10000/first50[:,-1]) * first50_max_min


    # Last 50 stocks
    last50 = prcSoFar[50:]

    # Use a weighted average as our theoretical stock value
    weighted_avg = polynomial_moving_average(last50, parameters["weighted_average_window_size"], parameters["weighted_average_powers"])

    # Calculate the edge of our determined theoretical price against the actual current price
    weighted_edge = (weighted_avg - last50[:,-1]) / weighted_avg

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
    currentPos[50:] = (10000/last50[:,-1]) * max_min_pos
    
    # Zero out stocks which we aren't trading
    currentPos *= trade_stocks

    return currentPos
