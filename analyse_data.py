import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import random
from Astronauts import moving_average
from eval import loadPrices, calcPL


df = pd.read_csv('prices500.txt', sep='     ', header=None)
print(df)

NUM_DAYS = df.shape[0]
NUM_STOCKS = df.shape[1]

def convert_to_percentage_growth(df):
    percent_df = df.copy()
    percent_df.iloc[0,:] = np.zeros(len(df.iloc[0,:]))
    for i in range(1, df.shape[0]):
        percent_df.iloc[i,:] = (df.iloc[i,:] - df.iloc[0,:]) / df.iloc[0,:]
    return percent_df

percent_df = convert_to_percentage_growth(df)
print(percent_df)



# final_return = (df.iloc[df.shape[0]-1,:] - df.iloc[0,:]) / df.iloc[0,:]
# print(final_return)
# print(sum(final_return) / len(final_return))



def plot_df(df, params=''):
    for i in range(df.shape[1]):
        data = df.iloc[:,i]
        plt.plot(data, params)
    


def get_std_dev(df):
    """Returns a list containing the standard deviations for every stock in the dataframe."""
    std_devs = []
    for i in range(df.shape[1]):
        std_devs.append(np.std(df.iloc[:,i].values))
    return std_devs


def get_mean_price(df):
    means = []
    for i in range(df.shape[1]):
        means.append(np.mean(df.iloc[:,i].values))
    return means


def gaussian_filter(df, std_dev):
    new_df = df.copy()
    for i in range(df.shape[1]):
        new_df.iloc[:,i] = gaussian_filter1d(df.iloc[:,i], std_dev)
    return new_df


def get_price_change(df):
    new_df = df.copy()
    new_df.iloc[0,:] = np.zeros(df.shape[1])
    for i in range(1, df.shape[0]):
        new_df.iloc[i,:] = df.iloc[i,:] - df.iloc[i-1,:]
    return new_df


def filter_stocks(df, keep_function=None, index_set=None):
    drop_columns = []
    for i in range(df.shape[1]):
        if keep_function is not None and not keep_function(df.iloc[:,i].values):
            drop_columns.append(i)
        if index_set is not None and i not in index_set:
            drop_columns.append(i)
    return df.drop(drop_columns, axis=1)






# def plot_price_changes()

index_set = {8}
# index_set=None

# index_set = set(i for i in range(50))

keep_function = lambda arr: random.randint(1,30) == 1
keep_function=None

percent_df = filter_stocks(percent_df, keep_function=keep_function, index_set=index_set)

# percent_df = filter_stocks(percent_df, keep_function=None, index_set=index_set)

plot_df(percent_df)

# gaussian_percent_df = gaussian_filter(percent_df, 10)
# plot_df(gaussian_percent_df)

# price_change_df = get_price_change(percent_df)
# plot_df(price_change_df)




def plot_price_change(price_change_df):
    stock_names = [str(i) for i in range(100)]
    price_change_means = price_change_df.abs().mean(axis=0)
    plt.bar(stock_names, price_change_means)
    plt.show()

# plot_price_change(price_change_df)

# means = get_mean_price(price_change_df)

# def total_price_change(price_change_df):
    

# print(np.mean(means))


# std_devs = get_std_dev(percent_df)
# mean_prices = get_mean_price(df)

# plt.scatter(mean_prices, std_devs)
# plt.xlabel('Average price of stock in dollars')
# plt.ylabel('Standard deviation in percentage increase from initial price')
# plt.show()


def moving_averages(df, window_size):
    new_df = df.copy()
    print("SHAPE1", df.shape[1])
    for i in range(df.shape[1]):
        new_df.iloc[:,i] = new_df.iloc[:,i].rolling(window=window_size).mean()
    return new_df


def exp_moving_averages(df, window_size):
    new_df = df.copy()
    print("SHAPE1", df.shape[1])
    for i in range(df.shape[1]):
        new_df.iloc[:,i] = new_df.iloc[:,i].ewm(span=window_size,adjust=False).mean()
    return new_df


def momentums(price_data, window_size):
    
    num_instruments, num_days = price_data.shape
    actual_window_size = min(window_size, num_days-1)
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



mov_avg_df = moving_averages(percent_df, window_size=10)
plot_df(mov_avg_df, params='-.')

mov_avg_df = moving_averages(percent_df.multiply(1.005), window_size=10)
plot_df(mov_avg_df, params='-.')

mov_avg_df = moving_averages(percent_df.multiply(0.995), window_size=10)
plot_df(mov_avg_df, params='-.')

mov_avg_df = exp_moving_averages(percent_df, window_size=20)
plot_df(mov_avg_df, params='-.')

plt.show()