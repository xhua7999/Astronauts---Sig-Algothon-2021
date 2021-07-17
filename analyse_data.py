import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import random

df = pd.read_csv('prices250.txt', sep='     ', header=None)
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



def plot_df(df):
    for i in range(df.shape[1]):
        data = df.iloc[:,i]
        plt.plot(data)
    plt.show()


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


index_set = {1, 2, 3, 4, 5}

# percent_df = filter_stocks(percent_df, keep_function=(lambda arr: random.randint(1,30) == 1), index_set=index_set)

percent_df = filter_stocks(percent_df, keep_function=None, index_set=index_set)

plot_df(percent_df)

filtered_percent_df = gaussian_filter(percent_df, 10)
plot_df(filtered_percent_df)

price_change_df = get_price_change(percent_df)
plot_df(price_change_df)

# means = get_mean_price(price_change_df)

# def total_price_change(price_change_df):
    

# print(np.mean(means))


# std_devs = get_std_dev(percent_df)
# mean_prices = get_mean_price(df)

# plt.scatter(mean_prices, std_devs)
# plt.xlabel('Average price of stock in dollars')
# plt.ylabel('Standard deviation in percentage increase from initial price')
# plt.show()