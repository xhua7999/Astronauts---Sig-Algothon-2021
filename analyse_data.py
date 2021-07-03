import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('prices250.txt', sep='     ', header=None)
print(df)

def convert_to_percentage_growth(df):
    percent_df = df.copy()
    percent_df.loc[0,:] = np.zeros(len(df.loc[0,:]))
    for i in range(1, df.shape[0]):
        percent_df.loc[i,:] = (df.loc[i,:] - df.loc[0,:]) / df.loc[0,:]
    return percent_df

percent_df = convert_to_percentage_growth(df)
print(percent_df)



# final_return = (df.loc[df.shape[0]-1,:] - df.loc[0,:]) / df.loc[0,:]
# print(final_return)
# print(sum(final_return) / len(final_return))



def plot_df(df):
    for i in range(df.shape[1]):
        data = df.loc[:,i]
        plt.plot(data)
    plt.show()

def get_std_dev(df):
    """Returns a list containing the standard deviations for every stock in the dataframe."""
    std_devs = []
    for i in range(df.shape[1]):
        std_devs.append(np.std(df.loc[:,i].values))
    return std_devs

def get_mean_price(df):
    means = []
    for i in range(df.shape[1]):
        means.append(np.mean(df.loc[:,i].values))
    return means

std_devs = get_std_dev(percent_df)
mean_prices = get_mean_price(df)

plt.scatter(mean_prices, std_devs)
plt.xlabel('Average price of stock in dollars')
plt.ylabel('Standard deviation in percentage increase from initial price')
plt.show()