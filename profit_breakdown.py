import numpy as np
from eval import loadPrices, calcPL
from Astronauts import parameters, trade_stocks


pricesFile="./prices500.txt"
prcAll = loadPrices(pricesFile)

eval_function = lambda: calcPL(prcAll)[0]

profits = []

for i in range(100):
    trade_stocks[i] = 0

for i in range(100):
    # trade_stocks[i] = 1 if i >= 50 else -1
    trade_stocks[i] = 1
    profit = eval_function()
    profits.append(profit)
    print(i, profit)
    trade_stocks[i] = 0

print(sum(profits))

print(sum(profits[:50]))
print(sum(profits[50:]))

# eval_function = lambda: calcPL(prcAll)[0]
# optimiser = Optimiser(parameters, eval_function)

# max_comb, max_score = optimiser.sweep_all(0.1)
# print(max_comb, max_score)
