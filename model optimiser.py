from itertools import combinations
import numpy as np
from eval import loadPrices, calcPL
from Astronauts import parameters, trade_stocks


class Optimiser():
    """Optimises a list of parameters to maximise the output of eval_function.
    The output of eval_function should be dependent on the input parameter list."""

    def __init__(self, parameters, eval_function):
        self.parameters = parameters
        self.eval_function = eval_function

    def sweep_single(self, resolution, sweep_range, parameter_index):
        sweep_values = np.arange(sweep_range[0], sweep_range[1], resolution)
        
        max_score = -float('inf')
        max_value = None

        for value in sweep_values:
            self.parameters[parameter_index] = value
            score = self.eval_function()
            if score > max_score:
                max_score = score
                max_value = value
            # print(value, score)

        return max_value, max_score

    def sweep_all(self, resolution):
        """Evaluates through all parameter combinations with a specified resolution, where
        each parameter is varied between 0 and 1. Returns the maximum score achieved and
        the combination which resulted in that score."""
        
        max_score = -float('inf')
        max_comb = None

        values = [resolution * i for i in range(int(1/resolution))]
        combs = combinations(values, len(parameters))
        for c in combs:
            for i in range(len(parameters)):
                parameters[i] = c[i]
            score = self.eval_function()
            
            
            if score > max_score:
                max_score = score
                max_comb = c

                print(int(score), c)

        # # Set parameters to best parameters found
        # for i in range(len(parameters)):
        #     parameters[i] = max_comb[i]

        return max_comb, max_score

    def gradient_ascent(self, step_size):
        pass

    def greedy_sweep(self, resolution):
        pass



pricesFile="./prices500.txt"
prcAll = loadPrices(pricesFile)

eval_function = lambda: calcPL(prcAll)[0]
optimiser = Optimiser(parameters, eval_function)

# max_comb, max_score = optimiser.sweep_all(0.1)
# print(max_comb, max_score)

# for i in range(50):
#     trade_stocks[i] = 0

# for i in range(50,100):
#     trade_stocks[i] = 0

# max_comb, max_score = optimiser.sweep_single(1, (6, 50), 4)
# max_comb, max_score = optimiser.sweep_single(0.001, (0, 0.15), 5)
# print(max_comb, max_score)

for i in range(100):
    trade_stocks[i] = 0

profits = []
max_param = []
for i in range(50, 100):
    # trade_stocks[i] = 1 if i >= 50 else -1
    trade_stocks[i] = 1
    max_comb, max_score = optimiser.sweep_single(1, (1, 60), 3)
    profits.append(max_score)
    max_param.append(max_comb)
    print(i, max_comb, max_score)
    trade_stocks[i] = 0

print("TOTAL: ", sum(profits))
print(max_param)