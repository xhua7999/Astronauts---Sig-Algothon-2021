from itertools import combinations
import numpy as np
from eval import loadPrices, calcPL, prcAll
from Astronauts import parameters, trade_stocks


class Optimiser():
    """Optimises a list of parameters to maximise the output of eval_function.
    The output of eval_function should be dependent on the input parameter list."""

    def __init__(self, parameters, eval_function):
        self.parameters = parameters
        self.eval_function = eval_function

    def sweep_single(self, resolution, sweep_range, parameter_name):
        sweep_values = np.arange(sweep_range[0], sweep_range[1], resolution)
        
        max_score = -float('inf')
        max_value = None

        for value in sweep_values:
            self.parameters[parameter_name] = value
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
            for param_name in parameters:
                parameters[param_name] = c[i]
            score = self.eval_function()
            
            if score > max_score:
                max_score = score
                max_comb = c

                print(int(score), c)

        return max_comb, max_score

    def gradient_ascent(self, step_size):
        pass

    def greedy_sweep(self, resolution):
        pass



# Set up evaluation function and optimiser
eval_function = lambda: calcPL(prcAll)[0]
optimiser = Optimiser(parameters, eval_function)

# Set up default parameters
parameters["weighted_average_window_size"] = 19
parameters["weighted_average_power"] = 3.19
parameters["momentum_window_size"] = 0
parameters["required_edge"] = 0.005

# print("eval", eval_function())

# max_comb, max_score = optimiser.sweep_all(0.1)
# print(max_comb, max_score)

# for i in range(50):
#     trade_stocks[i] = 0

# # for i in range(50,100):
# #     trade_stocks[i] = 0

max_comb, max_score = optimiser.sweep_single(1, (2, 40), "weighted_average_window_size")

# max_comb, max_score = optimiser.sweep_single(0.001, (0, 0.1), "required_edge")
# max_comb, max_score = optimiser.sweep_single(0.01, (0.1, 5), "weighted_average_power")
# max_comb, max_score = optimiser.sweep_single(1, (0, 20), "momentum_window_size")

print(max_comb, max_score)



# for i in range(100):
#     trade_stocks[i] = 0

# profits = []
# max_param = []
# # for i in range(50):
# for i in range(50, 100):
#     trade_stocks[i] = 1
    
#     # max_comb, max_score = optimiser.sweep_single(0.005, (0, 0.2), "required_edge")
#     # max_comb, max_score = optimiser.sweep_single(1, (3, 60), "weighted_average_window_size")
#     # max_comb, max_score = optimiser.sweep_single(0.05, (0.1, 8), "weighted_average_power")
#     max_comb, max_score = optimiser.sweep_single(1, (0, 1), "momentum_window_size")

#     profits.append(max_score)
#     max_param.append(max_comb)
#     print(i, max_comb, max_score)
#     trade_stocks[i] = 0

# print("TOTAL: ", sum(profits))
# print(max_param)