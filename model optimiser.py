from itertools import combinations
import numpy as np
from eval import loadPrices, calcPL
from Astronauts import parameters


class Optimiser():
    """Optimises a list of parameters to maximise the output of eval_function.
    The output of eval_function should be dependent on the input parameter list."""

    def __init__(self, parameters, eval_function):
        self.parameters = parameters
        self.eval_function = eval_function

    def sweep_single(self, resolution, range):
        pass

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
            print(c, score)
            if score > max_score:
                max_score = score
                max_comb = c

        # # Set parameters to best parameters found
        # for i in range(len(parameters)):
        #     parameters[i] = max_comb[i]

        return max_comb, max_score

    def gradient_ascent(self, step_size):
        pass


pricesFile="./prices250.txt"
prcAll = loadPrices(pricesFile)

eval_function = lambda: calcPL(prcAll)[0]
optimiser = Optimiser(parameters, eval_function)

max_comb, max_score = optimiser.sweep_all(0.05)
print(max_comb, max_score)
