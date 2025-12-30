# a loss function is a function that takes in the predicted output and the true output and returns a loss.
# a loss is a measure of how good a model is
# for now, we will only implement mean squared error loss function.

import numpy as np

class LossFunctions:
    def __init__(self, name, function):
        self.name = name
        self.function = function
    
    def mse(self, y_real, y_pred):
        return np.mean((y_real - y_pred)**2)
    