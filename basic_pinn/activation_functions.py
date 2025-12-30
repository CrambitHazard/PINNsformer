# activation functions are used to introduce non-linearity into the model.
# these are especially important for PINNs as we need the output function to be doubly differentiable.
# there are many different activation functions.

import numpy as np

class ActivationFunctions:
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative
    
    def linear(self, y):
        z=y
        return z

    def sigmoid(self, y):
        z=1/(1+np.exp(-y))
        return z

    def tanh(self, y):
        z=(np.exp(y)-np.exp(-y))/(np.exp(y)+np.exp(-y))
        return z
    
    def ReLU(self, y):
        z=max(0,y)
        return z

    def softplus(self, y):
        z=np.log(1+np.exp(y)) #natural logarithm
        return z

    def swish(self, y):
        z=y*self.sigmoid(y)
        return z

    def GELU(self, y):
        z=0.5*y*(1+self.tanh(np.sqrt(2/np.pi)*(y+0.044715*np.power(y,3))))
        return z
    
    def sine(self, y):
        z=np.sin(y)
        return z

    def gaussian(self, y): #also called RBF
        z=np.exp(-y**2)
        return z

    
    