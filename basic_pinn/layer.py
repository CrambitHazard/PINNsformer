# a layer is a collection of neurons where each neuron has the same number of inputs,
# but have different weights and biases, and therefore different outputs.

import numpy as np
#import random
from activation_functions import ActivationFunctions as af
af = af(None, None, None)  #instantiate the class

x=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) #no changes here
t=np.array([0.6, 0.7, 0.8, 0.9, 1.0])
print("shape of x: ", np.shape(x), "shape of t: ", np.shape(t))

w_x=np.array([0.5,0.4,0.3,0.2]) #these now become arrays instead of single values
w_t=np.array([1.0,0.9,0.8,0.7])
b=np.array([1.0,0.8,0.6,0.4])
print("shape of w_x: ", np.shape(w_x), "shape of w_t: ", np.shape(w_t), "shape of b: ", np.shape(b))

memory={'w_x':w_x, 'w_t':w_t, 'b':b}
#print("weights: ", memory)

# y=np.round(w_x*x+w_t*t+b, 2) #this is the output of the layer but, we cant use this directly because the shapes are not compatible.
y=np.round(w_x[None, :] * x[:, None] + w_t[None, :] * t[:, None] + b[None, :],2) #this is the correct way to do it.
print("shape of y: ", np.shape(y))
print("unactivated output: ", y)

print("output with linear activation: ", np.round(af.linear(y), 2))
print("output with sigmoid activation: ", np.round(af.sigmoid(y), 2))
print("output with tanh activation: ", np.round(af.tanh(y), 2))
print("output with ReLU activation: ", np.round(af.ReLU(y), 2))
print("output with softplus activation: ", np.round(af.softplus(y), 2))
print("output with swish activation: ", np.round(af.swish(y), 2))
print("output with GELU activation: ", np.round(af.GELU(y), 2))
print("output with sine activation: ", np.round(af.sine(y), 2))
print("output with gaussian activation: ", np.round(af.gaussian(y), 2))