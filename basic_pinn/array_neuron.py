# an array neuron is a single neuron that takes in an array of values and gives an array of values.
# this is useful for when we have multiple inputs and outputs.
# this is necessary as a precursor to the layers of the neural network.

import numpy as np
import random
from activation_functions import ActivationFunctions as af
af = af(None, None, None)  #instantiate the class

x=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) #we create an array of 5 values
t=np.array([0.6, 0.7, 0.8, 0.9, 1.0])

print("size of x: ", np.size(x), "size of t: ", np.size(t)) #to check the size of the arrays

w_x=round(random.random(), 2) #this is not an array because for a single neuron, the same weight is assigned
w_t=round(random.random(), 2)
b=round(random.random(), 2)

# w_x=0.1
# w_t=0.1
# b=0.3

memory={'w_x':w_x, 'w_t':w_t, 'b':b}
print("weights: ", memory)

# all the rounding is done to make the output more readable.
y=np.round(w_x*x+w_t*t+b, 2) #this is the output of the single neuron array
print("unactivated output: ", y)
print("size of y: ", np.size(y))

print("output with linear activation: ", np.round(af.linear(y), 2))
print("output with sigmoid activation: ", np.round(af.sigmoid(y), 2))
print("output with tanh activation: ", np.round(af.tanh(y), 2))
print("output with ReLU activation: ", np.round(af.ReLU(y), 2))
print("output with softplus activation: ", np.round(af.softplus(y), 2))
print("output with swish activation: ", np.round(af.swish(y), 2))
print("output with GELU activation: ", np.round(af.GELU(y), 2))
print("output with sine activation: ", np.round(af.sine(y), 2))
print("output with gaussian activation: ", np.round(af.gaussian(y), 2))