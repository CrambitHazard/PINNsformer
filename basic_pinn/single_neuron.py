# a single neuron is basically a function that takes in paramaters, 
# assigns weights to the parameters, adds a bias, gives the output, 
# then uses an activation function to give final output.

import numpy as np #required for all the maths
import random #required to assign random weights to the parameters and weights and biases
from activation_functions import ActivationFunctions as af
af = af(None, None, None)  #instantiate the class

# x and t are the two params chosen for this. 
# the params can be anything, but for PINNs they are generally space and time.
x=round(random.random(), 2) #rounding to 2 decimal places to make it easier to read
t=round(random.random(), 2) 

# x=0.6 #can be set to any value
# t=0.8 #can be set to any value

#print(x,t)

w_x=round(random.random(), 2)
w_t=round(random.random(), 2)
b=round(random.random(), 2)

memory={'w_x':w_x, 'w_t':w_t, 'b':b}
#print(memory)

y=w_x*x+w_t*t+b #this is the output of the single neuron
#print(y)

print("input params: ", x, t)
print("weights: ", memory)
print("output with linear activation: ", af.linear(y))
print("output with sigmoid activation: ", af.sigmoid(y))
print("output with tanh activation: ", af.tanh(y))
print("output with ReLU activation: ", af.ReLU(y))
print("output with softplus activation: ", af.softplus(y))
print("output with swish activation: ", af.swish(y))
print("output with GELU activation: ", af.GELU(y))
print("output with sine activation: ", af.sine(y))
print("output with gaussian activation: ", af.gaussian(y))
