# In a neural network, we have multiple layers of neurons,
# each with different weights and biases.
# each layer is allowed to have different number of neurons.

import numpy as np
from activation_functions import ActivationFunctions as af
af = af(None, None, None)  #instantiate the class

x=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) #no changes here
t=np.array([0.6, 0.7, 0.8, 0.9, 1.0])

w_x1=np.array([0.3,-0.2,0.4,-0.3]) #these are the same, but these are only the biases from input to first layer
w_t1=np.array([0.4,0.3,-0.2,0.2])
b1=np.array([0.1,-0.1,0.2,-0.2])

# if the shape of inputs is (n,), then the shape of the biases from input to first layer has to be (,m1)
# now after the first layer computation is done, the shape of output of layer 1 is (n,m1)
# so the input to layer 2 is of shape (n,m1)
# thus we have to make the shape of weights from layer 1 to layer 2 as (m1,m2) and biases as (m2,)
# this will make the shape of output of layer 2 as (n,m2)

# w_x2=np.array([[0.5,0.4,0.3,0.2],[0.1,0.2,0.3,0.4],[0.2,0.2,0.2,0.2]])  #this is wrong because after the input layer we dont care about x and t
# w_t2=np.array([[1.0,0.9,0.8,0.7],[0.5,0.6,0.7,0.8],[0.3,0.2,0.2,0.3]])  #so a single array of w should be used
b2=np.array([0.1,-0.1,0.05])
# print("shape of w_x2: ", np.shape(w_x2), "shape of w_t2: ", np.shape(w_t2), "shape of b2: ", np.shape(b2))

w2=np.array([[0.3,-0.2,0.4,-0.1],[-0.2,0.3,0.1,0.3],[0.1,-0.1,0.2,-0.2]]) 
print("shape of w2: ", np.shape(w2))

# first let's compute the output of layer 1
y1=np.round(w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :],2)
print("shape of y1: ", np.shape(y1))
print("output of layer 1: ", y1)

# now we apply the activation functions to the output of layer 1, lets use tanh activation function for this
y1_activated=np.round(af.tanh(y1), 2)
print("shape of y1_activated: ", np.shape(y1_activated))
print("output of layer 1 with tanh activation: ", y1_activated)

# now let's compute the output of layer 2
#y2=np.round(np.matmul(y1, w_x2) + np.matmul(y1, w_t2) + b2[None, :],2) 
y2=np.round(np.matmul(y1_activated, w2.T) + b2[None, :],2) #this is the correct way to do it.
print("shape of y2: ", np.shape(y2))
print("output of layer 2: ", y2)

# now we apply the activation functions to the output of layer 2, lets use tanh activation function for this
y2_activated=np.round(af.tanh(y2), 2)
print("shape of y2_activated: ", np.shape(y2_activated))
print("output of layer 2 with tanh activation: ", y2_activated)

# now we have the output of the 2 layer neural network
print("output of the neural network: ", y2_activated)