# a loss function is a function that takes in the predicted output and the true output and returns a loss.
# a loss is a measure of how good a model is
# for now, we will only implement mean squared error loss function.

# using the same 2 layer neural network from network.py

import numpy as np
from activation_functions import ActivationFunctions as af
af = af(None, None, None)

x=np.array([0.1, 0.2, 0.3, 0.4, 0.5])
t=np.array([0.6, 0.7, 0.8, 0.9, 1.0])

w_x1=np.array([0.3,-0.2,0.4,-0.3])
w_t1=np.array([0.4,0.3,-0.2,0.2])
b1=np.array([0.1,-0.1,0.2,-0.2])

b2=np.array([0.1,-0.1,0.05])

w2=np.array([[0.3,-0.2,0.4,-0.1],[-0.2,0.3,0.1,0.3],[0.1,-0.1,0.2,-0.2]]) 
print("shape of w2: ", np.shape(w2))

y1=np.round(w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :],2)
print("shape of y1: ", np.shape(y1))
print("output of layer 1: ", y1)

y1_activated=np.round(af.tanh(y1), 2)
print("shape of y1_activated: ", np.shape(y1_activated))
print("output of layer 1 with tanh activation: ", y1_activated)

y2=np.round(np.matmul(y1_activated, w2.T) + b2[None, :],2)
print("shape of y2: ", np.shape(y2))
print("output of layer 2: ", y2)

y2_activated=np.round(af.tanh(y2), 2)
print("shape of y2_activated: ", np.shape(y2_activated))
print("output of layer 2 with tanh activation: ", y2_activated)

# we will create a third layer with only one output neuron
# this is done for 2 reasons:
#     so that we can use a simple target function, y = x + t
#     most pinns have only one output neuron, so it is a good idea to have one here too.
w3=np.array([[0.5],[-0.4],[0.3]]) #this needs to be a column vector, otherwise numpy freaks out even after transposing
#print("shape of w3: ", np.shape(w3))
b3=np.array([0.1])
y3=np.round(np.matmul(y2_activated,w3)+b3[None, :],2)
print("shape of y3: ", np.shape(y3))
print("output of the neural network: ", y3)
y3_activated=np.round(af.tanh(y3), 2)
print("shape of y3_activated: ", np.shape(y3_activated))
#print("output of the neural network with tanh activation: ", y3_activated)
print("output of the neural network: ", y3_activated)

# now we will create a loss function
# i am arbitrarily picking y = x + t as the target function

y_real=(x+t)[:,None]
print("y_real: ", y_real)
print("shape of y_real: ", np.shape(y_real))
y_pred=y3_activated
print("y_pred: ", y_pred)
print("shape of y_pred: ", np.shape(y_pred)) #the shapes of y_real and y_pred should be the same

# we now import the loss functions from loss_functions.py
from loss_functions import LossFunctions as lf
lf = lf(None, None)
loss=lf.mse(y_real, y_pred)
print("loss: ", loss)