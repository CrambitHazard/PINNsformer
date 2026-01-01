# numerical gradient is nothing but the change of a variable with respect to the change in another variable,
# or simply, a first derivative.
# in ai ml, the variable whose change is calculated is loss.
# we will use the central difference method to calculate the numerical gradient for now.

# reusing the same network from loss_function_tutorial.py

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

y1=w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :]

y1_activated=af.tanh(y1)

y2=np.matmul(y1_activated, w2.T) + b2[None, :]

y2_activated=af.tanh(y2)

w3=np.array([[0.5],[-0.4],[0.3]])
b3=np.array([0.1])
y3=np.matmul(y2_activated,w3)+b3[None, :]
y3_activated=af.tanh(y3)

y_real=(x+t)[:,None]
y_pred=y3_activated

from loss_functions import LossFunctions as lf
lf = lf(None, None)

w_x1_original=w_x1[0] #we will change this by a small amount and see the change in loss. but we have to do so on the original elemnt, not a copy.
eps=0.00001 #this is the small amount by which we will change the weight. it is generally either 10^-5, or 10^-4
w_x1[0]=w_x1_original+eps
#print("w_x1[0] after adding eps: ", w_x1[0])
#print("w_x1_original: ", w_x1_original)

y1=w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :]
y1_activated=af.tanh(y1)

y2=np.matmul(y1_activated, w2.T) + b2[None, :]

y2_activated=af.tanh(y2)

w3=np.array([[0.5],[-0.4],[0.3]])
b3=np.array([0.1])
y3=np.matmul(y2_activated,w3)+b3[None, :]
y3_activated=af.tanh(y3)

y_real=(x+t)[:,None]
y_pred=y3_activated

l_plus=lf.mse(y_real, y_pred)
print("loss with w_x1[0] + eps: ", l_plus)

w_x1[0]=w_x1_original-eps
#print("w_x1[0] after subtracting eps: ", w_x1[0])
#print("w_x1_original: ", w_x1_original)

y1=w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :]
y1_activated=af.tanh(y1)

y2=np.matmul(y1_activated, w2.T) + b2[None, :]

y2_activated=af.tanh(y2)

w3=np.array([[0.5],[-0.4],[0.3]])
b3=np.array([0.1])
y3=np.matmul(y2_activated,w3)+b3[None, :]
y3_activated=af.tanh(y3)

y_real=(x+t)[:,None]
y_pred=y3_activated

l_minus=lf.mse(y_real, y_pred)
print("loss with w_x1[0] - eps: ", l_minus)

numerical_gradient=(l_plus-l_minus)/(2*eps)
print("numerical gradient: ", numerical_gradient)




