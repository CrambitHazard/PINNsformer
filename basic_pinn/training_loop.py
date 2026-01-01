# we will implement how neural networks train themselves, but with a few constraints.
# we wont implement backprop, and we will only calculate gradient for one param.
# training is simply nudging the param we are calculating loss with respect,
# based on the gradient and learning rate.

# we reuse the same network from refactor1.py

import numpy as np
from activation_functions import ActivationFunctions as af
from loss_functions import LossFunctions as lf

af = af(None, None, None)
lf = lf(None, None)

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
t = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

w_x1 = np.array([0.3, -0.2, 0.4, -0.3])
w_t1 = np.array([0.4,  0.3, -0.2,  0.2])
b1   = np.array([0.1, -0.1, 0.2, -0.2])

w2 = np.array([
    [ 0.3, -0.2,  0.4, -0.1],
    [-0.2,  0.3,  0.1,  0.3],
    [ 0.1, -0.1,  0.2, -0.2]
])

b2 = np.array([0.1, -0.1, 0.05])

w3 = np.array([[0.5],
               [-0.4],
               [0.3]])

b3 = np.array([0.1])

def forward_pass():
    y1 = w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :]
    y1_activated = af.tanh(y1)

    y2 = np.matmul(y1_activated, w2.T) + b2[None, :]
    y2_activated = af.tanh(y2)

    y3 = np.matmul(y2_activated, w3) + b3[None, :]
    y3_activated = af.tanh(y3)

    return y3_activated

y_real = (x + t)[:, None]

def compute_loss():
    y_pred = forward_pass()
    return lf.mse(y_real, y_pred)

# training loop
max_steps=100
alpha=1e-3
eps = 1e-5
loss_history=[]

for i in range(max_steps):
    loss = compute_loss()
    w_x1_original = w_x1[0]
    w_x1[0] = w_x1_original + eps
    l_plus = compute_loss()
    w_x1[0] = w_x1_original - eps
    l_minus = compute_loss()
    w_x1[0] = w_x1_original
    numerical_gradient = (l_plus - l_minus) / (2 * eps)
    w_x1[0] = w_x1[0] - alpha * numerical_gradient
    loss_history.append(loss)

    if i % 10 == 0:
        print("Loss after", i, "steps:", loss)

print("Final loss:", loss)








