# this is a refactored version of numerical_gradients.py

import numpy as np
from activation_functions import ActivationFunctions as af
from loss_functions import LossFunctions as lf

af = af(None, None, None)
lf = lf(None, None)

# -----------------------------
# inputs
# -----------------------------
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
t = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

# -----------------------------
# network parameters
# -----------------------------
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

# -----------------------------
# forward pass
# -----------------------------
def forward_pass():
    # layer 1
    y1 = w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :]
    y1_activated = af.tanh(y1)

    # layer 2
    y2 = np.matmul(y1_activated, w2.T) + b2[None, :]
    y2_activated = af.tanh(y2)

    # layer 3 (single output neuron)
    y3 = np.matmul(y2_activated, w3) + b3[None, :]
    y3_activated = af.tanh(y3)

    return y3_activated

# -----------------------------
# target and loss
# -----------------------------
y_real = (x + t)[:, None]

def compute_loss():
    y_pred = forward_pass()
    return lf.mse(y_real, y_pred)

# -----------------------------
# numerical gradient (central difference)
# -----------------------------
w_x1_original = w_x1[0]  # we will change this by a small amount and see the change in loss.
eps = 1e-5  # this is the small amount by which we will change the weight.

# loss with +eps
w_x1[0] = w_x1_original + eps
l_plus = compute_loss()
print("loss with w_x1[0] + eps:", l_plus)

# loss with -eps
w_x1[0] = w_x1_original - eps
l_minus = compute_loss()
print("loss with w_x1[0] - eps:", l_minus)

# restore original value
w_x1[0] = w_x1_original

# numerical gradient
numerical_gradient = (l_plus - l_minus) / (2 * eps)
print("numerical gradient:", numerical_gradient)
