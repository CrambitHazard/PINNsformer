# backprop is just a way to calculate the gradient of the loss with respect to the weights.
# here, we will only apply it to the last layer.

#reusing the same network from training_loop.py

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

cache={ 'y2_activated': None, 'y3_activated': None }
def forward_pass():
    y1 = w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :]
    y1_activated = af.tanh(y1)

    y2 = np.matmul(y1_activated, w2.T) + b2[None, :]
    y2_activated = af.tanh(y2)
    cache['y2_activated'] = y2_activated

    y3 = np.matmul(y2_activated, w3) + b3[None, :]
    y3_activated = af.tanh(y3)
    cache['y3_activated'] = y3_activated
    return y3_activated

y_real = (x + t)[:, None]

def backward_pass_last_layer(cache, y3_activated, y_real):
    error=2*(y_real-y3_activated)/np.size(y_real)
    tanh_slope=1-y3_activated**2
    gradient_y3=error*tanh_slope
    gradient_w3=np.matmul(cache['y2_activated'].T,gradient_y3)
    gradient_b3=np.sum(gradient_y3, axis=0)
    return gradient_w3, gradient_b3


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
    w3_original = w3[0][0]
    w3[0][0] = w3_original + eps
    l_plus = compute_loss()
    w3[0][0] = w3_original - eps
    l_minus = compute_loss()
    w3[0][0] = w3_original
    numerical_gradient = (l_plus - l_minus) / (2 * eps)
    w3[0][0] = w3[0][0] - alpha * numerical_gradient
    loss_history.append(loss)

    if i % 10 == 0:
        print("Loss after", i, "steps:", loss)

y3_activated = forward_pass()
gradient_w3, gradient_b3 = backward_pass_last_layer(cache, y3_activated, y_real)

print("Final loss:", loss)
print("Numerical gradient:", numerical_gradient)
print("Analytical gradient:", gradient_w3[0][0])