# here we will implement full backprop through all layers and variables
# it is same as backprop basically, just we have to do it more times

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

cache={'a1':None,'a2':None,'a3':None}
def forward_pass():
    y1 = w_x1[None, :] * x[:, None] + w_t1[None, :] * t[:, None] + b1[None, :]
    a1 = af.tanh(y1)
    cache['a1']=a1

    y2 = np.matmul(a1, w2.T) + b2[None, :]
    a2 = af.tanh(y2)
    cache['a2']=a2

    y3 = np.matmul(a2, w3) + b3[None, :]
    a3 = af.tanh(y3)
    cache['a3']=a3

    return a3

def backward_pass(cache,y_real):
    N = np.size(y_real)

    a3 = cache['a3']
    a2 = cache['a2']
    a1 = cache['a1']

    dl_da3 = 2 * (a3 - y_real) / np.size(y_real)
    dl_dy3 = dl_da3 * (1 - a3**2)

    dl_dw3 = np.matmul(a2.T, dl_dy3)
    dl_db3 = np.sum(dl_dy3, axis=0)
    dl_da2 = np.matmul(dl_dy3, w3.T)

    dl_dy2 = dl_da2 * (1 - a2**2)

    dl_dw2 = np.matmul(dl_dy2.T, a1)
    dl_db2 = np.sum(dl_dy2, axis=0)
    dl_da1 = np.matmul(dl_dy2, w2)

    dl_dy1 = dl_da1 * (1 - a1**2)

    dl_dwx1 = np.sum(dl_dy1 * x[:, None], axis=0)
    dl_dwt1 = np.sum(dl_dy1 * t[:, None], axis=0)
    dl_db1  = np.sum(dl_dy1, axis=0)

    grads = {'dw3': dl_dw3,'db3': dl_db3,'dw2': dl_dw2,'db2': dl_db2,'dw_x1': dl_dwx1,'dw_t1': dl_dwt1,'db1': dl_db1}
    return grads

def compute_loss():
    y_pred = forward_pass()
    return lf.mse(y_real, y_pred)

max_steps=100
eps=1e-4
alpha=1e-4
loss_history=[]

y_real=(x+t)[:,None]

for i in range(max_steps):
    loss=compute_loss()
    grads=backward_pass(cache,y_real)
    w3-=alpha*grads['dw3']
    w2-=alpha*grads['dw2']
    w_x1-=alpha*grads['dw_x1']
    w_t1-=alpha*grads['dw_t1']
    b3-=alpha*grads['db3']
    b2-=alpha*grads['db2']
    b1-=alpha*grads['db1']

    if i%10==0:
        print(loss)

