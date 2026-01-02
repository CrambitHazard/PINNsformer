# still unclear for implementation. must redo.

import numpy as np

def tanh(y):
    z=(np.exp(-y)-np.exp(y))/(np.exp(y)+np.exp(-y))
    return z

# x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# t = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
# print("shape of x: ", np.shape(x), "shape of t: ", np.shape(t))

# w_x1 = np.array([0.3, -0.2, 0.4, -0.3])
# w_t1 = np.array([0.4,  0.3, -0.2,  0.2])
# b1   = np.array([0.1, -0.1, 0.2, -0.2])
# print("shape of w_x1: ", np.shape(w_x1), "shape of w_t1: ", np.shape(w_t1), "shape of b1: ", np.shape(b1))

# w2 = np.array([
#     [ 0.3, -0.2,  0.4, -0.1],
#     [-0.2,  0.3,  0.1,  0.3],
#     [ 0.1, -0.1,  0.2, -0.2]
# ])
# print("shape of w2: ", np.shape(w2))

# b2 = np.array([0.1, -0.1, 0.05])
# print("shape of b2: ", np.shape(b2))

# w3 = np.array([[0.5],
#                [-0.4],
#                [0.3]])
# print("shape of w3: ", np.shape(w3))

# b3 = np.array([0.1])
# print("shape of b3: ", np.shape(b3))

x=np.array([0.3,0.2,0.3,0.6,0.7])
t=np.array([0.1,0.3,0.4,0.2,0.5]) #(5,)

w_x1=np.array([0.1,0.2,0.3,0.4])
w_t1=np.array([0.1,0.2,0.3,0.4])
b1=np.array([0.1,0.2,0.3,0.4]) #(4,)

w2=np.array([[0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.1,0.4,0.2,0.3]]) #(3,4)
b2=np.array([0.3,0.4,0.1]) #(3,)

w3=np.array([[0.1],[0.05],[0.12]]) #(1,3)
b3=np.array([0.02]) #(1,)


cache={ 'y2_activated': None, 'y3_activated': None }

def forward_pass():
    y1=w_x1[None,:]*x[:,None]+w_t1[None,:]*t[:,None]+b1[None,:]
    y1_activated=tanh(y1)
    print("shape of y1_activated: ", np.shape(y1_activated))

    y2=np.matmul(y1_activated,w2.T)+b2[None,:]
    y2_activated=tanh(y2)
    print("shape of y2_activated: ", np.shape(y2_activated))
    cache['y2_activated'] = y2_activated

    y3=np.matmul(y2_activated,w3)+b3[None,:]
    y3_activated=tanh(y3)
    print("shape of y3_activated: ", np.shape(y3_activated))
    cache['y3_activated'] = y3_activated
    return y3_activated

def backward_pass_last_layer(cache, y3_activated, y_real):
    error=2*(y_real-y3_activated)/np.size(y_real)
    print("shape of error: ", np.shape(error))

    tanh_slope=1-y3_activated**2
    print("shape of tanh_slope: ", np.shape(tanh_slope))

    gradient_y3=error*tanh_slope
    print("shape of gradient_y3: ", np.shape(gradient_y3))

    gradient_w3=np.matmul(cache['y2_activated'].T,gradient_y3)
    print("shape of gradient_w3: ", np.shape(gradient_w3))

    gradient_b3=np.sum(gradient_y3,axis=0)
    print("shape of gradient_b3: ", np.shape(gradient_b3))

    return gradient_w3, gradient_b3

def compute_loss():
    y_pred=forward_pass()
    return np.mean((y_real-y_pred)**2)

y_real=(x+t)[:,None]
print("shape of y_real: ", np.shape(y_real))

max_steps=100
alpha=1e-4
eps=1e-5
loss_history=[]

for i in range(max_steps):
    loss=compute_loss()
    w3_original=w3[0][0]
    w3[0][0]=w3_original+eps
    l_plus=compute_loss()
    w3[0][0]=w3_original-eps
    l_minus=compute_loss()
    w3[0][0]=w3_original
    numerical_gradient=(l_plus-l_minus)/(2*eps)
    w3[0][0]=w3[0][0]-alpha*numerical_gradient
    loss_history.append(loss)

    if i%10==0:
        print("Loss after", i, "steps:", loss)

y3_activated=forward_pass()
gradient_w3, gradient_b3=backward_pass_last_layer(cache, y3_activated, y_real)

print("Final loss:", loss)
print("Numerical gradient:", numerical_gradient)
print("Analytical gradient:", gradient_w3[0][0])