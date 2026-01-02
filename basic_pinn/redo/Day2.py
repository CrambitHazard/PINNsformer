import numpy as np

def tanh(y):
    z=(np.exp(-y)-np.exp(y))/(np.exp(-y)+np.exp(y))
    return z

cache={'a2':None,'a3':None}
def forward_pass():
    y1=w_x1[None,:]*x[:,None]+w_t1[None,:]*t[:,None]+b1[None,:]
    a1=tanh(y1)

    y2=np.matmul(a1,w2.T)+b2[None,:]
    a2=tanh(y2)
    cache['a2']=a2

    y3=np.matmul(a2,w3.T)+b3[None,:]
    a3=tanh(y3)
    cache['a3']=a3

    return a3

def backward_pass(cache,y_real,y_pred):
    tanh_slope=1-cache['a3']**2
    error=2*(y_real-y_pred)/np.size(y_real)

    y3_gradient=error*tanh_slope
    w3_gradient=np.matmul(cache['a2'].T,y3_gradient)
    b3_gradient=np.sum(y3_gradient,axis=0)

    return w3_gradient,b3_gradient

def compute_loss(y_real):
    y_pred=forward_pass()
    loss=np.mean((y_real-y_pred)**2)
    return loss

x=np.array([0.3,0.2,0.3,0.6,0.7])
t=np.array([0.1,0.3,0.4,0.2,0.5]) #(5,)

w_x1=np.array([0.1,0.2,0.3,0.4])
w_t1=np.array([0.1,0.2,0.3,0.4])
b1=np.array([0.1,0.2,0.3,0.4]) #(4,)

w2=np.array([[0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.1,0.4,0.2,0.3]]) #(3,4)
b2=np.array([0.3,0.4,0.1]) #(3,)

w3=np.array([[0.1,0.05,0.12]]) #(1,3)
b3=np.array([0.02]) #(1,)

y_real=(x+t)[:,None]
mse_loss=compute_loss(y_real)
#print(mse_loss)

eps=1e-4
max_steps=100
alpha=1e-4
loss_history=[]

for i in range(max_steps):
    w3_original=w3[0][0]
    loss=compute_loss(y_real)

    w3[0][0]=w3_original+eps
    l_plus=compute_loss(y_real)

    w3[0][0]=w3_original-eps
    l_minus=compute_loss(y_real)

    w3[0][0]=w3_original

    numerical_gradient=(l_plus-l_minus)/(2*eps)
    loss_history.append(loss)

    w3[0][0]=w3_original-alpha*numerical_gradient

    if i%10==0:
        print(loss)

a3=forward_pass()
w3_gradient,b3_gradient=backward_pass(cache,y_real,a3)

print('Numerical Gradient ', numerical_gradient)
print('Analytical Gradient ', w3_gradient[0][0])