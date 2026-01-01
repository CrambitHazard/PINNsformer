import numpy as np

def tanh(y):
    z=(np.exp(y)-np.exp(-y))/(np.exp(y)+np.exp(-y))
    return z

x=np.array([0.1, 0.2, 0.3, 0.22, 0.13])
t=np.array([0.2, 0.1, 0.04, 0.02, 0.14]) #(5,)
#print("shape of x: ", np.shape(x), "shape of t: ", np.shape(t))

w_x1=np.array([0.3,0.2,-0.2,-0.3])
w_t1=np.array([0.4,0.3,-0.2,0.2])
b1=np.array([0.1,-0.1,0.2,-0.2]) #(4,)
#print("shape of w_x1: ", np.shape(w_x1), "shape of w_t1: ", np.shape(w_t1), "shape of b1: ", np.shape(b1))

y1=np.round(w_x1[None,:]*x[:,None]+w_t1[None,:]*t[:,None]+b1[None,:],2) #(5,4)
#print("shape of y1: ", np.shape(y1))

y1_activated=np.round(tanh(y1), 2)
#print("shape of y1_activated: ", np.shape(y1_activated))

w2=np.array([[0.3,-0.2,0.4,-0.1],[-0.2,0.3,0.1,0.3],[0.1,-0.1,0.2,-0.2]]) #(3,4)
b2=np.array([0.1,-0.1,0.05]) #(3,)
#print("shape of w2: ", np.shape(w2), "shape of b2: ", np.shape(b2))

y2=np.round(np.matmul(y1_activated,w2.T)+b2[None,:],2) #(5,3)
#print("shape of y2: ", np.shape(y2))

y2_activated=np.round(tanh(y2), 2)
#print("shape of y2_activated: ", np.shape(y2_activated))

w3=np.array([[0.5,-0.4,0.3]]) #(1,3)
b3=np.array([0.1]) #(1,)
#print("shape of w3: ", np.shape(w3) , "shape of b3: ", np.shape(b3))

y3=np.round(np.matmul(y2_activated,w3.T)+b3[None,:],2) #(5,1)
#print("shape of y3: ", np.shape(y3))

y3_activated=np.round(tanh(y3), 2)
#print("shape of y3_activated: ", np.shape(y3_activated))

print("output of the neural network: ", y3_activated)

y_real=(x+t)[:,None] #(5,1)
#print("shape of y_real: ", np.shape(y_real))
print("y_real: ", y_real)

y_pred=y3_activated
print("y_pred: ", y_pred)

loss=np.mean((y_real-y_pred)**2)
print("loss: ", loss)