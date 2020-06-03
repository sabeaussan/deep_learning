import numpy as np


def reLu(h):
	h[h<0] = 0
	return h

def relu_derivative(h):
    h[h>0] = 1
    h[h<=0] = 0
    return h

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoid_derivative(sigmoid):
    return sigmoid*(1-sigmoid)


def softmax(y):
	return np.exp(y)/np.sum(np.exp(y))

def softmax_derivative(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)



# On average toute les cost pour un batch
def compute_loss(out,label):
    loss = np.sum(1/2*(out-label)**2, axis = 1)
    return loss

def loss_derivative(output,label):
    return (output-label)

def compute_loss_torch(out,label):
    loss = torch.sum(1/2*(out-label)**2)
    return loss