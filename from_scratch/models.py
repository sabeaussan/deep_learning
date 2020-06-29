import numpy as np
from Layers import Fully_Connected,ConvLayer,MaxPooling
import function as F

"""
    Build a CNN with sigmoid output activation (instead of softmax )
     and RMSE as cost function for simplicity
"""


class CNN():
    def __init__(self, output_size,lr):
        self.conv1 = ConvLayer(1,10,5)
        self.max_pool1 = MaxPooling(2)
        self.conv2 = ConvLayer(10,20,5)
        self.max_pool2 = MaxPooling(2)
        self.fc1 = self.fc1 = Fully_Connected(20*4*4,output_size)
        self.cache = 0
        self.lr = lr
    
    def zero_grad():
        # Empty buffer
        self.cache = 0
        self.conv1.empty()
        self.conv2.empty()
        self.fc1.empty()
    
    def forward(self,inputs):
        y = self.conv1.forward(inputs)
        y = self.max_pool1.forward(y)
        y = self.conv2.forward(y)
        y = self.max_pool2.forward(y)
        y = y.reshape(y.shape[0],-1)
        y = self.fc1.forward(y)
        y = F.sigmoid(y)
        self.cache = y
        return y
    
    def backward(self,labels):
        delta = F.loss_derivative(self.cache,labels) * F.sigmoid_derivative(self.cache)
        delta = self.fc1.backprop(delta)
        delta = delta.reshape(self.cache.shape[0],20,4,4)
        delta = self.max_pool2.backprop(delta)
        delta = self.conv2.backprop(delta)
        delta = self.max_pool1.backprop(delta)
        delta = self.conv1.backprop(delta)
    
    def sgd(self):
        # Stochastic Gradient Descent
        self.fc1.weights -= self.lr * self.fc1.dW
        self.fc1.biases -= self.lr * self.fc1.dB
        self.conv2.kernels -= self.lr * self.conv2.dW
        self.conv1.kernels -= self.lr * self.conv1.dW

