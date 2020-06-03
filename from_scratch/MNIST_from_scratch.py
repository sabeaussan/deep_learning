import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch
import function as F
from Layers import Fully_Connected


# Old implementation of FullyConnected Layer. Train and Test on MNIST

class Network:
	
	def __init__(self, input_size, hidden_size, output_size):
		self.num_layer = len(hidden_size)+1
		self.fc1 = Fully_Connected(input_size,hidden_size[0])
		self.fc2 = Fully_Connected(hidden_size[0],output_size)
		self.layers = []
		self.layers.append(self.fc2)
		self.layers.append(self.fc1)
		self.activations = []
		self.hidden_state = []
		
	def empty_buffer(self):
		self.activations = []
		self.hidden_state = []        
	
	def forward(self,inputs, train = True):
		if(train) : self.activations.append(inputs)
		x = self.fc1.forward(inputs)
		if(train) : self.hidden_state.append(x)
		x = F.reLu(x)
		if(train) : self.activations.append(x)
		x = self.fc2.forward(x)
		if(train) : self.hidden_state.append(x)
		x = F.sigmoid(x)
		return x

class Optimizer:
	def __init__(self, model):
		self.optimized_model = model
		self.grad_buffer_W = []
		self.grad_buffer_B = []
	
	def empty_buffer(self):
		self.grad_buffer_W = []
		self.grad_buffer_B = []
		
	def compute_grad(self,outputs,labels):
		model = self.optimized_model
		dL = F.loss_derivative(outputs,labels)
		dy = F.sigmoid_derivative(outputs)
		delta = dL * dy
		for layer in reversed(range(0,model.num_layer)):
			self.grad_buffer_W.append(np.dot(model.activations[layer].T,delta)/BATCH_SIZE)
			self.grad_buffer_B.append(np.sum(delta,axis = 0)/BATCH_SIZE)
			if(layer > 0):
				grad_relu = F.relu_derivative(model.hidden_state[layer-1]) 
				delta =  np.dot(delta,model.fc2.weights.T) * grad_relu
	
	def backprop(self):
		for layer in range(0,self.optimized_model.num_layer):
			self.optimized_model.layers[layer].weights -= lr/BATCH_SIZE * self.grad_buffer_W[layer]
			self.optimized_model.layers[layer].biases -= lr/BATCH_SIZE * self.grad_buffer_B[layer]
		return self.optimized_model



def test(model,epoch):
	correct = 0
	loss_memory = []
	for batch_idx, (data, targets) in enumerate(test_loader):
		input_data = data.view(-1,28*28).numpy()
		out = model.forward(input_data,train = False)
		pred = np.argmax(out,axis = 1)
		labels = np.zeros((len(targets),OUTPUT_DIM))
		for i in range(len(targets)):
			if pred[i] == targets[i] :
				correct += 1
			labels[i][targets[i]] = 1
		loss = F.compute_loss(out,labels)
		loss_memory.append(np.sum(loss,axis = 0))
	av_loss = np.sum(loss_memory)/len(test_loader.dataset)
	print('\n Test Epoch : {},  Accuracy: ({:.0f}%), Averaged loss : {:.4f} '.format(epoch, correct/len(test_loader.dataset)*100,av_loss))

# Hyper-parameter
lr = 3.0
BATCH_SIZE = 64
INPUT_DIM = 28*28
HIDDEN_DIM = 100
OUTPUT_DIM = 10
EPOCH = 10
loss_memory = []
loss_history = []

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../tmp/train', train=True, download=True,
							 transform=torchvision.transforms.Compose([
							   torchvision.transforms.ToTensor(),
							   torchvision.transforms.Normalize(
								 (0.1307,), (0.3081,))
							 ])),
  batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../tmp/test', train=False, download=True,
							 transform=torchvision.transforms.Compose([
							   torchvision.transforms.ToTensor(),
							   torchvision.transforms.Normalize(
								 (0.1307,), (0.3081,))
							 ])),
  batch_size=1000, shuffle=True)

net = Network(INPUT_DIM,[HIDDEN_DIM],OUTPUT_DIM)
opti = Optimizer(net)
for e in range(EPOCH):
	correct = 0
	test(net,e)
	for batch_idx, (data, targets) in enumerate(train_loader):
		input_data = data.view(-1,28*28).numpy()
		out = net.forward(input_data)
		pred = np.argmax(out,axis = 1)
		labels = np.zeros((len(targets),OUTPUT_DIM))
		for i in range(len(targets)):
			if pred[i] == targets[i] :
				correct += 1
			labels[i][targets[i]] = 1
		loss = F.compute_loss(out,labels)
		loss_memory.append(np.sum(loss,axis = 0))
		opti.empty_buffer()
		opti.compute_grad(out,labels)
		net = opti.backprop()
		net.empty_buffer()
	loss_history.append(loss_memory)
	av_loss = np.sum(loss_memory)/len(train_loader.dataset)
	loss_memory = []
	print('\n Training Epoch : {}, Accuracy: ({:.0f}%), Averaged loss : {:.4f} '.format(e, correct/len(train_loader.dataset)*100,av_loss))
	


