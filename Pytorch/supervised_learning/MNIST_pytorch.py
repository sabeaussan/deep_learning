import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
	def __init__(self,input_size, hidden_size, output_size):
		super(Network, self).__init__()                 
		self.fc1 = nn.Linear(input_size, hidden_size[0])           # FC layer
		self.fc2 = nn.Linear(hidden_size[0], output_size)          # FC layer
		
	def forward(self, x):
		x = F.relu(self.fc1(x))                                      # activation fonction ReLU
		x = self.fc2(x)                                              # FC layer output
		return F.sigmoid(x)


def test(model,epoch):
	correct = 0
	loss_memory = []
	with torch.no_grad():
		for batch_idx, (data, targets) in enumerate(test_loader):
			input_data = data.view(-1,28*28)
			out = net(input_data)
			_, pred = torch.max(out,axis = 1)
			labels = torch.zeros((len(targets),OUTPUT_DIM))
			for i in range(len(targets)):
				if pred[i] == targets[i] :
					correct += 1
				labels[i][targets[i]] = 1
			loss = F.mse_loss(out,labels)
			loss_memory.append(loss)
		total_loss = np.sum(loss_memory)
		print('\n Test Epoch : {},  Accuracy: ({:.0f}%), Total loss : {:.4f} '.format(epoch, correct/len(test_loader.dataset)*100,total_loss))


# Hyper-parameters
INPUT_DIM = 28*28
HIDDEN_DIM = 100
OUTPUT_DIM = 10
lr = 3.0
BATCH_SIZE = 64
EPOCH = 10
loss_memory = []
loss_history = []
net = Network(INPUT_DIM,[HIDDEN_DIM],OUTPUT_DIM)
opti = optim.SGD(net.parameters(), lr=lr)

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

for e in range(EPOCH):
	correct = 0
	test(net,e)
	for batch_idx, (data, targets) in enumerate(train_loader):
		input_data = data.view(-1,28*28)
		out = net(input_data)
		_, pred = torch.max(out,axis = 1)
		labels = torch.zeros((len(targets),OUTPUT_DIM))
		for i in range(len(targets)):
			if pred[i] == targets[i] :
				correct += 1
			labels[i][targets[i]] = 1
		loss = F.mse_loss(out,labels)
		loss_memory.append(loss)
		opti.zero_grad()
		loss.backward()
		opti.step()
	loss_history.append(loss_memory)
	total_loss = np.sum(loss_memory)
	loss_memory = []
	print('\n Training Epoch : {}, Accuracy: ({:.0f}%), Total loss : {:.4f} '.format(e, correct/len(train_loader.dataset)*100,total_loss))
	