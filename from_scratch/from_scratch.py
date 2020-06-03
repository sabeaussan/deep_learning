import numpy as np
from models import CNN
import torchvision
import torch
import function as F

# Train CNN on MNIST

BATCH_SIZE = 64

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../tmp/test', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE, shuffle=True)



OUTPUT_DIM = 10
EPOCH = 10
loss_memory = []
loss_history = []
lr = 0.3
conv_net = CNN(OUTPUT_DIM,lr)

for e in range(EPOCH):
    correct = 0
    #test(conv_net,e)
    for batch_idx, (data, targets) in enumerate(test_loader):
        print(batch_idx)
        input_data = data.numpy()
        out = conv_net.forward(input_data)
        pred = np.argmax(out,axis = 1)
        labels = np.zeros((len(targets),OUTPUT_DIM))
        for i in range(len(targets)):
            if pred[i] == targets[i] :
                correct += 1
            labels[i][targets[i]] = 1
        loss = F.compute_loss(out,labels)
        loss_memory.append(np.sum(loss,axis = 0))
        conv_net.backward(labels)
        conv_net.sgd()
    loss_history.append(loss_memory)
    av_loss = np.sum(loss_memory)/len(test_loader.dataset)
    loss_memory = []
    print('\n Training Epoch : {}, Accuracy: ({:.0f}%), Averaged loss : {:.4f} '.format(e, correct/len(test_loader.dataset)*100,av_loss))
    