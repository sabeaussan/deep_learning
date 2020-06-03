import numpy as np
import torchvision
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from models import ConvGenerator,ConvDiscriminator
from torch.optim import Adam
from utils import plot_image,rescale_image



train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../tmp/train', train=True, download=True,
							 transform=torchvision.transforms.Compose([
							   torchvision.transforms.Resize(32),
							   torchvision.transforms.ToTensor(),
							   torchvision.transforms.Normalize(
								 (0.5,), (0.5,))
							 ])),
  batch_size=128, shuffle=True)




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


NUM_FILTERS = 32
KERNEL_SIZE = 4
STRIDE = 2
N_CHANNEL = 1
BATCH_SIZE = 128
LATENT_SPACE_DIM = 100
UPDATE_FREQ = 1
EPOCH = 30
gen = ConvGenerator(LATENT_SPACE_DIM,NUM_FILTERS,KERNEL_SIZE,STRIDE,N_CHANNEL)
disc = ConvDiscriminator(N_CHANNEL,NUM_FILTERS//8,KERNEL_SIZE,STRIDE)

gen.model.apply(weights_init)
disc.model.apply(weights_init)

loss_function = nn.BCELoss()

loss_gene_memory = []
loss_disc_memory = []

optimizer_d = Adam(disc.parameters(), lr=0.0002)
optimizer_g = Adam(gen.parameters(), lr=0.0002)

test_input_z = noise = torch.randn(1,LATENT_SPACE_DIM)

for e in range(EPOCH) :
    loss_gene_memory = []
    loss_disc_memory = []
    with torch.no_grad():
        fake_image_test = gen(test_input_z)
        plot_image(rescale_image(fake_image_test[0][0],0.5))
    for idx, (images,_) in enumerate(train_loader):
        size_batch = images.shape[0]
        if idx%100 == 0 :
            print(idx)
            
        # Training Discriminator
        optimizer_d.zero_grad()
        noise_vector = torch.randn(size_batch, LATENT_SPACE_DIM)
        pred_real = disc(images).view(size_batch,-1)

        fake_image = gen(noise_vector).detach()

        pred_fake = disc(fake_image).view(size_batch,-1)
        loss_real = loss_function(pred_real,Variable(torch.ones((size_batch,1))))
        loss_real.backward()
        loss_fake = loss_function(pred_fake,Variable(torch.zeros((size_batch,1))))
        loss_fake.backward()
        optimizer_d.step()
        loss_disc_memory.append(loss_fake.item()+loss_real.item())
        
        if idx%UPDATE_FREQ == 0 :
            # Training Generator
            optimizer_g.zero_grad()
            noise_vector = torch.randn(size_batch, LATENT_SPACE_DIM)
            fake_image = gen(noise_vector)
            pred_fake = disc(fake_image).view(size_batch,-1)
            loss_gen = loss_function(pred_fake,Variable(torch.ones((size_batch,1))))
            loss_gen.backward()
            optimizer_g.step()
            loss_gene_memory.append(loss_gen.item())
    print('\n Training Epoch : {},\n loss discriminator : {:.4f}, loss generator : {:.4f} '.format(e,np.mean(loss_disc_memory),np.mean(loss_gene_memory)))

#PATH = "saved_models/model_DCGAN_MNIST.pt"
#torch.save(gen.state_dict(), PATH)