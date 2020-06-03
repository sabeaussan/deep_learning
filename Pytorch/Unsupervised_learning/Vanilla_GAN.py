import numpy as np
import torchvision
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.variable import Variable
from utils import plot_image,rescale_image
from models import Generator,Discriminator

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../tmp/train', train=True, download=True,
							 transform=torchvision.transforms.Compose([
							   torchvision.transforms.ToTensor(),
							   torchvision.transforms.Normalize(
								 (0.5,), (0.5,))
							 ])),
  batch_size=BATCH_SIZE, shuffle=True)



# Convertit un vecteur en image
# View se charge de detach() ?
def vectors_to_images(vectors,n_channels,image_size):
	return vectors.view(vectors.size(0), n_channels, image_size, image_size)


# Utiiser BCE pour l'entrainement
LATENT_SPACE_DIM = 100
IMAGE_SIZE = 28
INPUT_DIM = IMAGE_SIZE*IMAGE_SIZE

EPOCH = 70
UPDATE_FREQ = 1

# k = 1, on prend le max de log(D(G(z))) au lieu de min log(1 - D(G(z)))
gen = Generator(LATENT_SPACE_DIM,[256,512,1024],INPUT_DIM)
disc = Discriminator(INPUT_DIM,[1024,512,256])
loss_fucntion = nn.BCELoss()

loss_gene_memory = []
loss_disc_memory = []

optimizer_d = optim.Adam(disc.parameters(), lr=0.0002)
optimizer_g = optim.Adam(gen.parameters(), lr=0.0002)

test_input_z = Variable(torch.Tensor(np.random.normal(0,1,(1,LATENT_SPACE_DIM))))

for e in range(EPOCH) :
	loss_gene_memory = []
	loss_disc_memory = []
	if e%3 == 0:
		with torch.no_grad():
			fake_image_test = gen(test_input_z)
			fake_image_test = vectors_to_images(fake_image_test,1,28)
			plot_image(rescale_image(fake_image_test.data[0][0],0.5))
	for idx, (images,_) in enumerate(train_loader):
		images = torch.Tensor(images)
		
		size_batch = images.shape[0]
	  

		# Training Discriminator
		optimizer_d.zero_grad()
		noise_vector = Variable(torch.Tensor(np.random.normal(0,1,(size_batch,LATENT_SPACE_DIM))))
		real_images = images.view(-1,IMAGE_SIZE**2)
		pred_real = disc(real_images)
		fake_image = gen(noise_vector).detach()
		pred_fake = disc(fake_image)
		loss_real = loss_fucntion(pred_real,Variable(torch.ones((size_batch,1))))
		loss_real.backward()
		loss_fake = loss_fucntion(pred_fake,Variable(torch.zeros((size_batch,1))))
		loss_fake.backward()
		optimizer_d.step()
		loss_disc_memory.append(loss_fake.item()+loss_real.item())
		
		if idx%UPDATE_FREQ == 0 :
			# Training Generator
			optimizer_g.zero_grad()
			noise_vector = Variable(torch.Tensor(np.random.normal(0,1,(size_batch,LATENT_SPACE_DIM))))
			fake_image = gen(noise_vector)
			pred_fake = disc(fake_image)
			loss_gen = loss_fucntion(pred_fake,Variable(torch.ones((size_batch,1))))
			loss_gen.backward()
			optimizer_g.step()
			loss_gene_memory.append(loss_gen.item())
			if idx%5 ==0 :
				print(
					"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
					% (e,EPOCH, idx, len(train_loader), loss_fake.item()+loss_real.item(), loss_gen.item())
				)
	print('\n Training Epoch : {},\n loss discriminator : {:.4f}, loss generator : {:.4f} '.format(e,np.mean(loss_disc_memory),np.mean(loss_gene_memory)))
	

