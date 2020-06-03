import numpy as np
import torchvision
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import torch
import torch.autograd as auto
import torch.nn as nn
from torch.autograd.variable import Variable
from models import Generator,Critic
from torch.optim import Adam
from utils import plot_image,rescale_image

LAMBDA = 10
NC_CRITIC = 5
LATENT_SPACE_DIM = 100
INPUT_DIM = (1,28,28)
EPOCH = 70
BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../tmp/train', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5,), (0.5,))
                             ])),
  batch_size=BATCH_SIZE, shuffle=True)




def compute_grad_penalty(generated,real,critic):
    # Sample interpolate coefficient
    t = torch.tensor(np.random.random((real.shape[0], real.shape[1],1,1)),dtype=torch.float32)
    interpolation = Variable(t*real + (1-t)*generated, requires_grad=True)
    # if using critic is using conv_net we don't need to reshape
    if critic.__class__.__name__ == "ConvCritic" :
      pred = critic(interpolation)
    else :
      pred = critic(interpolation.view(-1,np.prod(generated.shape[1:])))
    gradients = auto.grad(
        outputs=pred,
        inputs=interpolation,
        grad_outputs=torch.ones(pred.size()),
        create_graph=True,
        retain_graph=True,
    )[0]  #Only one input for the derivative
    gradients = gradients.view(gradients.size(0), -1)
    norm = gradients.norm(2, dim=1)
    gradient_penalty = ((norm - 1) ** 2).mean()
    return gradient_penalty



gen = Generator(LATENT_SPACE_DIM,[256,512,1024],np.prod(INPUT_DIM))
critic = Critic(np.prod(INPUT_DIM),[1024,512,256])

loss_gene_memory = []
loss_critic_memory = []

optimizer_d = Adam(critic.parameters(), lr=0.0001,betas=(0, 0.9))
optimizer_g = Adam(gen.parameters(), lr=0.0001,betas=(0, 0.9))

test_input_z = Variable(torch.Tensor(np.random.normal(0,1,(1,LATENT_SPACE_DIM))))

for e in range(EPOCH) :
    loss_gene_memory = []
    loss_critic_memory = []
    if e %3 == 0 :
        with torch.no_grad():
            fake_image_test = gen(test_input_z)
            fake_image_test = fake_image_test.view(-1,INPUT_DIM[0],INPUT_DIM[1],INPUT_DIM[2])
            plot_image(rescale_image(fake_image_test.data[0][0],0.5))
    for idx, (images,_) in enumerate(train_loader):
        
        # Configure input
        real_imgs = Variable(images.type(torch.Tensor)).view(-1,np.prod(INPUT_DIM))
        size_batch = images.shape[0]
        
        # Training Discriminator
        optimizer_d.zero_grad()
        
        #Sample a latent vector
        noise_vector = Variable(torch.Tensor(np.random.normal(0,1,(size_batch,LATENT_SPACE_DIM))))
        
        # Generate batch of fake images
        fake_image = gen(noise_vector).detach()
        
        # check critic prediction
        pred_fake = critic(fake_image)
        pred_real = critic(real_imgs)

        # resize to match image shape
        fake_image = fake_image.view(size_batch,INPUT_DIM[0],INPUT_DIM[1],INPUT_DIM[2])
        real_imgs = real_imgs.view(size_batch,INPUT_DIM[0],INPUT_DIM[1],INPUT_DIM[2])
        
        # Flatten image and forward 
        #real_images = images.view(size_batch,28*28)

        
                
        # Compute gradient penalty 
        gradient_p = compute_grad_penalty(fake_image,real_imgs,critic)
        
        # Compute loss
        loss_critic = torch.mean(pred_fake) - torch.mean(pred_real) + LAMBDA*gradient_p
        loss_critic.backward()
        optimizer_d.step()
        loss_critic_memory.append(loss_critic.item())
        
        if idx%NC_CRITIC == 0 :
            # Training Generator
            optimizer_g.zero_grad()
            noise_vector = Variable(torch.Tensor(np.random.normal(0,1,(size_batch,LATENT_SPACE_DIM))))
            fake_image = gen(noise_vector)
            pred_fake = critic(fake_image)
            loss_gen = -torch.mean(pred_fake)
            loss_gen.backward()
            optimizer_g.step()
            loss_gene_memory.append(loss_gen.item())
            print(
                "Epoch : %d/%d , Batch : %d/%d, C loss: %f, G loss: %f"
                % (e, EPOCH, idx, len(train_loader), loss_critic.item(), loss_gen.item())
            )
    print('\n Training Epoch : {},\n loss discriminator : {:.4f}, loss generator : {:.4f} '.format(e,np.mean(loss_critic_memory),np.mean(loss_gene_memory)))

#PATH = "saved_models/model_WGAN-GP_MNIST.pt"
#torch.save(gen.state_dict(), PATH)