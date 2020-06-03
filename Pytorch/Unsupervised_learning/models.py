import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F



def create_body( sizes, activation=nn.LeakyReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Generator(nn.Module):
    # Build the generator with fully connected layer
    # Generate fake image in order to fool the discriminator
    def __init__(self,input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        # using tanh as output activation because image are scaled between -1 and 1
        self.model = create_body([input_size]+hidden_size+[output_size],output_activation = nn.Tanh)
        
    def forward(self, z):
        fake_img = self.model(z)
        return fake_img


class Discriminator(nn.Module):
    # Build the generator with fully connected layer
    # Try to discriminate between fake image and real image from the dataset
    def __init__(self,input_size, hidden_size):
        super(Discriminator, self).__init__()                 
        self.model = create_body([input_size]+hidden_size+[1],output_activation = nn.Sigmoid)
        
    def forward(self, data):
        valid = self.model(data)
        return valid 

class Critic(nn.Module):
	# Critic model for WGAN
    def __init__(self,input_size, hidden_size):
        super(Critic, self).__init__()                 
        self.model = create_body([input_size]+hidden_size+[1])
        
    def forward(self, data):
        valid = self.model(data)
        return valid 

def create_conv_net( activation,n_input,output_activation,conv_type,num_filters,kernel_n,stride,padding,normalize=False,dropout=False):
    layers = []
    size = len(num_filters)
    for j in range(size):
        act = activation if j < size-1 else output_activation
        if j==0 :
            layers += [conv_type(n_input,num_filters[j],kernel_n[j],stride[j],padding[j]),act()]
        else : 
            layers += [conv_type(num_filters[j-1],num_filters[j],kernel_n[j],stride[j],padding[j]),act()]
        if j < size - 1 :
            if normalize :
                layers.append(nn.BatchNorm2d(num_filters[j]))
            if dropout:
                layers.append(nn.Dropout(0.25))
    return nn.Sequential(*layers)

class ConvGenerator(nn.Module):
	# Build generator for DCGAN
    def __init__(self,input_size, num_filters,kernel_n,stride,nb_channels):
        super(ConvGenerator, self).__init__() 
        self.num_filters = num_filters
        # Project onto higher dimension
        self.fc = nn.Linear(input_size,4*4*num_filters)
        self.model = create_conv_net(
            nn.ReLU,
            num_filters,
            nn.Tanh,
            nn.ConvTranspose2d,
            [num_filters,num_filters//2,nb_channels],
            [kernel_n] *3,
            [stride] *3,
            [1] *3,
            True,
        )
        
        
    def forward(self, z):
        fake_img = self.fc(z)
        # Reshape for convnet
        fake_img = fake_img.view(-1,self.num_filters,4,4)
        fake_img = self.model(fake_img)
        return fake_img


class ConvDiscriminator(nn.Module):
	# Discriminator for DCGAN
    def __init__(self,nb_channels,num_filters,kernel_n,stride):
        super(ConvDiscriminator, self).__init__()
        self.model = create_conv_net(
            nn.LeakyReLU,
            nb_channels,
            nn.Sigmoid,
            nn.Conv2d,
            [num_filters,num_filters*2,num_filters*4,1],
            [kernel_n] *4,
            [stride] *3 + [1],
            [1] *3 + [0],
            True,
        )
        
        
    def forward(self, data):
        valid = self.model(data)
        return valid 





