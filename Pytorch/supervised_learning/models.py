import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


def create_ResNet(n_blocks,n_channels,n_filters,n_layers=2,down_sample = True):
    # Satck ResidualBlock
    """
        args :
            n_blocks : number of residual block to stack
            n_filters : number of filters for every conv
            n_channels : number of input channels
            n_layers : number of conv for each block
            down_sample : should down_sample data with stride of 2, most of the time
                down_sampling is done at the beginig of every batch
    """
    blocks = []
    for i in range(n_blocks):
        blocks += [ResidualBlock(
            n_layers,
            n_channels if i ==0 else n_filters,
            n_filters,
            down_sample = down_sample if i ==0 else False
        )]
    return nn.Sequential(*blocks)
        

class ResNet34(nn.Module):
    """
        Build a ResNet model with 34 layers
        args :
            n_channels : number of channels input
            input_size : size of the input data
            output_dim : number of class to classify
            
    """
    def __init__(self,n_channels,input_size,output_dim):
        self.n_filters = 64
        super(ResNet34, self).__init__()   
        self.n_channels = n_channels
        self.input_size = input_size
        self.base_conv = nn.Sequential(
            nn.Conv2d(n_channels,self.n_filters,7, stride = 2,padding = 3),
            nn.MaxPool2d(3,2,1)
        )
        self.conv3x3_1 = create_ResNet(3,self.n_filters,self.n_filters,down_sample = False)
        self.conv3x3_2 = create_ResNet(4,self.n_filters,self.n_filters*2)
        self.conv3x3_3 = create_ResNet(6,self.n_filters*2,self.n_filters*4)
        self.conv3x3_4 = create_ResNet(3,self.n_filters*4,self.n_filters*8)
        self.flat = self.compute_out_shape()
        self.avg_pooling = nn.AvgPool2d(self.flat[-1])
        self.fc = nn.Linear(self.flat[1],output_dim)
        
        
    def compute_out_shape(self):
        # Compute the shape of the output tensor of the convNet
        x = torch.FloatTensor(np.ones((1,self.n_channels,self.input_size,self.input_size)))
        x = self.base_conv(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv3x3_3(x)
        x = self.conv3x3_4(x)
        #x = self.avg_pooling(x)
        return x.shape
        
    def forward(self, x):
        x = self.base_conv(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv3x3_3(x)
        x = self.conv3x3_4(x)
        x = self.avg_pooling(x)
        x = x.view(-1,self.flat[1])
        x = self.fc(x)
        return F.softmax(x)
    
    
    
    
    #TODO : def create_ResNet


def create_ResNet(n_blocks,n_channels,n_filters,n_layers=2,down_sample = True):
    blocks = []
    for i in range(n_blocks):
        blocks += [ResidualBlock(
            n_layers,
            n_channels if i ==0 else n_filters,
            n_filters,
            down_sample = down_sample if i ==0 else False
        )]
    return nn.Sequential(*blocks)
        

class ResNet18(nn.Module):
    """
        Build a ResNet model with 34 layers
        agrs :
            n_channels : number of channels input
            input_size : size of the input data
            
    """
    def __init__(self,n_channels,input_size):
        self.n_filters = 64
        super(ResNet18, self).__init__()   
        self.n_channels = n_channels
        self.input_size = input_size
        self.base_conv = nn.Sequential(
            nn.Conv2d(n_channels,self.n_filters,3, stride = 2,padding = 3),
            nn.MaxPool2d(3,2,1)
        )
        self.conv3x3_1 = create_ResNet(2,self.n_filters,self.n_filters,down_sample = False)
        self.conv3x3_2 = create_ResNet(2,self.n_filters,self.n_filters*2)
        self.conv3x3_3 = create_ResNet(2,self.n_filters*2,self.n_filters*4)
        self.conv3x3_4 = create_ResNet(2,self.n_filters*4,self.n_filters*8)
        self.avg_pooling = nn.AvgPool2d(7)
        self.flat = np.prod(self.compute_out_shape())
        self.fc = nn.Linear(self.flat,10)
        
        
    def compute_out_shape(self):
        # Compute the shape of the output tensor of the convNet
        dummy = torch.FloatTensor(np.ones((1,self.n_channels,self.input_size,self.input_size)))
        x = self.base_conv(dummy)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv3x3_3(x)
        x = self.conv3x3_4(x)
        x = self.avg_pooling(x)
        return x.shape
        
    def forward(self, x):
        x = self.base_conv(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv3x3_3(x)
        x = self.conv3x3_4(x)
        x = self.avg_pooling(x)
        x = x.view(-1,self.flat)
        x = self.fc(x)
        return F.softmax(x)

# TODO : 
# try callbacks for early stopping
# add lr schedule

def create_conv_net( activation,n_input,output_activation,conv_type,num_filters,kernel_n,stride,padding,normalize=False):
    """Create a convnet:
        args :
            activation,padding,normalize,stride,kernel_n,num_filters : same as ResidualBlock
            output_activation : activation of last layer
            conv_type : convolution type 
    """
    layers = []
    size = len(num_filters)
    for j in range(size):
        act = activation if j < size-1 else output_activation
        if j==0 :
            layers += [
                conv_type(n_input,num_filters[j],kernel_n[j],stride[j],padding[j])
            ]
            if normalize :
                layers+= [nn.BatchNorm2d(num_filters[j])]
            layers+= [act()]
        else : 
            layers += [
                conv_type(num_filters[j-1],num_filters[j],kernel_n[j],stride[j],padding[j])
            ]
            if normalize :
                layers+= [nn.BatchNorm2d(num_filters[j])]
            #if j < size - 1 :
            layers+= [act()]
        
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    # TODO : check for dimensionnality and add padding to input
    
    """
        Build a residual block for ResNet
            args : 
                n_layer : number of layers
                n_channels : number of inputs channels
                stride, padding, n_filters, kernel_size : same as conv2d
                act : activation fonction
                batch_norm (bool) : should add batch_norm
                same : use same padding
    """
    def __init__(self,n_layer, n_channels, n_filters,padding = 0, kernel_size=3,stride=1,act=nn.ReLU,down_sample = True,batch_norm= True):
        super(ResidualBlock, self).__init__()   
        self.padding = [(kernel_size-1)//2] * n_layer
        self.stride = [2] + [1] * (n_layer-1) if down_sample else [1] * n_layer
        self.n_filters = [n_filters] * n_layer
        self.kernel_size = [kernel_size] * kernel_size
        self.res_block = create_conv_net(
            act,
            n_channels,
            act,
            nn.Conv2d,
            self.n_filters,
            self.kernel_size,
            self.stride,
            self.padding,
            batch_norm
        )
        
        
    def forward(self,x):
        res = x#.copy()
        out = self.res_block(x)
        # if dimensions does not match, we perform 1x1 conv
        if x.shape[1] != out.shape[1] :
            x = nn.Conv2d(x.shape[1],out.shape[1],1,2,0)(x)
        return F.relu(out+x)
        #return F.relu(out)
