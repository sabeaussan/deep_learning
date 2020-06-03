import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import cifar10,mnist
from keras.models import Sequential,Model
from keras.layers import Input,Flatten,Dense,Conv2D,BatchNormalization,LeakyReLU,Reshape,Conv2DTranspose,Activation,Dropout,Lambda
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import os
import cv2
import random

class Encoder():
    # Create an encoder for AutoEncoder or VariationnalAutoEncoder
    # Using Convolutionnal Layer
    # BatchNormalization and dropout is optionnal
    # for_VAE arg is used to determine if the encoder is used for an autoencoder or variationnal-autoencoder
    # alpha and beta are used to balance the reconstruction_loss and kl_loss
    def __init__(self,input_shape,num_layers,num_filters,kernel_size,strides,latent_space_dim,for_VAE=True,normalize=False,dropout=False):
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.latent_space_dim = latent_space_dim
        self.input_shape = input_shape
        self.input_encoder = None
        self.output_encoder = None
        self.shape_before_flattening = None
        self.normalize = normalize
        self.dropout = dropout
        self.mu = 0
        self.log_var = 0  # network output between [-inf,inf] so we use logarithm for compatibility
        self.for_VAE = for_VAE
        self.model = self.build_network()
        
        
    def sample_z(self,mu_var):
        # sample from the hidden variables z
        # using reparametrization trick to allow backprop through the encoder
        mu,log_var = mu_var
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1) # epsilon for sampling
        z = self.mu + K.exp(log_var / 2) * epsilon  # sample z
        return z
        
    def build_network(self):
        # build the model for the encoder
        self.input_encoder = Input(shape = self.input_shape, name="encoder_input")
        x = self.input_encoder
        
        # Stack convolutionnal layer 
        for i in range(self.num_layers):
            conv_layer = Conv2D(
                filters = self.num_filters[i],
                strides = self.strides[i],
                kernel_size = self.kernel_size,
                padding = "same",
                name = "encoder_conv_"+ str(i)
            )
            x = conv_layer(x)
            if self.normalize :
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.dropout :
                x = Dropout(0.25)(x)
        self.shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        
        # if using encoder for VAE
        if self.for_VAE :
            self.mu = Dense(self.latent_space_dim, name = "mu")(x) # mu head
            self.log_var = Dense(self.latent_space_dim,name = "std")(x) # log_var head
            self.output_encoder = Lambda(self.sample_z, name='encoder_out')([self.mu, self.log_var]) # sampling through lambda layer
        else :
            self.output_encoder = Dense(self.latent_space_dim, name = "encoder_out")(x)
        encoder = Model(self.input_encoder,self.output_encoder,name="encoder")
        return encoder
    

class Decoder():
    # Create an encoder for AutoEncoder or VariationnalAutoEncoder
    # Using Convolutionnal Layer
    # BatchNormalization and dropout is optionnal
    def __init__(self,num_layers,num_filters,kernel_size,input_shape,strides,latent_space_dim,normalize = False,dropout = False):
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.latent_space_dim = latent_space_dim
        self.input_shape = input_shape
        self.output_decoder = None
        self.normalize = normalize
        self.dropout = dropout
        self.model = self.build_network()

        
    def build_network(self):
       	# build the model for the encoder  
        input_layer = Input(shape = (self.latent_space_dim,), name="decoder_input")
        x = input_layer
        # Project and reshape
        x = Dense(units =np.prod(self.input_shape))(x)
        x = Reshape(self.input_shape)(x)
        
        # Stack TransposeConvolutionnal Layer
        for i in range(self.num_layers):
            deconv_layer = Conv2DTranspose(
                filters = self.num_filters[i],
                strides = self.strides[i],
                kernel_size = self.kernel_size,
                padding = "same",
                name = "decoder_deconv_"+ str(i)
            )
            x = deconv_layer(x)
            
            # if we are not stacking las layer
            if i < self.num_layers - 1 :
                if self.normalize :
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.dropout :
                    x = Dropout(0.25)(x)
            else :
            # Last layer
            # squeezing activation between valid image range (0 and 1)
                x = Activation('sigmoid')(x)
        self.output_decoder = x
        decoder = Model(input_layer,self.output_decoder, name = "decoder")
        return decoder


class AutoEncoder():
    def __init__(self,encoder,decoder,learning_rate):
        self.encoder = encoder
        self.decoder = decoder
        self.output_decoder = self.decoder.model(self.encoder.output_encoder)
        self.model = Model(self.encoder.input_encoder,self.output_decoder)
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer =self.optimizer, loss = "mean_squared_error" )
    
    def train(self,x_train,batch_size,epochs):
        self.model.fit(
            x=x_train, 
            y=x_train, # labels for reconstruction loss is the same as the input data
            batch_size = batch_size,
            shuffle = True,
            epochs = epochs,
        )


class VariationnalAutoEncoder():
    # Builde an Variationnal autoencoder
    # alpha and beta are used to balance the reconstruction_loss and kl_loss
    def __init__(self,encoder,decoder,learning_rate,alpha,beta=1):
        self.encoder = encoder
        self.decoder = decoder
        self.output_decoder = self.decoder.model(self.encoder.output_encoder)
        self.model = Model(self.encoder.input_encoder,self.output_decoder)
        self.optimizer = Adam(lr=learning_rate)
        self.alpha = alpha
        self.beta = beta
        self.model.compile(optimizer =self.optimizer, loss = self.compute_loss)
    
    
    def compute_loss(self,pred,real):
        r_loss = K.mean(K.square(real - pred), axis = [1,2,3])
        kl_loss = -0.5 * K.sum(1 + self.encoder.log_var - K.square(self.encoder.mu) - K.exp(self.encoder.log_var), axis = 1)
        vae_loss = self.alpha*r_loss + self.beta*kl_loss
        return vae_loss
    
    
    def train(self,x_train,batch_size,epochs):
        self.model.fit(
            x=x_train,
            y=x_train,
            batch_size = batch_size,
            shuffle = True,
            epochs = epochs,
        )