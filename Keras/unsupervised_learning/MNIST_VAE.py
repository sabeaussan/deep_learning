from models import Encoder,Decoder,VariationnalAutoEncoder
from keras.datasets import mnist
import matplotlib.pyplot as plt


enc = Encoder(
    (28,28,1),
    4,
    [32,64,64,64],
    3,
    [1,2,2,1],
    10,
    True,
    True,
    True
)

dec = Decoder(
    4,
    [64,64,32,1],
    3,
    enc.shape_before_flattening,
    [1,2,2,1],
    enc.latent_space_dim
)

vae = VariationnalAutoEncoder(enc,dec,0.0005,1000)

vae.encoder.model.summary()
vae.decoder.model.summary()

ae.train(x_train[:1000].reshape(-1,28,28,1),64,50)

