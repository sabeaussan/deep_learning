from models import Encoder,Decoder,VariationnalAutoEncoder
from utils import create_training_data



PATH = "/Users/admin/Documents/ml_dev/Pytorch/deep_learning/data_sets/pokedigi/mix_jpg"
IMG_SIZE = 64
training_data = create_training_data(PATH,IMG_SIZE)
random.shuffle(training_data)
training_data = np.array(training_data)
training_data =training_data/ 255.0 

# Create encoder
enc = Encoder(
    (64,64,3),
    4,
    [32,64,64,64],
    3,
    [1,2,2,1],
    200,
    True,
    True,
    True
)

# Create decoder
dec = Decoder(
    4,
    [64,64,32,3],
    3,
    enc.shape_before_flattening,
    [1,2,2,1],
    enc.latent_space_dim,
    True,
    True
)

vae = VariationnalAutoEncoder(enc,dec,0.0005,1000) 
vae.encoder.model.summary()
vae.decoder.model.summary()
vae.train(training_data,32,50)