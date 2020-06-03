import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F



class ReplayFIFO:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0
	
	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity :
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity
	
	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done
	
	def __len__(self):
		return len(self.buffer)

class OUNoise(object):
	def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.2, decay_period=100000):
		self.mu           = mu
		self.theta        = theta
		self.sigma        = max_sigma
		self.max_sigma    = max_sigma
		self.min_sigma    = min_sigma
		self.decay_period = decay_period
		self.action_dim   = action_space.shape[0]
		self.low          = action_space.low
		self.high         = action_space.high
		self.reset()
		
	def reset(self):
		self.state = np.ones(self.action_dim) * self.mu
		
	def evolve_state(self):
		x  = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
		self.state = x + dx
		return self.state
	
	def get_action(self, action, t=0):
		ou_state = self.evolve_state()
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
		return np.clip(action + ou_state, self.low, self.high)        


def print_weights(model):
	for param in model.parameters():
		print(param.data)


"""def init_weights(m):
	if isinstance(m, nn.Linear):
		nn.init.normal_(m.weight, mean=0., std=0.1)
		nn.init.constant_(m.bias, 0.1)"""
	

def compute_return(reward,gamma = 0.99):
	# Caclul la rewardToGo de tous les états pour un épisode
	G = []
	for t in range(len(reward)):
		G_sum = 0
		discount = 1
		for k in range(t,len(reward)):
			#Pour les timeStep restante à partir de t, on calcule la récompense totale obtenue
			G_sum += discount*reward[k]
			discount *= gamma
		G.append(G_sum)
	return G


def plot(frame_idx, rewards):
	clear_output(True)
	plt.figure(figsize=(20,5))
	plt.subplot(131)
	plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
	plt.plot(rewards)
	plt.show()

def compute_advantage(next_value,rewards, values,gamma= 0.99,lambda_ = 0.95, normalize = False):
	val = values + [next_value]
	advantages = []  
	gae = 0
	for i in range(len(rewards),0,-1):
		delta = rewards[i-1] +  gamma * val[i] - val[i-1]
		gae = delta + gae * gamma * lambda_ 
		advantages.insert(0,gae)
	if normalize :
		std =advantages.std() 
		advantages = (advantages-advantages.mean())/(advantages.std()+ 1e-5)   # Normalize pour diminuer la variance
	return advantages



def build_dataset(load_path,save_path,dataset_size):
    filenames = glob.glob(load_path)
    for iteration in range(dataset_size):
        img = Image.open(filenames[iteration])
        img = img.resize((64,64))
        img.save('{}{}{}'.format(save_path,iteration,".jpg"))
        img.close()
    print("done")

def build_dataset_jpg(load_path,save_path):
	# Remove alpha channel and convert to JPG format
    filenames = glob.glob(load_path)
    print(len(filenames))
    for iteration in range(len(filenames)):
        img = Image.open(filenames[iteration])
        img.load() # required for png.split()
        background = Image.new("RGB", img.size, (255, 255, 255))
        if(len(img.split()) == 4): 
            background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
            background.save('{}{}{}'.format(save_path,iteration,".jpg"), quality=80)
            background.close()
        img.close()
    print("done")

def build_batch_tensor(dataset,batch_size):
	# Given a dataset, yield batch of batch_size
    nb_batch = len(dataset)//batch_size +1
    start = 0
    tmp = 0
    end = batch_size
    for i in range(nb_batch):
        tmp += batch_size
        if i != 0 :
            if tmp > len(dataset):
                end = len(dataset) - tmp + batch_size
            else :
                end = batch_size
            start += batch_size
        yield torch.narrow(dataset,0,start,end)

def normalize(dataset,scale):
	# Normalize and scale given dataset
    s = dataset
    s = (s/255.0)
    s = (s-scale)/scale
    return s

def load_dataset(load_path):
    loaded_img = []
    filenames = glob.glob(load_path)
    #cpt = 0
    for iteration in range(len(filenames)):
        #if cpt% 10000 == 0:
        #    print(cpt)
        #cpt+=1
        img = Image.open(filenames[iteration])
        img = img.resize((64,64))
        print(np.asarray(img).shape)
        loaded_img.append(np.asarray(img))
        img.close()
    print("done")
    return loaded_img

def store_many_hdf5(images):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    file.close()
    print("done")

def rescale_image(image,scale):
	# rescale image to be plotted
    return (image*scale)+scale

def plot_image(image,transpose=False):
	# Transpose image if channel is not at the right axis
    plt.subplot(1,2,2)
    plt.axis("off")
    if transpose : 
        t_image = np.transpose(image,(1,2,0))
        plt.imshow(t_image)
    else:
        plt.imshow(image)
    plt.show()

"""def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
	values = values + [next_value]
	gae = 0
	returns = []
	for step in reversed(range(len(rewards))):
		delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
		gae = delta + gamma * tau * masks[step] * gae
		returns.insert(0, gae + values[step])
	return returns"""

