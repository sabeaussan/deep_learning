import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F



def create_body( sizes, activation=nn.ReLU, output_activation=nn.Identity):
	layers = []
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes)-2 else output_activation
		layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
	return nn.Sequential(*layers)



class ActorCritic(nn.Module):


	def __init__(self,observation_space,  action_space, sizes, std=0.0 ):
		super(ActorCritic, self).__init__()

		self.observation_space = observation_space
		self.action_space = action_space

		if action_space.__class__.__name__ == "Discrete":
			action_dim = action_space.n
		elif action_space.__class__.__name__ == "Box":
			action_dim = action_space.shape[0]
			self.log_std = nn.Parameter(torch.ones(1, action_dim) * std)
		else:
			raise NotImplementedError

		sizes.insert(0,observation_space.shape[0])
		sizes_actor = sizes + [action_dim]
		sizes_critic = sizes + [1]

		self.actor = create_body(sizes_actor)
		self.critic = create_body(sizes_critic)

		self.log_std = nn.Parameter(torch.ones(1, action_dim) * std)


		self.actor_optimizer = Adam(self.parameters(), lr=7e-4)
		self.critic_optimizer = Adam(self.parameters(), lr=7e-4)

	def forward(self, x):
		value = self.critic(x)
		mu    = nn.tanh(self.actor(x))
		std   = self.log_std.exp().expand_as(mu)
		dist  = Normal(mu, std)
		return dist, value

	def predict_value(self,state):
		return self.critic(state)

	def get_dist(self,state):
		out = self.actor(state)
		if self.action_space.__class__.__name__ == "Discrete":
			dist = Categorical(logits = out)
		elif self.action_space.__class__.__name__ == "Box":
			mean = torch.tanh(out)
			std   = self.log_std.exp().expand_as(out)
			dist = Normal( loc = mean , scale = std)
		else:
			raise NotImplementedError
		return dist

	def get_action(self,state):
		return self.get_dist(state).sample()

	def get_log_prob(self,state,action):
		return self.get_dist(state).log_prob(action)



class ActorDDPG(nn.Module):

	def __init__(self, observation_space,  action_space, hidden_size_1,hidden_size_2,init_1,init_2,init_3):
		super(ActorDDPG, self).__init__()
		
		self.observation_space = observation_space
		self.action_space = action_space
		if action_space.__class__.__name__ == "Discrete":
			action_dim = action_space.n
		elif action_space.__class__.__name__ == "Box":
			action_dim = action_space.shape[0]
			self.log_std = nn.Parameter(torch.zeros(1,action_dim))
		else:
			raise NotImplementedError

		self.linear1 = nn.Linear(observation_space.shape[0], hidden_size_1)
		nn.init.uniform_(self.linear1.weight.data,-init_1,init_1)
		self.linear1.bias.data.fill_(0.01)

		self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
		nn.init.uniform_(self.linear2.weight.data,-init_2,init_2)
		self.linear2.bias.data.fill_(0.01)

		self.linear3 = nn.Linear(hidden_size_2, action_dim)
		nn.init.uniform_(self.linear3.weight.data,-init_3,init_3)
		self.linear3.bias.data.fill_(0.01)
		
		



	def forward(self,state):
		output = self.linear1(state)
		output = self.linear2(output)
		output = self.linear3(output)
		return output


	def get_action(self,state):
		return torch.tanh(self.forward(state))		# pour le squeeze entre -1 et 1




class CriticDDPG(nn.Module):
	def __init__(self,observation_space, action_space, hidden_size_1,hidden_size_2,init_1,init_2,init_3 ):
		super(CriticDDPG, self).__init__()

		if action_space.__class__.__name__ == "Discrete":
			action_dim = action_space.n
		elif action_space.__class__.__name__ == "Box":
			action_dim = action_space.shape[0]
		else:
			raise NotImplementedError

		self.linear1 = nn.Linear(observation_space.shape[0]+action_dim, hidden_size_1)
		nn.init.uniform_(self.linear1.weight.data,-init_1,init_1)
		self.linear1.bias.data.fill_(0.01)

		self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
		nn.init.uniform_(self.linear2.weight.data,-init_2,init_2)
		self.linear2.bias.data.fill_(0.01)

		self.linear3 = nn.Linear(hidden_size_2, 1)
		nn.init.uniform_(self.linear3.weight.data,-init_3,init_3)
		self.linear3.bias.data.fill_(0.01)



	def predict_value(self,state, action):
		input_ = torch.cat([state,action],1)
		output = self.linear1(input_)
		output = self.linear2(output)
		output = self.linear3(output)
		return output

class AgentDDPG(object):

	def __init__(self,observation_space,  action_space, hidden_size_1, hidden_size_2 = None, init_1 = 0, init_2 = 0, init_3 = 0.003 ):
		super(AgentDDPG, self).__init__()
		self.init_1 = 1/np.sqrt(hidden_size_1)
		if hidden_size_2 is None :
			hidden_size_2 = hidden_size_1
		self.init_2 = 1/np.sqrt(hidden_size_2)
		self.init_3 = init_3
		self.actor = ActorDDPG(observation_space, action_space, hidden_size_1,hidden_size_2,self.init_1,self.init_2,self.init_3)
		self.critic = CriticDDPG(observation_space, action_space, hidden_size_1,hidden_size_2,self.init_1,self.init_2,self.init_3)
		self.target_actor = ActorDDPG(observation_space, action_space, hidden_size_1,hidden_size_2,self.init_1,self.init_2,self.init_3)
		self.target_critic = CriticDDPG(observation_space, action_space, hidden_size_1,hidden_size_2,self.init_1,self.init_2,self.init_3)
		#self.init_weight(self.actor.network)
		#self.init_weight(self.actor.head)
		#self.init_weight(self.critic.network)
		for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
			target_param.data.copy_(param.data)

		for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
			target_param.data.copy_(param.data)

		self.optimizer_actor = Adam(self.actor.parameters(), lr=1e-4)
		self.optimizer_critic = Adam(self.critic.parameters(), lr=1e-3)

	"""def init_weight(self,net):
		index = 0
		for m in enumerate(net.modules()):
			#print(m)
			if isinstance(m, nn.Linear):
				if index == 0 :
					print(self.init_1)
					nn.init.uniform_(m.weight.data,-init_1,init_1)
					index+=1
					m.bias.data.fill_(0.01)
				elif index == 1 :
					nn.init.uniform_(m.weight.data,-init_2,init_2)
					m.bias.data.fill_(0.01)
					index+=1
				else :
					nn.init.uniform_(m.weight.data,-init_3,init_3)
					m.bias.data.fill_(0.01)"""
			
	




