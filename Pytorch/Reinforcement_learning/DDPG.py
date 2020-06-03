import numpy as np
import gym
import torch

from model import AgentDDPG
import matplotlib.pyplot as plt
from torch.optim import Adam
from utils import OUNoise,ReplayFIFO
import torch.nn.functional as F

#TODO : soft + hard update

# Batch norm -> permet de réuire l'influence de la différence entre les vecteurs d'observation de différent env
# donc permet de mieux généraliser



# Hyper-parameters
UPDATE_EPOCH = 20
SIZES = [32]
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.01
NB_STEP = 400
ACTION_BOUND = 2.0
REWARD_THRESHOLD = -200
#init weight


def test(model,num_test):
	score_test = 0
	done_test = True
	env_test = gym.make('Pendulum-v0')
	obs_test = env_test.reset()
	score_test = 0
	reached = False
	done_test = False
	while not done_test :   
		obs_test = torch.FloatTensor(obs_test)  
		act_test = model.actor.get_action(obs_test).detach() * ACTION_BOUND 
		obs_test, rew_test, done_test , _ = env_test.step(act_test)
		score_test += rew_test
	if score_test >= REWARD_THRESHOLD:
		print("REWARD_THRESHOLD")
		reached = True
	print(" test numéro %d score %.3f" % (num_test, score_test))
	return reached




if __name__=="__main__":

	env = gym.make('Pendulum-v0')
	state_dim = env.observation_space
	action_dim = env.action_space
	model = AgentDDPG(state_dim,action_dim,SIZES)
	state = env.reset()
	fifo = ReplayFIFO(50000)
	noise = OUNoise(action_dim)
	done = False
	early_stop = False
	train_epoch = 0
	rewards = 0

	while not early_stop : 
		
		# Sample trajectories 														
		for step in range(NB_STEP):   
				 
			action = model.actor.get_action(torch.FloatTensor(state)) * ACTION_BOUND  		# récupère l'action de l'agent et multiplie par la range de l'action
			action = noise.get_action(action.detach().numpy(),step)				   			# ajout de bruit pour l'exploration
			next_state, reward, done, _ = env.step(action)        							# avance dans l'environment

			# On record les données de ce time_step
			fifo.push(state,action,reward,next_state,done)
			rewards += reward
			state = next_state

			if len(fifo) > BATCH_SIZE :

				states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = fifo.sample(BATCH_SIZE)

				states_batch = torch.FloatTensor(states_batch)
				rewards_batch = torch.FloatTensor(rewards_batch).unsqueeze(-1)
				next_states_batch = torch.FloatTensor(next_states_batch)
				done_batch = torch.FloatTensor(done_batch).unsqueeze(-1)
				actions_batch = torch.FloatTensor(actions_batch)

				# Critic loss 
				action_target = model.target_actor.get_action(next_states_batch) * ACTION_BOUND
				q_targets = rewards_batch + (1- done_batch) * GAMMA * model.target_critic.predict_value(next_states_batch,action_target.detach()).detach()
				q_values = model.critic.predict_value(states_batch,actions_batch)
				loss_value = F.mse_loss(q_targets,q_values)
				#print(loss_value)

				# Actor loss
				actions = model.actor.get_action(states_batch) * ACTION_BOUND
				loss_policy = -1 * model.critic.predict_value(states_batch,actions).mean()

				model.optimizer_actor.zero_grad()
				model.optimizer_critic.zero_grad()

				# Update
				loss_value.backward()
				loss_policy.backward()

				model.optimizer_critic.step()
				model.optimizer_actor.step()

				for target_param, param in zip(model.target_actor.parameters(), model.actor.parameters()):
					target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

				for target_param, param in zip(model.target_critic.parameters(), model.critic.parameters()):
					target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

			if done :
				state = env.reset()
				noise.reset()
				print(rewards)
				rewards = 0
				break;


			

		train_epoch += 1
		if train_epoch % 10 == 0 : 
			early_stop = test(model, train_epoch//10)

























