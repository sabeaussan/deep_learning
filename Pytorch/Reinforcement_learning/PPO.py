import math
import random

import gym
import numpy as np
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from IPython.display import clear_output
import matplotlib.pyplot as plt
from utils import compute_return,compute_advantage
from models import ActorCritic

from multiprocessing_env import SubprocVecEnv

num_envs = 16
env_name = "Pendulum-v0"


# TODO : rajouter env reset

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        act = model.get_action(state)[0]
        next_state, reward, done, _ = env.step(act)
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        #print(states[rand_ids, :])
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):

    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            value = model.predict_value(state)
            entropy = model.get_dist(state).entropy().mean()
            new_log_probs = model.get_log_prob(state, action)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 1 * critic_loss + actor_loss - 0.001 * entropy

            model.actor_optimizer.zero_grad()
            model.critic_optimizer.zero_grad()
            loss.backward()
            model.actor_optimizer.step()
            model.critic_optimizer.step()

num_inputs  = envs.observation_space
num_outputs = envs.action_space

# Hyper-parameters
NB_STEP = 128
UPDATE_EPOCH = 10
MINI_BATCH_SIZE = 512
SIZES = [64]
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.2
#REWARD_ADAPTIVE_LR_THRESHOLD = -300
REWARD_THRESHOLD = 190


model = ActorCritic(num_inputs, num_outputs, SIZES)


frame_idx  = 0
test_rewards = []
#env_render = False


state = envs.reset()
early_stop = False
PATH = "saved_models/model_ppo_pendulum.pt"

while not early_stop :

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []

    for _ in range(NB_STEP):
        state = torch.FloatTensor(state)
        value = model.predict_value(state)
        action = model.get_action(state)
        action = action.squeeze(0)
        next_state, reward, done, _ = envs.step(action)
        log_prob = model.get_log_prob(state, action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1))
        
        states.append(state)
        actions.append(action)
        
        state = next_state
        frame_idx += 1


        if frame_idx % 1000 == 0:
            test_reward = np.mean([test_env() for _ in range(10)])
            test_rewards.append(test_reward)
            #plot(frame_idx, test_rewards)
            print("test reward: ")
            print(test_reward)
            if test_reward > REWARD_THRESHOLD :
                early_stop = True
                torch.save(model.state_dict(), PATH)
                



            

    next_state = torch.FloatTensor(next_state)
    next_value = model.predict_value(next_state)

    returns = compute_advantage(next_value, rewards, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values


    
    ppo_update(UPDATE_EPOCH, MINI_BATCH_SIZE, states, actions, log_probs, returns, advantage)
