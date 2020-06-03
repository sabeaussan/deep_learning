import numpy as np
import gym
import torch
from policyGradientBis import Agent
import matplotlib.pyplot as plt

#def trainOneEpoch():



if __name__=="__main__":

    # Training parameters
    epochs=50

    # Nombre d'observation qu'on va faire avant de calculer la loss
    # On n'utilise pas un nombre de trajectoire prédéfinie 
    # car une trajectoire longue a beaucoup d'observation donc ça pourrait donnée des batch inégaux
    nb_episode_par_epoch=0             
    batch_size = 5000
    # Init training
    env = gym.make('CartPole-v0')
    obs = env.reset()
    action_history = []
    obs_dim = env.observation_space.shape[0]
    action_dims = env.action_space.n 
    actor = Agent(1e-2,obs_dim,128,128,action_dims)     # lr, obs_dim, fc1_dim, fc2_dim, action_dims
    critic = Agent(1e-2,obs_dim,128,128,1)              # lr, obs_dim, fc1_dim, fc2_dim, 1 car on prédit V(s)
    ac_cr = ActorCritic(actor,critic)
    ep_rews = []                                        # Reward totale pour cette épisode
    score = 0                                           # score atteint sur cete épisode
    obs_history = []                                    # liste des observation pour une epoch
    score_history = []                                  # liste des sccores pour une epoch

    print()

    # Begin training
    for i in range(epochs):
        
        while True:
            # save obs
            obs_history.append(obs.copy())              # On enregistre une première observation

            # On choisi une action et on l'applique à l'environment
            act = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
            action_history.append(act)
            # On récupère le nouvel état, la récompense associé à l'action et si le nouvel état est terminal
            obs, rew, done , _ = env.step(act)

            # On augmente le score obtenu par l'agent à cette épisode
            score += rew

            # On ajoute à la liste des récompense de cette épisode  
            ep_rews.append(rew)

            # Une fois l'épisode terminé :
            if done:      
                nb_episode_par_epoch += 1
                # On ajoute à la liste des récompenses de l'agent la Reward-to-go de chaque état
                agent.reward_memory += agent.computeRewardToGo(ep_rews)

                # On enregistre le score pour l'afficher à la fin de l'epoch
                score_history.append(score)

                # On reset les variable qui track cette épisode
                obs, done, ep_rews, score  = env.reset(), False, [], 0
                
                if len(obs_history) > batch_size:
                    break

        # A la fin de l'epoch, on a exécuté un certain nombre de trajectoire
        # donc on peut entrainer l'agent
        agent.learn(
            torch.as_tensor(obs_history, dtype=torch.float32),
            torch.as_tensor(action_history, dtype=torch.float32),
            torch.as_tensor(agent.reward_memory, dtype=torch.float32),
        )

        # On affiche les perfs liées à cette agent
        print('nb episode %d loss: %.3f \t return: %.3f' % (nb_episode_par_epoch,np.mean(agent.loss_memory.detach().numpy()), np.mean(score_history)))

        # On réinitialise les variables qui track cette epoch
        score_history = []
        agent.loss_memory = []
        action_history = []
        agent.reward_memory = []
        agent.action_memory = []
        obs_history = []
        nb_episode_par_epoch = 0
        


    