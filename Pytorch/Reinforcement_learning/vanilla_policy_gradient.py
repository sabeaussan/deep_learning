import numpy as np
import gym
import torch
from model_discrete import ActorCritic,Actor,Critic
import matplotlib.pyplot as plt

#def trainOneEpoch():


def plot_result(nb_episode,reward):
    fig = plt.figure()
    plt.plot(nb_episode, reward, color='blue')
    plt.show()
    fig


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
    ac_cr = ActorCritic(1e-2,obs_dim,128,128,action_dims)
    ep_rews = []                                        # Reward totale pour cette épisode
    ep_val = []
    score = 0                                           # score atteint sur cete épisode
    obs_history = []                                    # liste des observation pour une epoch
    score_history = []                                  # liste des sccores pour une epoch
    value_history = []                                  # liste des V(s) pour une epoch
    advantage_history = []
    nb_episode = 0
    buffer_score = []


    # Begin training
    for i in range(epochs):
        
        while True:
            # save obs
            obs_history.append(obs.copy())              # On enregistre une première observation

            # L'acteur choisis une action et l'applique sur l'environment
            # Le critique prédit une valeur qui servira pour l'estimation de la fonction objective
            act = ac_cr.actor.get_action(torch.as_tensor(obs, dtype=torch.float32))
            val = ac_cr.critic.get_value(torch.as_tensor(obs, dtype=torch.float32))
            action_history.append(act)
            ep_val.append(val)
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
                ac_cr.actor.reward_memory += ac_cr.computeRewardToGo(ep_rews)

                # Calcul les advantages pour l'épisode
                ac_cr.advantage_memory += ac_cr.computeAdvantage(ac_cr.actor.reward_memory,ep_val)

                # On enregistre le score pour l'afficher à la fin de l'epoch
                score_history.append(score)

                #
                value_history += ep_val

                # On reset les variable qui track cette épisode
                obs, done, ep_rews, score, ep_val  = env.reset(), False, [], 0, []
                
                if len(obs_history) > batch_size:
                    break

        # A la fin de l'epoch, on a exécuté un certain nombre de trajectoire
        # donc on peut entrainer l'agent
        
        ac_cr.learnPolicy(
            torch.as_tensor(obs_history, dtype=torch.float32),
            torch.as_tensor(action_history, dtype=torch.float32),
            torch.as_tensor(ac_cr.advantage_memory, dtype=torch.float32),
        )
        ac_cr.learnValue(
            value_history,
            ac_cr.actor.reward_memory
        )

        # On affiche les perfs liées à cette agent
        print('nb episode %d \t loss_value: %.3f \t loss_policy: %.3f \t score: %.3f' % 
            (nb_episode_par_epoch,np.mean(ac_cr.critic.loss_memory.detach().numpy()),np.mean(ac_cr.actor.loss_memory.detach().numpy()), np.mean(score_history))
            )

        # On réinitialise les variables qui track cette epoch
        buffer_score += score_history
        score_history = []
        ac_cr.actor.loss_memory = []
        ac_cr.critic.loss_memory = []
        action_history = []
        ac_cr.actor.reward_memory = []
        obs_history = []
        value_history = []
        nb_episode += nb_episode_par_epoch
        nb_episode_par_epoch = 0
        ac_cr.advantage_memory = []
    nb_episode = nb_episode* [0]
    for i in range(len(nb_episode)):
        nb_episode[i] = i
    plot_result(nb_episode,buffer_score)

        


    