import numpy as np
import torch
import math
from torch import nn, Tensor
import time
from matplotlib import pyplot as plt

from dataset import UCB_dataset



def plot_regret(rewards, mab , filename = "Figs/regret.png"):
    """
    Plot the regret from the rewards
    """
    T = rewards.size(1)
    best_reward_avg = mab.best_reward_avg
    y = best_reward_avg - 1/torch.arange(1, T+1) * torch.cumsum(rewards, dim=1).mean(0) 
    plt.plot(y)
    plt.title("Regret")
    plt.savefig(filename)


def plot_regrets(datasets_list, mab, names):
    """
    Plot the regret from the rewards
    """
    T = datasets_list[0].rewards.size(1)
    best_reward_avg = mab.best_reward_avg
    for i, dataset in enumerate(datasets_list):
        rewards = dataset.rewards
        y = best_reward_avg * torch.arange(1, T+1) - torch.cumsum(rewards, dim=1).mean(0) 
        plt.plot(y, label = names[i])
    plt.legend()
    plt.title("Regret comparison" + ' '.join(names))
    plt.savefig("Figs/regrets/regrets"+'_'.join(names)+".png")
    plt.clf()


def compute_regret(rewards, mab):
    """
    Compute the regret from the rewards
    """
    T = rewards.size(1)
    best_reward_avg = mab.best_reward_avg
    y =  best_reward_avg*torch.arange(1, T+1) -    torch.cumsum(rewards, dim=1).mean(0) 
    return y

def compute_regret_normalized(rewards, mab):
    """
    Compute the regret from the rewards
    """
    T = rewards.size(1)
    best_reward_avg = mab.best_reward_avg
    y =  best_reward_avg -   1/torch.arange(1, T+1) * torch.cumsum(rewards, dim=1).mean(0) 
    return y


def compute_reward(rewards):
    """
    Compute the reward from the rewards
    """
    T = rewards.size(1)
    y =  torch.cumsum(rewards, dim=1).mean(0) 
    return y

def compute_best_action_selection_rate(actions, best_action):
    """
    Compute the best action selection rate from the actions
    """
    m = actions.size(0) # number of sequences
    y =  torch.sum(actions == best_action+1, dim = 0)/m
    return y
    

def plot_best_action_selection_rate(datasets_list, mab, names):
    """
    Plot the best action selection rate from the actions
    """
    T = datasets_list[0].rewards.size(1)
    best_action = mab.best_action
    for i, dataset in enumerate(datasets_list):
        actions = dataset.actions
        y = compute_best_action_selection_rate(actions, best_action)
        plt.plot(y, label = names[i])
    plt.legend()
    plt.title("Best action selection rate comparison" + ' '.join(names))
    plt.savefig("Figs/basr/basr"+'_'.join(names)+".png")
    plt.clf()


def plot_reward(actions, rewards, filename = "Figs/reward.png"):
    """
    Plot the cumulative sum of rewards
    """
    y = torch.cumsum(rewards, dim = 1).mean(0)
    plt.plot(y , label = 'Rewards')
    plt.savefig(filename)



    


