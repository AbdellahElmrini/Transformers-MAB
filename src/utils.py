import numpy as np
import torch
import math
from torch import nn, Tensor
import time
from matplotlib import pyplot as plt

from dataset import UCB_dataset




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
    Compute the normalized regret from the rewards
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

def compute_best_action_selection_rate(actions, mab):
    """
    Compute the best action selection rate from the actions
    """
    m = actions.size(0) # number of sequences
    best_action = mab.best_action
    y =  torch.sum(actions == best_action+1, dim = 0)/m
    return y

def plot_best_action_selection_rate(datasets_list, mab, names, suffix = ''):
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
    plt.title("Best Action Selection Rate Comparison ") 
    for i in range(len(names)):
        names[i] = names[i][0:3]
    plt.savefig("Figs/basr/basr"+'_'.join(names)+suffix+".png")
    plt.clf()

def plot_regrets(datasets_list, mab, names, suffix = ''):
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
    plt.title("Regret Comparison ")
    for i in range(len(names)):
        names[i] = names[i][0:3]
    plt.savefig("Figs/regrets/regrets"+'_'.join(names)+suffix+".png")
    plt.clf()





    


