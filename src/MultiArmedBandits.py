import numpy as np
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from collections import defaultdict
import time

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class MAB(ABC):
    def __init__(self):
        self.n = None

    @abstractmethod
    def pull(self, a):
        ...

class MAB_normal(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form N(a,1)
        """
        super().__init__()
        self.n = n
    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        return np.random.rand()+ a

class Agent(ABC):
    def __init__(self):
        self.mab = None
        self.rewards = np.array([])
        self.visits = np.array([])
        self.record = {'actions':[], 'rewards':[] } # Dict listing actions and rewards
        # List containing (a_1,r_1,a_2,r_2, ..)
       
    @abstractmethod
    def take_action(self):
        ...
    
class Epsilon_greedy(Agent):
    def __init__(self, mab, eps):
        super().__init__()
        self.mab = mab
        self.eps = eps
        self.visits = np.zeros(mab.n)
        self.rewards = np.zeros(mab.n)
            
    def get_random_bandit(self):
        return np.random.randint(self.mab.n)
    
    def get_current_best_bandit(self):
        return np.random.choice(np.flatnonzero(self.rewards == max(self.rewards))) #To randomize tie break
        
    def take_action(self):
        p = np.random.rand()
        if p<self.eps:
            a = self.get_random_bandit()
        else:
            a = self.get_current_best_bandit()
        reward = self.mab.pull(a)
        Na = self.visits[a]+1
        self.visits[a] += 1
        self.rewards[a] =  (Na-1)/Na*self.rewards[a] + 1/Na * reward 
        self.record["actions"].append(a)
        self.record["rewards"].append(reward)
        return reward
    
class UCB1(Agent):
    def __init__(self, mab):
        super().__init__()
        self.mab = mab
        self.visits = np.zeros(mab.n)
        self.rewards = np.zeros(mab.n)
        self.initialized = False
    
    def get_current_best_bandit(self):
        N = sum(self.visits)
        estimates = self.rewards + np.sqrt(2*np.log(N)/self.visits)
        return np.random.choice(np.flatnonzero(estimates == max(estimates))) #To randomize tie breaks
    
    def initialize(self):
        if self.initialized:
            pass
        else:
            L = np.arange(self.mab.n)
            np.random.shuffle(L)
            for a in L:
                reward = self.mab.pull(a)
                self.visits[a] = 1
                self.rewards[a] = reward
                self.record["actions"].append(a)
                self.record["rewards"].append(reward)
            self.initialized = True
    
    def take_action(self):
        if not self.initialized:
            raise Exception('Initialisation step needs to be executed first.')
        a = self.get_current_best_bandit()
        reward = self.mab.pull(a)
        Na = self.visits[a]+1
        self.visits[a] += 1
        self.rewards[a] = (Na-1)/Na*self.rewards[a] + 1/Na * reward 
        self.record["actions"].append(a)
        self.record["rewards"].append(reward)
        return reward
    
    def run_N_actions(self, N):
        for _ in range(N):
            self.take_action()