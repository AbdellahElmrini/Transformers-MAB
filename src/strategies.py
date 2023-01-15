import numpy as np
from abc import ABC, abstractmethod
from multiarmedbandits import MAB_normal

class Agent(ABC):
    def __init__(self):
        self.mab = None
        self.rewards = np.array([])
        self.visits = np.array([])
        self.actions = np.array([])
        self.record = {'actions':[], 'rewards':[] } # Dict listing actions and rewards
       
    @abstractmethod
    def take_action(self):
        ...
    
class Epsilon_greedy(Agent):
    """
    Epsilon greedy strategy
    """
    def __init__(self, mab, eps):
        super().__init__()
        self.mab = mab
        self.eps = eps
        self.visits = np.zeros(mab.n)
        self.rewards = np.zeros(mab.n)
            
    def get_random_bandit(self):
        return np.random.randint(self.mab.n)
    
    def get_current_best_bandit(self):
        return np.random.choice(np.flatnonzero(self.rewards == max(self.rewards))) #To randomize tie breaks
        
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
    """
    Upper Confidence Interval strategy"""
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


if __name__ == "__main__":
    mab1= MAB_normal(n=5)
    ucb_agent = UCB1(mab1)
    ucb_agent.initialize()
    ucb_agent.run_N_actions(20)
    print(ucb_agent.record)