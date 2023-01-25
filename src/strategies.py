import numpy as np
import torch
from abc import ABC, abstractmethod
from multiarmedbandits import MAB_normal

class ABCAgent(ABC):
    def __init__(self):
        self.mab = None
        self.criterion = np.array([]) # Crtierion used to choose the action
        self.visits = np.array([]) # Number of visits to each action
        self.record = {'actions':[], 'rewards':[] } # Dict listing actions and rewards
       
    @abstractmethod
    def take_action(self):
        ...

    def reinitialize(self):
        ...
    
class Agent(ABCAgent):
    def __init__(self):
        super().__init__()

    def reinitialize(self):
        self.criterion = np.zeros(self.mab.n)
        self.visits = np.zeros(self.mab.n)
        self.record = {'actions':[], 'rewards':[] }
    
    def initialize(self):
        # Necessary only for some strategies, such as UCB
        pass

    def run_N_actions(self, N):
        # TODO : Change name to take_N_actions
        for _ in range(N):
            self.take_action()

class UniformAgent(Agent):
    "Completely random actions"
    def __init__(self, mab):
        super().__init__()
        self.mab = mab
        self.visits = np.zeros(mab.n)
        self.criterion = np.zeros(mab.n)
    def take_action(self):
        action = np.random.randint(self.mab.n)
        reward = self.mab.pull(action)
        self.record["actions"].append(action)
        self.record["rewards"].append(reward)
        return reward

    

class EpsilonGreedy(Agent):
    """
    Epsilon greedy strategy
    """
    def __init__(self, mab, eps):
        super().__init__()
        self.mab = mab
        self.eps = eps
        self.visits = np.zeros(mab.n)
        self.criterion = np.zeros(mab.n)
            
    def get_random_bandit(self):
        return np.random.randint(self.mab.n)
    
    def get_current_best_bandit(self):
        return np.random.choice(np.flatnonzero(self.criterion == np.max(self.criterion))) #To randomize tie breaks
        
    def take_action(self):
        p = np.random.rand()
        if p<self.eps:
            a = self.get_random_bandit()
        else:
            a = self.get_current_best_bandit()
        reward = self.mab.pull(a)
        Na = self.visits[a]+1
        self.visits[a] += 1
        self.criterion[a] =  (Na-1)/Na*self.criterion[a] + 1/Na * reward 
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
        self.criterion = np.zeros(mab.n)
        self.initialized = False
    
    def reinitialize(self):
        self.visits = np.zeros(self.mab.n)
        self.criterion = np.zeros(self.mab.n)
        self.initialized = False
        self.record["actions"] = []
        self.record["rewards"] = []

    def get_current_best_bandit(self):
        N = sum(self.visits)
        estimates = self.criterion + np.sqrt(2*np.log(N)/self.visits)
        #return np.random.choice(np.flatnonzero(estimates == max(estimates))) #To randomize tie breaks
        return np.argmax(estimates)

    def initialize(self):
        if not self.initialized:
            for a in range(self.mab.n):
                reward = self.mab.pull(a)
                self.visits[a] = 1
                self.criterion[a] = reward
                self.record["actions"].append(a)
                self.record["rewards"].append(reward)
            self.initialized = True
    def initialize0(self):
        if not self.initialized:
            L = np.arange(self.mab.n)
            np.random.shuffle(L)
            for a in L:
                reward = self.mab.pull(a)
                self.visits[a] = 1
                self.criterion[a] = reward
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
        self.criterion[a] = (Na-1)/Na*self.criterion[a] + 1/Na * reward 
        self.record["actions"].append(a)
        self.record["rewards"].append(reward)
        return reward


class TransformerAgent(Agent):
    """
    Transformer learned strategy
    """
    def __init__(self, mab, model):
        super().__init__()
        self.mab = mab
        self.model = model
        self.visits = np.zeros(mab.n)
        self.criterion = np.zeros(mab.n)
        self.initialized = False
    
    def take_action(self):
        #TODO : to(device) support
        if not self.initialized:
            action, reward = self.model.generate(torch.zeros((1,1), dtype= torch.int), torch.zeros((1,1)), 1) 
            self.initialized = True
        else:
            action, reward = self.model.generate(self.record["actions"], self.record["rewards"], 1)
        return reward.item()
    
    def reinitialize(self):
        self.initialized = False
        self.record["actions"] = []
        self.record["rewards"] = []

    def run_N_actions(self, N):
        if not self.initialized:
            device = next(self.model.parameters()).device
            starting_action = torch.zeros((1,1), dtype= torch.int).to(device)
            starting_reward = torch.zeros((1,1)).to(device)
            next_actions, next_rewards = self.model.generate(starting_action, starting_reward, N)
            self.record["actions"].extend( next_actions[0, -N:].tolist())   
            self.record["rewards"].extend( next_rewards[0, -N:].tolist())
            self.initialized = True
        else:
            # This was not tested, here for consistency only.
            actions = torch.Tensor(self.record["actions"]).unsqueeze(0)
            rewards = torch.Tensor(self.record["rewards"]).unsqueeze(0)
            next_actions, next_rewards = self.model.generate(actions, rewards, N)
            self.record["actions"].extend( next_actions[-N:].tolist())   
            self.record["rewards"].extend( next_rewards[-N:].tolist())
        return next_rewards[-N:]


if __name__ == "__main__":
    mab1= MAB_normal(n=5)
    ucb_agent = UCB1(mab1)
    ucb_agent.initialize()
    ucb_agent.run_N_actions(20)
    print(ucb_agent.record)