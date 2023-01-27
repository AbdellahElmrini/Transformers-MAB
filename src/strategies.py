import numpy as np
import torch
from abc import ABC, abstractmethod
from multiarmedbandits import MAB_normal

class ABCAgent(ABC):
    def __init__(self):
        self.mab = None
        self.rewards_avg = np.array([]) # Rewards average for each action
        self.visits = np.array([]) # Number of visits to each action
        self.record = {'actions':[], 'rewards':[] } # Dict listing actions and rewards
       
    @abstractmethod
    def take_action(self):
        ...

    def reinitialize(self):
        ...
    
class Agent(ABCAgent):
    def __init__(self, mab):
        super().__init__()
        self.reinitialize(mab)

    def reinitialize(self, mab = None):
        if mab is not None:
            self.mab = mab
        self.rewards_avg = np.zeros(self.mab.n)
        self.visits = np.zeros(self.mab.n)
        self.record = {'actions':[], 'rewards':[] }

    def take_N_actions(self, N):
        for _ in range(N):
            self.take_action()

class UniformAgent(Agent):
    """
    Agent taking completely random actions"""
    def __init__(self, mab):
        super().__init__(mab)

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
        super().__init__(mab)
        self.eps = eps
            
    def get_random_bandit(self):
        return np.random.randint(self.mab.n)
    
    def get_current_best_bandit(self):
        return np.random.choice(np.flatnonzero(self.rewards_avg == np.max(self.rewards_avg))) #To randomize tie breaks
        
    def take_action(self):
        p = np.random.rand()
        if p<self.eps:
            a = self.get_random_bandit()
        else:
            a = self.get_current_best_bandit()
        reward = self.mab.pull(a)
        Na = self.visits[a]+1
        self.visits[a] += 1
        self.rewards_avg[a] =  (Na-1)/Na*self.rewards_avg[a] + 1/Na * reward 
        self.record["actions"].append(a)
        self.record["rewards"].append(reward)
        return reward
    
class UCBAgent(Agent):
    """Upper Confidence Bound strategy"""
    def __init__(self, mab):
        super().__init__(mab)
    
    def get_current_best_bandit(self):
        N = sum(self.visits)
        #estimates = self.rewards_avg + np.sqrt(2*np.log(N)/self.visits) # Divisions by zero warnings
        estimates = np.zeros(self.mab.n)
        for a in range(self.mab.n):
            if self.visits[a] == 0:
                estimates[a] = np.inf
            else:
                estimates[a] = self.rewards_avg[a] + np.sqrt(2*np.log(N)/self.visits[a])
        return np.random.choice(np.flatnonzero(estimates == max(estimates))) #To randomize tie breaks

    def take_action(self):
        a = self.get_current_best_bandit()
        reward = self.mab.pull(a)
        Na = self.visits[a]+1
        self.visits[a] += 1
        self.rewards_avg[a] =  (Na-1)/Na*self.rewards_avg[a] + 1/Na * reward 
        self.record["actions"].append(a)
        self.record["rewards"].append(reward)
        return reward

class TransformerAgent(Agent):
    """
    Transformer learned strategy
    """
    def __init__(self, mab, model):
        """
        model should have a generate method that takes as input the actions and rewards
        and outputs the next action and reward
        """
        super().__init__(mab)
        self.model = model 
        self.device = next(self.model.parameters()).device
        self.initialized = False
    
    def reinitialize(self, mab = None):
        super().reinitialize(mab)
        self.initialized = False

    def take_action(self):
        #TODO : to(device) support
        if not self.initialized:
            starting_action = torch.zeros((1,1), dtype= torch.long).to(self.device)
            starting_reward = torch.zeros((1,1)).to(self.device)
            next_action, next_reward = self.model.generate(starting_action, starting_reward, 1)
            self.initialized = True
        else:
            starting_actions = torch.tensor(self.record["actions"], dtype= torch.long).unsqueeze(0).to(self.device)+1
            starting_rewards = torch.tensor(self.record["rewards"]).unsqueeze(0).to(self.device)
            next_action, next_reward = self.model.generate(starting_actions, starting_rewards, 1)
        #self.visits[next_action.item()] += 1
        self.record["actions"].append(next_action[0, -1].item() - 1)
        self.record["rewards"].append(next_reward[0, -1].item())
        return next_reward[0, -1].item()

    def take_N_actions(self, N):
        if not self.initialized:
            starting_action = torch.zeros((1,1), dtype= torch.long).to(self.device)
            starting_reward = torch.zeros((1,1)).to(self.device)
            next_actions, next_rewards = self.model.generate(starting_action, starting_reward, N)
            self.record["actions"].extend((next_actions[0, -N:]-1).tolist())   
            self.record["rewards"].extend( next_rewards[0, -N:].tolist())
            self.initialized = True
        else:
            # This was not tested, here for consistency only.
            starting_actions = torch.tensor(self.record["actions"], dtype = torch.long).unsqueeze(0).to(self.device) + 1
            starting_rewards = torch.tensor(self.record["rewards"]).unsqueeze(0).to(self.device)
            next_actions, next_rewards = self.model.generate(starting_actions, starting_rewards, N)
            self.record["actions"].extend( (next_actions[0, -N:]-1).tolist())
            self.record["rewards"].extend( next_rewards[0, -N:].tolist())
        return next_rewards[0,-N:]

if __name__ == "__main__":
    mab1= MAB_normal(n=5)
    ucb_agent = UCB1(mab1)
    ucb_agent.initialize()
    ucb_agent.take_N_actions(20)
    print(ucb_agent.record)