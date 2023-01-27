import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from multiarmedbandits import MAB_normal
from strategies import EpsilonGreedy, UCBAgent


class UCB_dataset(Dataset):
    """
    Dataset of UCB strategy runs
    """
    def __init__(self, mab, N, T):
        """
        mab : A MAB class instance used to generate the rewards.
        N : Dataset size.
        T : Horizon.
        """
        self.mab = mab
        self.N = N
        self.T = T
        self.n_arms = self.mab.n
        self.actions = torch.zeros([N, T+1], dtype = torch.long)
        self.rewards = torch.zeros([N, T+1], dtype = torch.float)
        for i in range(N):
            ucb_agent = UCBAgent(mab)
            ucb_agent.take_N_actions(T) 
            self.actions[i] = torch.cat((torch.Tensor([0]), torch.Tensor(ucb_agent.record["actions"])+1)) #0 is our $<bos>$
            self.rewards[i] = torch.cat((torch.Tensor([0]), torch.Tensor(ucb_agent.record["rewards"])))
            
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        return self.actions[idx], self.rewards[idx]

class Strategy_dataset(Dataset):
    """
    Dataset of a strategy runs
    """
    def __init__(self, agent, mab, N, T):
        """
        agent (Agent) : The strategy to be executed
        mab : A MAB class instance used to generate the rewards.
        N : Dataset size.
        T : Horizon.
        """
        self.mab = mab
        self.agent = agent
        self.N = N
        self.T = T
        self.n_arms = self.mab.n
        self.actions = torch.zeros([N, T+1], dtype = torch.long)
        self.rewards = torch.zeros([N, T+1], dtype = torch.float)
        for i in range(N):
            agent.reinitialize()
            agent.take_N_actions(T) 
            self.actions[i] = torch.cat((torch.Tensor([0]), torch.Tensor(agent.record["actions"])+1)) #0 is our $<bos>$
            self.rewards[i] = torch.cat((torch.Tensor([0]), torch.Tensor(agent.record["rewards"])))
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        return self.actions[idx], self.rewards[idx]

class Transformer_dataset(Dataset):
    """
    Dataset of a transformer strategy runs
    """
    def __init__(self, model, N, T):
        """
        model (Transformer) : The transformer to be executed.
        N : Dataset size.
        T : Horizon.
        """
        self.mab = model.mab
        self.model = model
        self.device = model.parameters().__next__().device
        self.N = N
        self.T = T
        self.n_arms = self.mab.n

        starting_actions = torch.zeros([N, 1], dtype = torch.long).to(self.device)
        starting_rewards = torch.zeros([N, 1], dtype = torch.float).to(self.device)
        actions, rewards = model.generate(starting_actions, starting_rewards, T, top_k = 2)
        self.actions = actions
        self.rewards = rewards

        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        return self.actions[idx], self.rewards[idx]


class Mixed_strategy_dataset(Dataset):
    """
    Dataset of a mixed strategies runs
    """
    def __init__(self, agents, N, T):
        """
        agent (Agent) :  List of the strategy to be executed
        N : Dataset size.
        T : Horizon.
        """
        self.k = len(agents)
        self.agents = agents
        self.N = N
        self.T = T
        self.n_arms = self.agents[0].mab.n
        self.actions = torch.zeros([N, T+1], dtype = torch.long) 
        self.rewards = torch.zeros([N, T+1], dtype = torch.float)
        for i in range(N):
            agent = np.random.choice(self.agents)
            agent.reinitialize()
            agent.take_N_actions(T)
            self.actions[i] = torch.cat((torch.Tensor([0]), torch.Tensor(agent.record["actions"])+1)) #0 is our $<bos>$
            self.rewards[i] = torch.cat((torch.Tensor([0]), torch.Tensor(agent.record["rewards"])))
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        return self.actions[idx], self.rewards[idx]

class Mixed_mab_dataset(Dataset):
    """
    Dataset of a strategy run on different problems
    NOT TESTED YET
    """
    def __init__(self, mabs, agent, N, T):
        """
        mabs (list) : List of MAB class instances used to generate the rewards.
        agent (Agent) : The strategy to be executed
        N : Dataset size.
        T : Horizon.
        """
        self.mabs = mabs
        self.agent = agent
        self.N = N
        self.T = T
        self.n_arms = self.mabs[0].n #TODO: check if all mabs have the same number of arms
        self.actions = torch.zeros([N, T+1], dtype = torch.long)
        self.rewards = torch.zeros([N, T+1], dtype = torch.float)
        self.chosen_mabs = torch.zeros([N], dtype = torch.long)
        for i in range(N):
            mab = np.random.choice(self.mabs)
            self.chosen_mabs[i] = self.mabs.index(mab)
            agent.reinitialize(mab)
            agent.take_N_actions(T) 
            self.actions[i] = torch.cat((torch.Tensor([0]), torch.Tensor(agent.record["actions"])+1))
            self.rewards[i] = torch.cat((torch.Tensor([0]), torch.Tensor(agent.record["rewards"])))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.actions[idx], self.rewards[idx]

if __name__ == "__main__":
    mab= MAB_normal(n=5)
    agent1 = UCBAgent(mab)
    agent2 = EpsilonGreedy(mab, 0.1)
    train_data = Mixed_strategy_dataset([agent1, agent2], 100, 10)
    print(train_data[:5])


