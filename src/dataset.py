import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from multiarmedbandits import MAB_normal
from strategies import Epsilon_greedy, UCB1


class UCB_dataset(Dataset):
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
        self.actions = torch.zeros([N, T+self.n_arms], dtype = torch.long)
        self.rewards = torch.zeros([N, T+self.n_arms], dtype = torch.float)
        for i in range(N):
            ucb_agent = UCB1(mab)
            ucb_agent.initialize()
            ucb_agent.run_N_actions(T) 
            self.actions[i] = torch.Tensor(ucb_agent.record["actions"])
            self.rewards[i] = torch.Tensor(ucb_agent.record["rewards"])
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        return self.actions[idx], self.rewards[idx]


if __name__ == "__main__":
    mab1= MAB_normal(n=5)
    train_data = UCB_dataset(mab1, 100, 10)
    print(train_data[:5])