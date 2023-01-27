import numpy as np
import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import time

from layers import Block
from config import LightConfig, TransformerConfig, TrainConfig
from multiarmedbandits import MAB_normal_V0, MAB_normal_V1, MAB_normal, MAB_Bernoulli_V0, MAB_Bernoulli, MAB_Deterministic

from model import UCBTransformerModel
from dataset import UCB_dataset, Strategy_dataset, Mixed_strategy_dataset, Transformer_dataset, Mixed_mab_dataset
from utils import plot_regret, compute_regret, compute_best_action_selection_rate, plot_regrets, plot_best_action_selection_rate
from strategies import EpsilonGreedy, UCBAgent, TransformerAgent, UniformAgent

from main import train
from matplotlib import pyplot as plt



def experiment_mixed_training(N_train = 2000, N_eval = 1000, T = 100, n_arms = 5, n_mabs = 20):
    """
    Comparing Transformer, UCB, and Epsilon Greedy cumulative regret and best action selection rate
    """
    print("Comparing UCB and Epsilon Greedy cumulative regret ...")
    t = time.time()
    mabs = []
    for i in range(n_mabs):
        mabs.append(MAB_normal(n_arms))
        print("Best action : ", mabs[i].best_action)
    agent_ucb = UCBAgent(mabs[0])
    train_data = Mixed_mab_dataset(mabs, agent_ucb, N_train, T)
    train_loader = DataLoader(train_data, batch_size = 32)
    print("Built train_data in : ", time.time() - t)
    # Transformer agent
    t = time.time()
    vocab_size = n_arms
    max_len = T+1
    config = LightConfig(vocab_size+1, max_len)
    device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UCBTransformerModel(config, mabs[0]).to(device)
    train_config = TrainConfig(lr=0.001, epochs = 20, device = device)
    train(model, train_loader, train_config)
    print("Training time: ", time.time() - t)
    print("Changing mab")
    mab = MAB_normal(n_arms)
    print("New multi armed bandit permutation : ", mab.perm)
    print("New best action : ", mab.best_action)
    model.reinit(mab)
    t = time.time()
    data_T = Transformer_dataset(model, N_eval, T)
    print("Transformer generated data" , data_T[:3][:30])
    print("Built data_T in : ", time.time() - t)
    t = time.time()
    agent_Eps = EpsilonGreedy(mab, 0.1)
    agent_Uni = UniformAgent(mab)
    data_Uni = Strategy_dataset(agent_Uni, mab, N_eval, T)
    data_Eps = Strategy_dataset(agent_Eps, mab, N_eval, T)
    data_UCB = UCB_dataset(mab, N_eval, T)
    print("Built data_UCB, data_EPS and data_Uni in : ", time.time() - t)
    datasets_list = [data_Uni, data_T, data_Eps, data_UCB]
    names = ["Uniform", "TransformerM", "Epsilon Greedy", "UCB"]
    plot_regrets(datasets_list, mab, names, suffix = '_n')
    plot_best_action_selection_rate(datasets_list, mab, names, suffix = '_n')

    

def experiment_mixed_training_new_eval(N_train = 2000, N_eval = 2000, T = 100, n_arm = 5, n_mabs = 20):
    """
    Comparing Transformer, UCB, and Epsilon Greedy cumulative regret and best action selection rate
    """
    print("Training the model on normal, and testing on bernoulli bandits")
    t = time.time()
    mabs = []
    for i in range(n_mabs):
        mabs.append(MAB_Bernoulli(n_arm))
    agent_ucb = UCBAgent(mabs[0])
    train_data = Mixed_mab_dataset(mabs, agent_ucb, N_train, T)
    train_loader = DataLoader(train_data, batch_size = 32)
    print("Built train_data in : ", time.time() - t)
    # Transformer agent
    t = time.time()
    vocab_size = mabs[0].n
    max_len = T+1
    config = LightConfig(vocab_size+1, max_len)
    device = torch.device("cpu")# torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UCBTransformerModel(config, mabs[0]).to(device)
    train_config = TrainConfig(lr=0.001, epochs = 20, device = device)
    train(model, train_loader, train_config)
    print("Training time: ", time.time() - t)
    print("Using bernoulli MAB for evaluation")
    mab = MAB_normal(5)
    print("New Multi armed bandit permutation : ", mab.perm)
    print("New best action : ", mab.best_action)
    model.reinit(mab)
    t= time.time()
    data_T = Transformer_dataset(model, N_eval, T)
    print("Built data_T in : ", time.time() - t)
    print("Transformer generated data" , data_T[:3][:30])
    t = time.time()
    agent_Eps = EpsilonGreedy(mab, 0.1)
    agent_Uni = UniformAgent(mab)
    data_Uni = Strategy_dataset(agent_Uni, mab, N_eval, T)
    data_Eps = Strategy_dataset(agent_Eps, mab, N_eval, T)
    data_UCB = UCB_dataset(mab, N_eval, T)
    print("Built data_UCB, data_EPS and data_Uni in : ", time.time() - t)
    datasets_list = [data_Uni, data_T, data_Eps, data_UCB]
    names = ["Uniform", "TransformerM", "Epsilon Greedy", "UCB"]
    plot_regrets(datasets_list, mab, names, suffix = '_b_to_n')
    plot_best_action_selection_rate(datasets_list, mab, names, suffix = '_b_to_n')

if __name__ == "__main__":
    experiment_mixed_training()

