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
from multiarmedbandits import MAB_normal_V0, MAB_normal_V1, MAB_normal, MAB_Bernoulli_V0, MAB_Bernoulli

from model import UCBTransformerModel
from dataset import UCB_dataset, Strategy_dataset, Mixed_strategy_dataset
from utils import plot_regret, compute_regret, compute_best_action_selection_rate
from strategies import EpsilonGreedy, UCB1, TransformerAgent, UniformAgent

from main import train
from matplotlib import pyplot as plt


def experiment1():
    """
    Untrained vs trained transformer
    """

    # Test code on Strategy_dataset
    mab = MAB_Bernoulli_random(5)
    
    # UCB strategy
    t= time.time()
    agent1 = UCB1(mab)
    stg_data = Strategy_dataset(agent1, mab, 500, 50)
    y = compute_regret(stg_data.rewards, mab)
    print("UCB1 time: ", time.time() - t)

    # Model config
    vocab_size = 5
    max_len = 106
    config = LightConfig(vocab_size+1, max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)
    model = UCBTransformerModel(config, mab).to(device)

    # Untrained transformer strategy
    t = time.time()
    agent2 = TransformerAgent(mab, model)
    stg_data2 = Strategy_dataset(agent2, mab, 500, 50)
    y2 = compute_regret(stg_data2.rewards, mab)
    print("Untrained transformer time: ", time.time() - t)

    ########
    # Model
    device = torch.device("cpu") # ("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = UCBTransformerModel(config, mab).to(device)
    # Data
    print("Building training data")
    t = time.time()
    train_data = UCB_dataset(mab, 5000, 100)
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size = batch_size)
    print('Data built in ', time.time() - t) 

    # Training
    train_config = TrainConfig(lr=0.001, epochs = 1, device = device)
    print("Training ...")
    t = time.time()
    train(trained_model, train_loader, train_config)
    print("Training time: ", time.time() - t)
    ########

    # Trained transformer strategy
    t = time.time()
    agent3 = TransformerAgent(mab, trained_model)
    stg_data3 = Strategy_dataset(agent3, mab, 500, 50)
    y3 = compute_regret(stg_data3.rewards, mab)
    print("Trained transformer time: ", time.time() - t)

    plt.plot(y[1:], label = 'UCB1')
    plt.plot(y2[1:], label = 'Untrained transformer')
    plt.plot(y3[1:], label = 'Trained Transformer')
    plt.legend()
    plt.title("Regret")
    plt.savefig("Figs/regret_comp.png")



def experiment2():
    """
    Trained transformer on data distribution vs out of data distribution
    Same MAB problems, but arms order is different
    """
    # Train a transformer model on a MAB, and test it on another MAB
    mab1 = MAB_Bernoulli_random(5)
    mab2 = MAB_Bernoulli_random(5)

    # Model config
    vocab_size = 5
    max_len = 56
    config = LightConfig(vocab_size+1, max_len)

    # Model
    device = torch.device("cpu") # ("cuda" if torch.cuda.is_available() else "cpu")
    model = UCBTransformerModel(config, mab1).to(device)
    # Data
    print("Building training data")
    t = time.time()
    train_data = UCB_dataset(mab1, 2000, 50)
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size = batch_size)
    print('Data built in ', time.time() - t)

    # Training
    train_config = TrainConfig(lr=0.001, epochs = 1, device = device)
    print("Training ...")
    t = time.time()
    train(model, train_loader, train_config)
    print("Training time: ", time.time() - t)

    # Trained transformer on data distribution strategy
    t = time.time()
    tr_agent = TransformerAgent(mab1, model)
    test_data = Strategy_dataset(tr_agent, mab1, 500, 50)
    y1 = compute_regret(test_data.rewards, mab1)
    print("Built sequences in : ", time.time() - t)
    
    # Trained transformer on out of data distribution strategy
    t = time.time()
    tr_agent = TransformerAgent(mab2, model)
    test_data = Strategy_dataset(tr_agent, mab2, 500, 50)
    y2 = compute_regret(test_data.rewards, mab2)
    print("Built sequences in : ", time.time() - t)

    plt.plot(y1[1:], label = 'Train performance')
    plt.plot(y2[1:], label = 'Test performance')
    plt.legend()
    plt.title("Regret")
    plt.savefig("Figs/regret_comp.png")


def experiment3():
    """
    Trained transformer on data distribution vs out of data distribution
    Different MAB problems
    """
    # Train a transformer model on a MAB, and test it on another MAB
    mab1 = MAB_Bernoulli_random(5)
    mab2 = MAB_normal_random(5)

    # Model config
    vocab_size = 5
    max_len = 56
    config = LightConfig(vocab_size+1, max_len)

    # Model
    device = torch.device("cpu") # ("cuda" if torch.cuda.is_available() else "cpu")
    model = UCBTransformerModel(config, mab1).to(device)
    # Data
    print("Building training data")
    t = time.time()
    train_data = UCB_dataset(mab1, 2000, 50)
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size = batch_size)
    print('Data built in ', time.time() - t)

    # Training
    train_config = TrainConfig(lr=0.001, epochs = 1, device = device)
    print("Training ...")
    t = time.time()
    train(model, train_loader, train_config)
    print("Training time: ", time.time() - t)

    # Trained transformer on data distribution strategy
    t = time.time()
    tr_agent = TransformerAgent(mab1, model)
    test_data = Strategy_dataset(tr_agent, mab1, 500, 50)
    y1 = compute_regret(test_data.rewards, mab1)
    print("Built sequences in : ", time.time() - t)
    
    # Trained transformer on out of data distribution strategy
    t = time.time()
    tr_agent = TransformerAgent(mab2, model)
    test_data = Strategy_dataset(tr_agent, mab2, 500, 50)
    y2 = compute_regret(test_data.rewards, mab2)
    print("Built sequences in : ", time.time() - t)

    plt.plot(y1[1:], label = 'Train performance')
    plt.plot(y2[1:], label = 'Test performance')
    plt.legend()
    plt.title("Regret")
    plt.savefig("Figs/regret_comp_different_mabs.png")


    
def experiment0():
    # plot ucb cumulative regret
    print("Comparing UCB and Epsilon Greedy cumulative regret ...")
    mab = MAB_Bernoulli(5)
    agent_UCB = UCB1(mab)
    agent_Eps = EpsilonGreedy(mab, 0.1)
    #stg_data_UCB = Strategy_dataset(agent_UCB, mab, 500, 200)
    stg_data_UCB = UCB_dataset(mab, 500, 200)
    stg_data_Eps = Strategy_dataset(agent_Eps, mab, 500, 200)
    regret_UCB = compute_regret(stg_data_UCB.rewards, mab)
    regret_Eps = compute_regret(stg_data_Eps.rewards, mab)
    plt.plot(regret_UCB[1:], label = "UCB1")
    plt.plot(regret_Eps[1:], label = "Epsilon Greedy")
    plt.legend()
    plt.title("UCB1-EpsGreedy cumulative regret")
    plt.savefig("Figs/ucb1_EG_cum_regret.png")
    plt.clf()
    best_action = mab.best_action
    best_action_selection_rate_UCB = compute_best_action_selection_rate(stg_data_UCB.actions, best_action)
    best_action_selection_rate_Eps = compute_best_action_selection_rate(stg_data_Eps.actions, best_action)
    plt.plot(best_action_selection_rate_UCB[1:], label = "UCB1")
    plt.plot(best_action_selection_rate_Eps[1:], label = "Epsilon Greedy")
    plt.legend()
    plt.title("UCB1-EpsGreedy best action selection rate")
    plt.savefig("Figs/ucb1_EG_best_action_selection_rate.png")









if __name__ == "__main__":
    experiment0()

