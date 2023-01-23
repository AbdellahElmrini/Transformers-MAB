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
from multiarmedbandits import MAB_normal, MAB_normal2, MAB_Bernoulli, MAB_normal_random, MAB_Bernoulli_random
from model import UCBTransformerModel
from dataset import UCB_dataset, Strategy_dataset
from utils import plot_regret, compute_regret
from strategies import Epsilon_greedy, UCB1, TransformerAgent, Uniform_agent

from main import train
from matplotlib import pyplot as plt


def exeperiment_1():
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

if __name__ == "__main__":
    exeperiment_1()

