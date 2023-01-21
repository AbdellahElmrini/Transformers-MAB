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
from multiarmedbandits import MAB_normal, MAB_normal2, MAB_Bernoulli, MAB_normal_random_order
from model import UCBTransformerModel
from dataset import UCB_dataset

def train(model, train_loader, train_config : TrainConfig):
    model.train()
    device = train_config.device
    optimizer = Adam(model.parameters(), lr = train_config.lr)
    start = time.time()
    epochs = train_config.epochs
    train_loss = 0
    count = 0
    for epoch in range(epochs):
        train_loss = 0
        count = 0
        for actions, rewards in train_loader:
            optimizer.zero_grad()
            actions = actions.to(device)
            rewards = rewards.to(device)
            loss, logits = model(actions, rewards)
            loss.backward()
            train_loss += loss*actions.shape[0]
            optimizer.step()
            count += actions.shape[0]
        if epoch%2 == 1:
            tpe = (time.time()-start)/2
            print("Epoch : {} - Loss : {} - Train_Loss : {} - Time per epoch {:.2f}s".format(
                epoch, loss, train_loss/count, tpe))
            start = time.time()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Building the datasets
    vocab_size = 5
    mab1 = MAB_Bernoulli(5)
    train_data = UCB_dataset(mab1, 10000, 25)
    #test_data = UCB_dataset(mab1, 1000, 10)

    batch_size = 32
    train_loader = DataLoader(train_data, batch_size = batch_size)
    #test_loader = DataLoader(test_data, batch_size = batch_size)

    
    max_len = 30
    config = LightConfig(vocab_size+1, max_len)
    model = UCBTransformerModel(config, mab1).to(device)

    train_config = TrainConfig(lr=0.001, epochs = 1, device = device)
    print("Training ...")
    train(model, train_loader, train_config)


    print("Comparing the model with a UCB strategy in inference time")

    ucb_data = UCB_dataset(mab1, 1000, 25)
    gen_actions, gen_rewards = model.generate(ucb_data.actions[:,[0]].to(device),
                                                ucb_data.rewards[:,[0]].to(device), 30 )
    print("Mean total regret of a UCB strategy : ", mab1.best_reward_avg - ucb_data.rewards.mean())
    print("Mean total regret of transformer strategy : ", mab1.best_reward_avg - gen_rewards.mean())




