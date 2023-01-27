import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time

from layers import Block
from config import LightConfig, TransformerConfig
from multiarmedbandits import MAB_normal

#TODO Debug
class UCBTransformerModel(nn.Module):
    """
    Decoder only transformer (GPT) to learn a multi-armed bandits strategy.
    """
    def __init__(self, config, mab):
        super().__init__()
        self.mab = mab # Multi-armed bandits instance, used in inference time for generation
        n_embd = config.n_embd - 1 # The rewards vector is added as an embedding to be fed to the transformer block
        self.max_len = config.max_len
        self.tok_embed = nn.Embedding(
            config.vocab_size, n_embd
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, n_embd)
        )
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_blocks)]
        )
        n_embd += 1
        self.ln = nn.LayerNorm(n_embd)
        self.fc = nn.Linear(n_embd, config.vocab_size) #TODO Test if a two layer MLP is better
        
    
    def reinit(self, mab):
        """
        Reinitalize the model with a new MAB instance, while keeping the same model parameters.
        """
        assert mab.n == self.mab.n, "The new MAB instance must have the same number of arms as the previous one."
        self.mab = mab
    def forward(self, actions, rewards, target=None):
        # batch_size = x.size(0)
        x = actions
        seq_len = x.size(1)
        assert seq_len <= self.max_len, "sequence longer than model capacity"
        tok_embedding = self.tok_embed(x)
        # tok_embedding.shape == (batch_size, seq_len, embed_dim)
        pos_embedding = self.pos_embed[:, :seq_len, :]
        # pos_embedding.shape == (1, seq_len, embed_dim)
        x = self.dropout(tok_embedding + pos_embedding)
        x = torch.cat((x, rewards.unsqueeze(-1)), dim=-1)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.fc(x)
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = actions[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # x.shape == (batch_size, seq_len, vocab_size)
        return loss, logits
    
    #TODO never return 0 as next action 
    @torch.no_grad()
    def generate(self, actions, rewards, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of actions, rewards (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_len
            actions_cond = actions if actions.size(1) <= self.max_len else actions[:, -self.max_len:]
            rewards_cond = rewards if rewards.size(1) <= self.max_len else rewards[:, -self.max_len:]
            
            # forward the model to get the logits for the index in the sequence
            _, logits = self(actions_cond, rewards_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Excluding the token 0 <BOS>
            logits.index_fill_(1, torch.tensor([0]).to(logits.device),-float('Inf') )
            #logits[:, 0, : ] = -float('Inf')


            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                action_next = torch.multinomial(probs, num_samples=1)
            else:
                _, action_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            
            reward_next = torch.Tensor(self.mab.pull_V(action_next.cpu()-1)).to(rewards.device)
         
            
            actions = torch.cat((actions, action_next), dim=1)
            rewards = torch.cat((rewards, reward_next), dim=1)

        return actions, rewards


if __name__ == "__main__":
    vocab_size = 5
    mab1 = MAB_normal(5)
    max_len = 25
    config = LightConfig(vocab_size, max_len)
    model = UCBTransformerModel(config, mab1)
    print(model)