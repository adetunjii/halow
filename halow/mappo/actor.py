import torch
from torch import nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        self.input_dim = input_dim
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def act(self, x, action_mask=None):
        logits = self.logits(x, action_mask)
        dist = Categorical(logits=logits)
        dist._validate_args = False
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def logits(self, x, action_mask=None):
        for layer in self.layers:
            x = layer(x)
        if action_mask is not None:
            x = x.masked_fill(~action_mask, -1e9)
        return x
    
    def compute_entropy(self, x, action, action_mask=None):
        logits = self.logits(x, action_mask)
        dist = Categorical(logits=logits)
        dist._validate_args = False
        log_prob = dist.log_prob(action)
        return log_prob, dist.entropy()
 