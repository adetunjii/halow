import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        )
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x