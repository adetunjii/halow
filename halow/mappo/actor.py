from torch import nn
from torch.nn import functional

class Actor(nn.Module):
    def __init__(self, input_dim: int, otuput_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Linear(hidden_dim, otuput_dim))
    
    def act(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
        