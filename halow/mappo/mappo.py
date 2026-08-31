from mappo.actor import Actor
from mappo.critic import Critic
from torch.optim import Adam
from typing import Any

class Mappo:
    def __init__(self, env: Any):
        assert env is not None
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self._init_hyperparameters()
        
        self.drone_actor = Actor(self.obs_dim, self.hidden_dim, self.action_dim, self.num_layers)
        self.rover_actor = Actor(self.obs_dim, self.hidden_dim, self.action_dim, self.num_layers)
        self.critic = Critic(self.obs_dim, self.hidden_dim, self.num_layers)
        
        self.drone_actor_optim = Adam(self.drone_actor.parameters(), lr=self.learning_rate_actor)
        self.rover_actor_optim = Adam(self.rover_actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        
    def _init_hyperparameters(self):
        self.gamma = 0.95
        self.learning_rate = 0.01
        self.max_timesteps_per_episode = 200
        self.total_timesteps_per_batch = 2048
        self.num_layers = 3
        self.hidden_dim = 64
        self.num_layers = 3
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001
    
    def batch_experiences(self):
        pass