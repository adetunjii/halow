from halow.mappo.actor import Actor
from halow.mappo.critic import Critic
from halow.mappo.buffer import RolloutBuffer
from torch.optim import Adam
from typing import Any
from halow.environment import CustomEnvironment
from gymnasium.spaces import Discrete, Box

class Mappo:
    def __init__(self, env: CustomEnvironment):
        assert env is not None
        self.env = env        
        
        obs_space = self.env.observation_space("drone")
        action_space = self.env.action_space("drone")
        assert type(obs_space) == Box
        assert type(action_space) == Discrete
        assert self.env.global_state is not None
   
        self.obs_dim = obs_space.shape[0]        
        self.action_dim = int(action_space.n)
        self.global_state_dim = self.env.global_state.shape[0]
        
        self.num_agents = 2
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
        self.num_episodes_per_batch = 100
        self.num_layers = 3
        self.hidden_dim = 64
        self.num_layers = 3
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001
        self.td_lambda = 0.95
    
    def train(self):
        buf = RolloutBuffer(
            num_agents=2,
            num_episodes=self.num_episodes_per_batch,
            td_lambda = self.td_lambda,
            gamma=self.gamma,
            observation_space=self.obs_dim,
            action_space=self.action_dim,
            global_state_space=self.global_state_dim
        )

        step = 0
        while step < self.total_timesteps_per_batch:
            
            num_episode = 0
            while num_episode < self.num_episodes_per_batch:
                episode = {
                    "observations": [],
                    "actions": [],
                    "action_masks": [],
                    "rewards": [],
                    "advantages": [],
                    "values": [],
                    "global_state": []
                }
                
                num_episode += 1 
            step += 1