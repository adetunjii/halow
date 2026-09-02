from halow.mappo.actor import Actor
from halow.mappo.critic import Critic
from halow.mappo.buffer import RolloutBuffer
import torch
from torch.nn import functional
from torch.optim import Adam
from typing import Any
from halow.environment import CustomEnvironment
from gymnasium.spaces import Discrete, Box
import numpy as np

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
    
        self._init_hyperparameters()
        
        self.drone_actor = Actor(self.obs_dim, self.hidden_dim, self.action_dim, self.num_layers)
        self.rover_actor = Actor(self.obs_dim, self.hidden_dim, self.action_dim, self.num_layers)
        self.critic = Critic(self.obs_dim, self.hidden_dim, self.num_layers)
        
        self.drone_actor_optim = Adam(self.drone_actor.parameters(), lr=self.learning_rate_actor)
        self.rover_actor_optim = Adam(self.rover_actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
    def _init_hyperparameters(self):
        self.gamma = 0.95
        self.learning_rate = 0.01
        self.max_timesteps_per_episode = 200
        self.total_timesteps_per_batch = 2048
        self.num_episodes_per_batch = 100
        self.num_layers = 3
        self.hidden_dim = 64
        self.num_agents = 2
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001
        self.td_lambda = 0.95
        self.epochs = 10
        self.batch_size = 32
    
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
                    "log_probs": [],
                    "rewards": [],
                    "advantages": [],
                    "values": [],
                    "global_state": [],
                    "final_value": torch.tensor(0.)
                }
                
                obs, infos = self.env.reset()
                terminated, truncated = False, False
                while not terminated and not truncated:
                    with torch.no_grad():
                        drone_action_mask = torch.tensor(infos["drone"]["action_mask"], device=self.device).bool()
                        rover_action_mask = torch.tensor(infos["rover"]["action_mask"], device=self.device).bool()
                        x = torch.tensor(np.stack(list(obs.values())), device=self.device)
                        drone_action, drone_log_probs = self.drone_actor.act(x, action_mask=drone_action_mask)
                        rover_action, rover_log_probs = self.rover_actor.act(x, action_mask=rover_action_mask)
                        global_state = torch.tensor(self.env.global_state, device=self.device, dtype=torch.float32)
                        value = self.critic(global_state)
                        actions = {
                            "drone": drone_action.cpu().numpy(),
                            "rover": rover_action.cpu().numpy()
                        }
                    next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                    episode["observations"].append(np.stack(list(obs.values())))
                    episode["action_masks"].append(np.stack((drone_action_mask.cpu().numpy(), rover_action_mask.cpu().numpy())))
                    episode["actions"].append(np.stack((drone_action.cpu().numpy(), rover_action.cpu().numpy())))
                    episode["log_probs"].append(np.stack((drone_log_probs.cpu().numpy(), rover_log_probs.cpu().numpy())))
                    episode["global_state"].append(global_state.cpu().numpy())
                    episode["values"].append(value)
                    episode["rewards"].append(list(rewards.values()))
                    obs = next_obs
                    terminated = any(terminated.values())
                    truncated = any(truncated.values())
                    step += 1
                if truncated:
                    global_state = torch.tensor(self.env.global_state, device=self.device, dtype=torch.float32)
                    final_value = self.critic(global_state)
                    episode["final_value"] = final_value
                buf.add(episode)
                num_episode += 1

        (batch_obs, batch_action_masks, batch_actions, batch_returns, batch_adv, batch_global_states) = buf.batchify()
        
        for i in range(self.epochs):
            self.drone_actor_optim.zero_grad()
            self.rover_actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            
            # optimize critic network
            num_samples_critic = batch_global_states.size(0)
            for start in range(0, batch_global_states.size(0), self.batch_size):
                end = start + self.batch_size
                current_values = self.critic(x=batch_global_states[start:end])
                critic_loss = functional.mse_loss(current_values, batch_returns[start:end], reduction="sum")
                norm_critic_loss = critic_loss / num_samples_critic
                # cr_loss += norm_critic_loss.detach()
                critic_loss.backward()
            
            
            self.drone_actor_optim.step()
            self.rover_actor_optim.step()
            self.critic_optim.step()