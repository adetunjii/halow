import torch
import numpy as np
from typing import Any

class RolloutBuffer:
    def __init__(
        self,
        action_space,
        observation_space, 
        gamma,
        td_lambda,
        total_num_episodes,
        num_agents=2,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.total_num_episodes = total_num_episodes
        self.episodes = [None] * self.total_num_episodes
        self.episode_idx = 0
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.num_agents = num_agents
        self.td_lambda = td_lambda
        
    def add(self, episode) -> None:
        self.compute_advantages(episode)
        self.episodes[self.episode_idx] = episode
        self.episode_idx += 1

    @staticmethod
    def rollout_to_tensor(episode: dict[str, Any]):
        for key, val in episode.items():
            episode[key] = torch.from_numpy(np.stack(episode[val]))
        return episode
    
    def compute_advantages(self, episode):
        rewards_to_go = advantages = torch.zeros_like(episode["rewards"], dtype=torch.float32)
        N = episode["obs"].size(0)
        
        future_reward = 0 if episode["terminated"][-1] else episode["values"][-1]
        for t in reversed(range(N)):
            next_val = 0 if t == N-1 else episode["values"][t+1]
            lambda_return = episode["rewards"][t] + self.gamma * (self.td_lambda * future_reward + (1 - self.td_lambda) * next_val)
            rewards_to_go[t] = future_reward = lambda_return

            advantages[t] = lambda_return - episode["values"][t]
        episode["rewards_to_go"] = rewards_to_go
        episode["advantages"] = advantages
        
        del episode["rewards"]
        del episode["values"]
            
    def batchify(self):
        length_per_episode = []
        total_obs_collected = 0
        for episode in self.episodes:
            if episode is not None:
                length_per_episode.append(len(episode))
                total_obs_collected += len(episode)
        observations = torch.zeros((total_obs_collected, self.num_agents, self.observation_space), dtype=torch.float32).to(self.device)
        action_masks = torch.zeros((total_obs_collected, self.num_agents, self.action_space)).bool().to(self.device)
        actions = torch.zeros((total_obs_collected, self.num_agents)).int().to(self.device)
        rewards_to_go = torch.zeros(total_obs_collected, dtype=torch.float32).to(self.device)
        advantages = torch.zeros(total_obs_collected, dtype=torch.float32).to(self.device)
        
        idx = 0
        for episode, length in zip(self.episodes, length_per_episode):
            assert episode is not None
            observations[idx : idx+length] = torch.flatten(episode["obs"], 0, 1)
            action_masks[idx : idx+length] = torch.flatten(episode["action_masks"], 0, 1) # actions that can be taken
            actions[idx : idx + length] = episode["actions"] # actions the network output
            rewards_to_go[idx : idx + length] = episode["rewards"]
            rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / rewards_to_go.std() + 1e-9
            advantages[idx : idx + length] = episode["advantages"]
            advantages = (advantages - advantages.mean()) / advantages.std() + 1e-9
            
            idx += length
        
        self.episodes = [None] * self.total_num_episodes
        
        return (
            observations.flatten(0, 1),
            action_masks.flatten(0, 1),
            actions.flatten(0, 1),
            rewards_to_go,
            advantages,
        )