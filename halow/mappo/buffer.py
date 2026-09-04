import torch
import numpy as np
from typing import Any

class RolloutBuffer:
    def __init__(
        self,
        action_space,
        observation_space,
        global_state_space,
        gamma,
        td_lambda,
        device,
        num_episodes,
        num_agents=2,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.global_state_space = global_state_space
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.episodes: list[dict[str, Any] | None] = [None] * self.num_episodes
        self.episode_idx = 0
        self.device = device if device else "cpu"
        self.num_agents = num_agents
        self.td_lambda = td_lambda
        
    def add(self, episode) -> None:
        """Adds a completed episode to the buffer and computes its advantages"""
        final_value = episode.pop("final_value")
        eps = self.rollout_to_tensor(episode)
        eps["final_value"] = torch.tensor(final_value, dtype=torch.float32)
        self.compute_advantages(eps)
        self.episodes[self.episode_idx] = eps # type: ignore
        self.episode_idx += 1
        
    def rollout_to_tensor(self, episode: dict[str, Any]):
        eps = dict({})
        for key, values in episode.items():
            eps[key] = torch.from_numpy(np.stack(values))
        return eps
    
    def compute_advantages(self, episode):
        """Computes TD(lambda) returns and GAE advantages backwards from episode end"""
        returns = torch.zeros_like(episode["rewards"], dtype=torch.float32)
        advantages = torch.zeros_like(episode["rewards"], dtype=torch.float32)
        N = episode["observations"].size(0)
        values = episode["values"]
        final_value = episode["final_value"]
        
        lambda_return = final_value
        for t in reversed(range(N)):
            next_val = final_value if t == N-1 else values[t+1]
            lambda_return = episode["rewards"][t] + self.gamma * (self.td_lambda * lambda_return + (1 - self.td_lambda) * next_val)
            returns[t] = lambda_return
            advantages[t] = lambda_return - values[t]
        episode["returns"] = returns
        episode["advantages"] = advantages
        del episode["rewards"]
        del episode["values"]
            
    def batchify(self):
        """Stacks all collected episodes into flattened tensors and normalizes advantages"""
        print("batchifying.....")
        length_per_episode = []
        total_obs_collected = 0
        
        for episode in self.episodes:
            if episode is not None:
                length_per_episode.append(len(episode["observations"]))
                total_obs_collected += len(episode["observations"])
        observations = torch.zeros((total_obs_collected, self.num_agents, self.observation_space), dtype=torch.float32).to(self.device)
        action_masks = torch.zeros((total_obs_collected, self.num_agents, self.action_space)).bool().to(self.device)
        actions = torch.zeros((total_obs_collected, self.num_agents)).int().to(self.device)
        log_probs = torch.zeros(total_obs_collected, self.num_agents, dtype=torch.float32).to(self.device)
        returns = torch.zeros(total_obs_collected, self.num_agents, dtype=torch.float32).to(self.device)
        advantages = torch.zeros(total_obs_collected, self.num_agents, dtype=torch.float32).to(self.device)
        global_states = torch.zeros(total_obs_collected, self.global_state_space, dtype=torch.float32).to(self.device)

        idx = 0
        for episode, length in zip(self.episodes, length_per_episode):
            assert episode is not None
            observations[idx : idx+length] = episode["observations"]
            action_masks[idx : idx+length] = episode["action_masks"] # actions that can be taken
            actions[idx : idx + length] = episode["actions"] # actions output from the actor network
            returns[idx : idx + length] = episode["returns"]
            advantages[idx : idx + length] = episode["advantages"]
            global_states[idx: idx + length] = episode["global_state"]
            log_probs[idx: idx + length] = episode["log_probs"]
            idx += length
        advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-9)
        self.episodes = [None] * self.num_episodes
        self.episode_idx = 0
        
        return (
            observations,
            action_masks,
            actions,
            log_probs,
            returns,
            advantages,
            global_states
        )
