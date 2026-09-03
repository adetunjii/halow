import os
import csv
from halow.mappo.actor import Actor
from halow.mappo.critic import Critic
from halow.mappo.buffer import RolloutBuffer
import torch
import torch.nn
from torch.optim import Adam
from typing import Any
from halow.environment import CustomEnvironment
from gymnasium.spaces import Discrete, Box
from halow.constants import MAX_STEPS_PER_EPISODE
import numpy as np
import cProfile
import pstats

root = os.path.dirname(__file__)

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
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self._init_hyperparameters()
        self.drone_actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim,  self.num_layers).to(self.device)
        self.rover_actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.critic = Critic(self.global_state_dim, self.num_agents, self.hidden_dim, self.num_layers).to(self.device)
        self.drone_actor_optim = Adam(self.drone_actor.parameters(), lr=self.learning_rate_actor)
        self.rover_actor_optim = Adam(self.rover_actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        
    def _init_hyperparameters(self):
        self.gamma = 0.95
        self.max_timesteps_per_episode = MAX_STEPS_PER_EPISODE
        self.num_episodes_per_batch = 512
        self.num_layers = 3
        self.hidden_dim = 64
        self.num_agents = 2
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001
        self.td_lambda = 0.95
        self.epochs = 10
        self.batch_size = 64
        self.clip_rate = 0.2
        self.entropy_coefficient = 0.01
        self.max_grad_norm = 0.5
    
    def train(self, runs=10):
        log_path = self.setup_logger()
        
        for run in range(runs):
            buf = RolloutBuffer(
                num_agents=2,
                num_episodes=self.num_episodes_per_batch,
                td_lambda = self.td_lambda,
                gamma=self.gamma,
                observation_space=self.obs_dim,
                action_space=self.action_dim,
                global_state_space=self.global_state_dim,
                device= self.device
            )
            
            collected_episodes = []
            num_episode = 0
            while num_episode < self.num_episodes_per_batch:
                episode = {
                    "observations": [],
                    "actions": [],
                    "action_masks": [],
                    "log_probs": [],
                    "rewards": [],
                    "values": [],
                    "global_state": [],
                    "final_value": [0.0] * self.num_agents
                }
                seed = np.random.randint(0, 10000)
                obs, infos = self.env.reset(seed=seed)
                terminated, truncated = False, False
                # self.env.render()
                profiler = cProfile.Profile()
                profiler.enable()
                while not terminated and not truncated:
                    with torch.no_grad():
                        drone_obs = torch.tensor(obs["drone"], device=self.device)
                        rover_obs = torch.tensor(obs["rover"], device=self.device)
                        drone_action_mask = torch.tensor(infos["drone"]["action_mask"], device=self.device).bool()
                        rover_action_mask = torch.tensor(infos["rover"]["action_mask"], device=self.device).bool()
                        drone_action, drone_log_prob = self.drone_actor.act(drone_obs, action_mask=drone_action_mask)
                        rover_action, rover_log_prob = self.rover_actor.act(rover_obs, action_mask=rover_action_mask)
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
                    episode["log_probs"].append(np.stack((drone_log_prob.cpu().numpy(), rover_log_prob.cpu().numpy())))
                    episode["global_state"].append(global_state.cpu().numpy())
                    episode["values"].append(value.cpu().numpy())
                    episode["rewards"].append(list(rewards.values()))
                    obs = next_obs
                    terminated = any(terminated.values())
                    truncated = any(truncated.values())
                    # self.env.render()
                if truncated:
                    print("Yes truncated")
                    global_state = torch.tensor(self.env.global_state, device=self.device, dtype=torch.float32)
                    final_value = self.critic(global_state)
                    episode["final_value"] = final_value.detach().cpu().numpy()
                profiler.disable()
                stats = pstats.Stats(profiler)
                stats.sort_stats("cumulative")
                stats.print_stats(15)
                collected_episodes.append({
                    "observations": episode["observations"],
                    "rewards": episode["rewards"]
                })
            
                print("Yes terminated")
                buf.add(episode)
                num_episode += 1
            
            (batch_obs, batch_action_masks, batch_actions, batch_log_probs, batch_returns, batch_adv, batch_global_states) = buf.batchify()
            critic_losses = []
            drone_actor_losses = []
            rover_actor_losses = []
            drone_entropies = []
            rover_entropies = []
            
            for i in range(self.epochs):
                self.drone_actor_optim.zero_grad()
                self.rover_actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                
                # optimize critic network
                for start in range(0, batch_global_states.size(0), self.batch_size):
                    end = start + self.batch_size
                    current_values = self.critic(batch_global_states[start:end])
                    critic_loss = torch.nn.functional.mse_loss(current_values, batch_returns[start:end], reduction="mean")
                    critic_loss.backward()
                    critic_losses.append(critic_loss.item())
                
                # optimize the actor network            
                for start in range(0, batch_obs.size(0), self.batch_size):
                    end = start + self.batch_size
                    
                    drone_obs = batch_obs[start:end, 0, :]
                    drone_actions = batch_actions[start:end, 0]
                    drone_action_masks = batch_action_masks[start:end, 0]
                    drone_log_probs = batch_log_probs[start:end, 0]
                    drone_advantages = batch_adv[start:end, 0]
                    current_drone_log_prob, drone_entropy_loss = self.drone_actor.compute_entropy(drone_obs, drone_actions, drone_action_masks)
                    drone_log_ratio = current_drone_log_prob - drone_log_probs
                    drone_ratio = torch.exp(drone_log_ratio)
                    drone_grad_loss = torch.min((drone_advantages * drone_ratio), (drone_advantages * torch.clamp(drone_ratio, 1 - self.clip_rate, 1 + self.clip_rate)))
                    drone_grad_loss = drone_grad_loss.mean()
                    drone_actor_loss = -drone_grad_loss - self.entropy_coefficient * drone_entropy_loss.mean() # drone_entropy_loss.mean() to get the average entropy loss for all timesteps in the batch
                    drone_actor_loss.backward()
                    drone_actor_losses.append(drone_actor_loss.item())
                    drone_entropies.append(drone_entropy_loss.mean().item())
                    
                    rover_obs = batch_obs[start:end, 1, :]
                    rover_actions = batch_actions[start:end, 1]
                    rover_action_masks = batch_action_masks[start:end, 1]
                    rover_log_probs = batch_log_probs[start:end, 1]
                    rover_advantages = batch_adv[start:end, 1]
                    current_rover_log_prob, rover_entropy_loss = self.rover_actor.compute_entropy(rover_obs,rover_actions,rover_action_masks)
                    rover_log_ratio = current_rover_log_prob - rover_log_probs
                    rover_ratio = torch.exp(rover_log_ratio)             
                    rover_grad_loss = torch.min((rover_advantages * rover_ratio), (rover_advantages * torch.clamp(rover_ratio, 1 - self.clip_rate, 1 + self.clip_rate)))
                    rover_actor_loss = -rover_grad_loss.mean() - (self.entropy_coefficient * rover_entropy_loss.mean())
                    rover_actor_loss.backward()
                    rover_actor_losses.append(rover_actor_loss.item())
                    rover_entropies.append(rover_entropy_loss.mean().item())

                
                # gradient clipping
                drone_grad_norm = torch.nn.utils.clip_grad_norm_(self.drone_actor.parameters(), max_norm=self.max_grad_norm)
                rover_grad_norm = torch.nn.utils.clip_grad_norm_(self.rover_actor.parameters(), max_norm=self.max_grad_norm)
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
                
                self.drone_actor_optim.step()
                self.rover_actor_optim.step()
                self.critic_optim.step()
                
                self.log_run(
                    log_path, run,
                    collected_episodes,
                    critic_losses, drone_actor_losses, rover_actor_losses,
                    drone_entropies, rover_entropies,
                    drone_grad_norm.item(), rover_grad_norm.item(), critic_grad_norm.item()
                )
                self.save_policy(run)
                
        
    def save_policy(self, run: int):
        checkpoint = {
            "drone_actor": self.drone_actor.state_dict(),
            "rover_actor": self.rover_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "drone_optim": self.drone_actor_optim.state_dict(),
            "rover_optim": self.rover_actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict()
        }
        torch.save(checkpoint, os.path.join(root, f"weights/checkpoint_run_{run}.pt"))
        torch.save(checkpoint, os.path.join(root, "weights/policy.pt"))
         
    def setup_logger(self):
        log_path = os.path.join(root, "logs/training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run",
                "mean_return_drone",
                "mean_return_rover", 
                "std_return_drone",
                "std_return_rover",
                "mean_ep_length",
                "std_ep_length",
                "critic_loss",
                "drone_actor_loss",
                "rover_actor_loss",
                "drone_entropy",
                "rover_entropy",
                "drone_grad_norm",
                "rover_grad_norm",
                "critic_grad_norm"
            ])
        return log_path


    def log_run(self, log_path, run, episodes, critic_losses, drone_actor_losses, rover_actor_losses, drone_entropies, rover_entropies, drone_grad_norm, rover_grad_norm, critic_grad_norm):
        # episode-level stats — computed before batchify() clears the buffer
        episode_returns_drone = [sum(r[0] for r in ep["rewards"]) for ep in episodes]
        episode_returns_rover = [sum(r[1] for r in ep["rewards"]) for ep in episodes]
        episode_lengths = [len(ep["observations"]) for ep in episodes]

        row = [
            run,
            np.mean(episode_returns_drone),
            np.mean(episode_returns_rover),
            np.std(episode_returns_drone),
            np.std(episode_returns_rover),
            np.mean(episode_lengths),
            np.std(episode_lengths),
            np.mean(critic_losses),
            np.mean(drone_actor_losses),
            np.mean(rover_actor_losses),
            np.mean(drone_entropies),
            np.mean(rover_entropies),
            drone_grad_norm,
            rover_grad_norm,
            critic_grad_norm
        ]