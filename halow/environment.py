import os
from typing import Any
from gymnasium.spaces import Discrete, Box
from gymnasium.spaces.space import Space
from pettingzoo import ParallelEnv
import numpy as np
from halow.agent import Agent, ConfidenceLevels, UpdateLimit
from copy import copy
from halow.helpers import get_observable_cells, normalize_pos, interpolate_path, astar_search, get_animation_frames, generate_map
from halow.frontiers import top_k_frontiers, pack_frontiers
from halow.constants import *
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import functools

root = os.path.dirname(__file__)

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "CustomGridv0",
    }
    
    def __init__(self):
        self.height = HEIGHT
        self.width = WIDTH
        self.ground_truth = None
        self.cell_confidence = None
        self.shared_belief_map = None
        self.drone = None
        self.rover = None
        self.current_drone_pos = None
        self.current_rover_pos = None
        self.drone_frontier_target = None
        self.rover_frontier_target = None
        self.drone_path, self.rover_path = None, None
        self.fig, self.ax_mapped, self.ax_ground_truth = None, None, None
        self.global_state = None
        self.possible_agents = ["drone", "rover"]
    
    def reset(self, seed=None, options=None):
        self.ground_truth = generate_map(self.height, self.width, seed=seed,)
        self.agents = copy(self.possible_agents)
        self.shared_belief_map = np.full((self.height, self.width), 0.5, dtype=np.float32)
        self.cell_confidence = np.zeros((self.height, self.width), dtype=np.float32)
        
        self.drone = Agent(
            name='drone',
            confidence=ConfidenceLevels(occupied=0.7, free=0.7),
            belief_update_limit=UpdateLimit(min=-1, max=1.2),
            radius=3,
            belief_state=np.full((self.height, self.width), 0.5, dtype=np.float32),
            start_pos=(0, 0),
            battery_level=1
        )
        
        self.current_drone_pos = self.drone.start_pos
        self.drone.update_internal_belief_state(self.current_drone_pos, self.ground_truth)
        self._update_shared_belief(self.drone, self.current_drone_pos)

        self.rover = Agent(
            name='rover',
            confidence=ConfidenceLevels(occupied=0.95, free=0.99),
            belief_update_limit=UpdateLimit(min=-2.5, max=2.5),
            radius=1,
            belief_state=np.full((self.height, self.width), 0.5, dtype=np.float32),
            start_pos=(1, 0),
            battery_level=1
        )
        
        self.current_rover_pos = self.rover.start_pos
        self.rover.update_internal_belief_state(self.current_rover_pos, self.ground_truth)
        self._update_shared_belief(self.rover, self.current_rover_pos)
        
        drone_k_frontiers = top_k_frontiers(
            self.shared_belief_map,
            self.drone.detect_frontiers(),
            self.current_drone_pos,
            self.drone.radius,
        )
    
        reachable_rover_frontiers = self._get_reachable_frontiers(self.rover.detect_frontiers(), self.current_rover_pos)
        reachable_frontiers = [f for f, _ in reachable_rover_frontiers ]
        rover_k_frontiers = top_k_frontiers(
            self.shared_belief_map,
            reachable_frontiers,
            self.current_rover_pos,
            self.rover.radius
        )

        drone_action_mask = self._get_action_mask(drone_k_frontiers)
        rover_action_mask = self._get_action_mask(rover_k_frontiers)
        
        self.drone_frontier_target = None
        self.rover_frontier_target = None
        
        self.drone_path = deque([])
        self.rover_path = deque([])
        
        self._step_count = 0
        infos = {a: {} for a in self.agents}
        infos["drone"]["action_mask"] = drone_action_mask
        infos["rover"]["action_mask"] = rover_action_mask
        
        observations = self._get_observations()
        self.global_state = self._global_state(observations)
        
        return observations, infos
    
    def step(self, actions):
        assert self.drone is not None
        assert self.rover is not None
        assert self.current_drone_pos is not None
        assert self.current_rover_pos is not None
        assert self.shared_belief_map is not None
        assert self.drone_path is not None
        assert self.rover_path is not None
            
        drone_action = actions['drone']
        rover_action = actions['rover']
        
        rewards = {"drone": 0.0, "rover": 0.0}
        terminated = {a: False for a in self.agents}
        truncated = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        drone_k_frontiers = top_k_frontiers(
            self.shared_belief_map,
            self.drone.detect_frontiers(),
            self.current_drone_pos,
            self.drone.radius,
        )
    
        reachable_rover_frontiers = self._get_reachable_frontiers(self.rover.detect_frontiers(), self.current_rover_pos)
        reachable_frontiers = [f for f, _ in reachable_rover_frontiers ]
        rover_k_frontiers = top_k_frontiers(
            self.shared_belief_map,
            reachable_frontiers,
            self.current_rover_pos,
            self.rover.radius
        )
        self.rover_cached_path = {f: path for f, path in reachable_rover_frontiers}
        
        # define action_mask to determine valid actions for this step 
        drone_action_mask = self._get_action_mask(drone_k_frontiers)
        rover_action_mask = self._get_action_mask(rover_k_frontiers) 
        
        infos["drone"]["action_mask"] = drone_action_mask
        infos["rover"]["action_mask"] = rover_action_mask
        
        drone_current_target  = self._map_action_to_frontier(drone_action, drone_k_frontiers)
        drone_penalty = self._compute_action_penalty(action=drone_action, scored_frontiers=drone_k_frontiers)
        rover_current_target = self._map_action_to_frontier(rover_action, rover_k_frontiers)
        rover_penalty = self._compute_action_penalty(action=rover_action, scored_frontiers=rover_k_frontiers)
        
        drone_target_changed = drone_current_target is not None and (drone_current_target != self.drone_frontier_target or len(self.drone_path) == 0)
        rover_target_changed = rover_current_target is not None and (rover_current_target != self.rover_frontier_target or len(self.rover_path) == 0)
        
        # recompute a path if a frontier target changes mid-way
        if drone_target_changed:
            self.drone_path.clear()
            self.drone_frontier_target = drone_current_target
            new_path = interpolate_path(self.current_drone_pos, drone_current_target)
            self.drone_path.extend(new_path)
            
        if rover_target_changed:
            self.rover_path.clear()
            self.rover_frontier_target = rover_current_target
            
            path = self.rover_cached_path.get(self.rover_frontier_target)
            if path:
                self.rover_path.extend(path)
            
            # no path found this step
            if len(self.rover_path) == 0:
                print("No valid path for rover")
                self.rover_frontier_target = None
            
        # count number of resolved cells for information gain
        num_resolved_before = self._count_resolved_cells() 
        
        # drone acts as a scout, therefore takes more steps than the rover
        for _ in range(DRONE_STEPS):
            if not self.drone_path:
                break
            
            next_drone_pos = self.drone_path.popleft()
            next_drone_row, next_drone_col = int(round(next_drone_pos[0])), int(round(next_drone_pos[1]))
            self.current_drone_pos = (next_drone_row, next_drone_col)
            
            self.drone.update_internal_belief_state(self.current_drone_pos, self.ground_truth)
            self._update_shared_belief(self.drone, self.current_drone_pos)
            self.drone.battery_level = max(0.0, self.drone.battery_level - BATTERY_DEPLETION_RATE_PER_STEP)
            
        if self.rover_path:
            next_rover_pos = self.rover_path.popleft()
            next_rover_row, next_rover_col = int(round(next_rover_pos[0])), int(round(next_rover_pos[1]))  
            
            self.current_rover_pos = (next_rover_row, next_rover_col)
            self.rover.update_internal_belief_state(self.current_rover_pos, self.ground_truth)
            self._update_shared_belief(self.rover, self.current_rover_pos)
            self.rover.battery_level = max(0.0, self.rover.battery_level - BATTERY_DEPLETION_RATE_PER_STEP)
        
        # compute rewards per step
        num_resolved_after = self._count_resolved_cells()
        new_info_gained = num_resolved_after - num_resolved_before
        norm_info_gained = new_info_gained / (self.height * self.width)
        
        joint_reward = norm_info_gained
        rewards["drone"] = joint_reward - (ALPHA * drone_penalty)
        rewards["rover"] = joint_reward - (ALPHA * rover_penalty)
                    
        coverage = num_resolved_after / (self.height * self.width)
        if coverage >= COVERAGE_TARGET:
            print(f"Terminating: coverage {coverage:.2f}%")
            terminated = {a: True for a in self.agents}
        
        self._step_count += 1
        if self._step_count >= MAX_STEPS_PER_EPISODE:
            truncated = {a: True for a in self.agents}
            
        if self.drone.battery_level <= 0:
            terminated["drone"] = True
        if self.rover.battery_level <= 0:
            terminated["rover"] = True
        
        active_agents = []        
        for agent in self.agents:
            if not terminated[agent] and not truncated[agent]:
                active_agents.append(agent)

        self.agents = active_agents
        observations = self._get_observations()
        self.global_state = self._global_state(observations) 
        return observations, rewards, terminated, truncated, infos    
    
    def render(self):
        assert self.drone is not None and self.drone.start_pos is not None
        assert self.rover is not None and self.rover.start_pos is not None
        
        if self.fig is None:
            self.fig, (self.ax_ground_truth, self.ax_mapped) = plt.subplots(nrows=1, ncols=2)
            plt.ion()
            
            self.ax_ground_truth.set_title("Ground Truth")
            self.ax_ground_truth.imshow(self.ground_truth, cmap="terrain", vmin=0, vmax=2, interpolation='bicubic')
            self.mapped_img = self.ax_mapped.imshow(self.shared_belief_map, cmap="Greys", vmin=0, vmax=1)
            
            for ax in (self.ax_ground_truth, self.ax_mapped):
                ax.set_ylim(-0.5, self.height - 0.5)
                ax.set_xlim(-0.5, self.width - 0.5)
                ax.invert_yaxis()
                ax.xaxis.tick_top()
            
            self.drone_animation_frames = get_animation_frames(os.path.join(root, "./assets/drone.gif"))
            self.rover_animation_frames = get_animation_frames(os.path.join(root, "./assets/rover.gif"))

            assert self.drone_animation_frames is not None, f"Drone animation frames failed to load"
            assert self.rover_animation_frames is not None, f"Rover animation frames failed to load"
            
            self.frame_idx = 0
            drone_start_row, drone_start_col = self.drone.start_pos
            self.drone_img_box = OffsetImage(self.drone_animation_frames[0], zoom=0.35)
            self.drone_annotation = AnnotationBbox(self.drone_img_box, (drone_start_col, drone_start_row), frameon=False)
            
            rover_start_row, rover_start_col = self.rover.start_pos
            self.rover_img_box = OffsetImage(self.rover_animation_frames[0], zoom=0.15)
            self.rover_annotation = AnnotationBbox(self.rover_img_box, (rover_start_col, rover_start_row), frameon=False)
            
            self.ax_mapped.add_artist(self.drone_annotation)
            self.ax_mapped.add_artist(self.rover_annotation)
            
            plt.tight_layout()
        
        self.mapped_img.set_data(self.shared_belief_map)
        assert self.current_drone_pos is not None, f"Cannot plot drone, current_drone_pos is None"
        assert self.current_rover_pos is not None, f"Cannot plot rover, current_rover_pos is None"
        assert self.drone_animation_frames is not None and self.rover_animation_frames is not None

        drone_row, drone_col = self.current_drone_pos
        self.drone_annotation.xy = (drone_col, drone_row)
        self.drone_annotation.xybox = (drone_col, drone_row)
        self.drone_img_box.set_data(self.drone_animation_frames[self.frame_idx % len(self.drone_animation_frames)])
        
        rover_row, rover_col = self.current_rover_pos
        self.rover_annotation.xy = (rover_col, rover_row)
        self.rover_annotation.xybox = (rover_col, rover_row)
        self.rover_img_box.set_data(self.rover_animation_frames[self.frame_idx % len(self.rover_animation_frames)])
        self.frame_idx += 1
        
        coverage = self._count_resolved_cells() / (self.height * self.width)
        # TODO: plot coverage, step, battery level
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    @functools.cache
    def observation_space(self, agent: Any) -> Space: # type: ignore
        obs_size = (self.height * self.width) + 4 + (NUM_FRONTIERS * 3) + 1
        return Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)
    
    @functools.cache
    def action_space(self, agent: Any): # type: ignore
        return Discrete(NUM_FRONTIERS + 1)
    
    def _global_state(self, observation: dict):
        assert self.cell_confidence is not None
        assert self.current_drone_pos is not None
        assert self.current_rover_pos is not None
        assert self.drone is not None and self.rover is not None
        assert self.ground_truth is not None
        
        return np.concatenate([np.stack(list(observation.values())).flatten(), 
                               self.ground_truth.flatten(), 
                               normalize_pos(self.current_drone_pos), 
                               normalize_pos(self.current_rover_pos), 
                               [self.drone.battery_level, self.rover.battery_level], 
                               self.cell_confidence.flatten()]
                            ).astype(np.float32)

    def _get_observations(self):
        assert self.drone is not None
        assert self.rover is not None
        assert self.current_drone_pos is not None
        assert self.current_rover_pos is not None
        assert self.shared_belief_map is not None
        
        drone_frontiers = self.drone.detect_frontiers()
        packed_drone_frontiers = pack_frontiers(
            top_k_frontiers(self.shared_belief_map, drone_frontiers, self.current_drone_pos, self.drone.radius),
        )
        drone_observation = np.concatenate([
            self.drone.belief_state.flatten(),
            normalize_pos(self.current_drone_pos),
            normalize_pos(self.current_rover_pos),
            packed_drone_frontiers,
            [self.drone.battery_level]
        ]).astype(np.float32)

        rover_frontiers = self.rover.detect_frontiers()
        packed_rover_frontiers = pack_frontiers(
            top_k_frontiers(self.shared_belief_map, rover_frontiers, self.current_rover_pos, self.rover.radius),
        )
        rover_observation = np.concatenate([
            self.shared_belief_map.flatten(),
            normalize_pos(self.current_rover_pos),
            normalize_pos(self.current_drone_pos),
            packed_rover_frontiers,
            [self.rover.battery_level]  
        ]).astype(np.float32)
        
        observations = {
            'drone': drone_observation,
            'rover': rover_observation
        }
        return observations
        
    def _map_action_to_frontier(self, action: int, scored_frontiers: list[tuple[int, int]]):
        k = NUM_FRONTIERS
        if action == k: return None
        if not scored_frontiers: return None
        
        idx = min(action, len(scored_frontiers)-1)
        frontier, _ = scored_frontiers[idx]
        return frontier
    
    def _compute_action_penalty(self, action, scored_frontiers) -> float:
        """Computes the penalty for selecting a suboptimal frontier, normalized to range [0, 1]"""
        
        k = NUM_FRONTIERS                    
        if action == k or len(scored_frontiers) <= 1: return 0.0
        _, best_score = scored_frontiers[0]
        if best_score < 1e-6:
            return 0.0
        
        idx = min(action, len(scored_frontiers)-1)
        _, chosen_score = scored_frontiers[idx] 
        return max(0.0, best_score - chosen_score) / best_score
        
    def _update_shared_belief(self, agent: Agent, current_pos: tuple[int, int]):
        assert self.ground_truth is not None
        observable_cells = get_observable_cells(agent.belief_state, current_pos, agent.radius)
        
        for r, c in observable_cells:
            sensor_reading = agent.get_sensor_reading((r, c), self.ground_truth)
            
            if sensor_reading == OCCUPIED:
                max_cell_accuracy = agent.confidence.occupied
                log_reading = np.log(agent.confidence.occupied / (1.0 - agent.confidence.occupied))
            else:
                max_cell_accuracy = agent.confidence.free
                log_reading = np.log((1.0 - agent.confidence.free) / agent.confidence.free)
            
            noisy_log_reading = np.random.normal(loc=log_reading, scale=COMMUNICATION_NOISE_SCALE)
            clipped_log_reading = np.clip(noisy_log_reading, agent.belief_update_limit.min, agent.belief_update_limit.max)
            
            assert self.cell_confidence is not None
            assert self.shared_belief_map is not None
            
            if max_cell_accuracy < self.cell_confidence[r, c]: continue
            
            prior = np.clip(self.shared_belief_map[r, c], 0.01, 0.99)
            log_prior = np.log(prior / (1.0 - prior))
            log_update = log_prior + clipped_log_reading
            
            self.shared_belief_map[r, c] = 1.0 / (1.0 + np.exp(-log_update))
            self.cell_confidence[r, c] = max(self.cell_confidence[r, c], max_cell_accuracy) 
            
    def _count_resolved_cells(self):
        """Counts the number of cells in the shared belief map that have a known status"""
        assert self.shared_belief_map is not None
        return int(np.sum((self.shared_belief_map < FREE_THRESHOLD) | (self.shared_belief_map > OBSTACLE_THRESHOLD)))
    
    def _get_reachable_frontiers(self, frontiers, current_pos):
        assert self.shared_belief_map is not None
        reachable = []
        
        for frontier in frontiers:
            path = astar_search(self.shared_belief_map, current_pos, frontier, threshold=0.55)
            if path:
                reachable.append((frontier, path))
        return reachable

    def _get_action_mask(self, actions):
        k = NUM_FRONTIERS
        mask = np.zeros((k+1), dtype=np.float32)
        for i in range(min(k, len(actions))):
            mask[i] = 1.0
        mask[k] = 1.0
        return mask