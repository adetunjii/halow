from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
from halow.constants import *
from halow.helpers import get_observable_cells

class ConfidenceLevels(NamedTuple):
    free: float
    occupied: float
    
class UpdateLimit(NamedTuple):
    min: float
    max: float

@dataclass(slots=True)
class Agent:
    name: str
    confidence: ConfidenceLevels
    belief_update_limit: UpdateLimit
    radius: int
    belief_state: np.ndarray
    start_pos: tuple[int, int]
    battery_level: float
    
    def get_sensor_reading(self, current_pos: tuple[int, int], ground_truth: np.ndarray):
        """Get current sensor estimate of current cell"""
        
        row, col = current_pos 
        if ground_truth[row, col] == OCCUPIED:
            return OCCUPIED if np.random.random() < self.confidence.occupied else FREE
        elif ground_truth[row, col] == FREE:
            return FREE if np.random.random() < self.confidence.free else OCCUPIED
            
    def detect_frontiers(self):
        frontiers = []
        
        height, width = self.belief_state.shape
        for r in range(height):
            for c in range(width):
                if self.belief_state[r, c] > FREE_THRESHOLD:
                    continue
            
                for nr, nc in neighboring_cells:
                    frontier_row, frontier_col = r + nr, c + nc            
                    if 0 <= frontier_row < height and 0 <= frontier_col < width:
                        # a frontier cell has at least one unknown neighboring cell
                        if FREE_THRESHOLD <= self.belief_state[frontier_row, frontier_col] <= OBSTACLE_THRESHOLD:
                            frontiers.append((r, c))
                            break
        return frontiers    
        
    def update_internal_belief_state(self, current_pos: tuple[int, int], ground_truth):
        """Update agent's internal belief map based on current sensor readings"""
        
        observable_cells = get_observable_cells(self.belief_state, current_pos, self.radius)
        
        for r, c in observable_cells:
            prior = np.clip(self.belief_state[r, c], 0.01, 0.99)
            log_prior = np.log(prior / (1.0 - prior))
            
            sensor_reading = self.get_sensor_reading(current_pos, ground_truth)
            if sensor_reading == OCCUPIED:
                log_reading = np.log(self.confidence.occupied / (1.0 - self.confidence.occupied))
            else:
                log_reading = np.log((1.0 - self.confidence.free) / self.confidence.free) 
            log_update = log_prior + log_reading
            clipped_log_update = np.clip(log_update, self.belief_update_limit.min, self.belief_update_limit.max)
            
            posterior = 1.0 / (1.0 + np.exp(-clipped_log_update))
            self.belief_state[r, c] = posterior