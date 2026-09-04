import numpy as np
from halow.helpers import get_observable_cells, normalize_pos
from halow.constants import NUM_FRONTIERS, FREE_THRESHOLD, OBSTACLE_THRESHOLD

# def next_best_view_score(belief_map: np.ndarray, frontier: tuple[int, int], current_pos: tuple[int, int], radius: int):
#     observable_cells = get_observable_cells(belief_map, frontier, radius)    
#     total_info = sum(1 for r, c in observable_cells if FREE_THRESHOLD <= belief_map[r, c] <= OBSTACLE_THRESHOLD)
#     distance = np.hypot(current_pos[0] - frontier[0], current_pos[1] - frontier[1])
#     distance = max(distance, 1.0)
#     eps = 1e-2
#     score = (total_info + eps) / distance
#     return score

def nbv_score(belief_map: np.ndarray, frontier: tuple[int, int], current_pos: tuple[int, int], radius: int, agent: str, secondary_target: tuple[int, int] | None, steps_per_action: int):
    """Scores a frontier based on info gain, agent speed, verification bonus, and peer deconfliction"""
    observable_cells = get_observable_cells(belief_map, frontier, radius)
    
    if agent == "drone":
        total_info = sum(1 for r, c in observable_cells if FREE_THRESHOLD <= belief_map[r, c] <= OBSTACLE_THRESHOLD)
    else:
        total_info = sum(1 for r, c in observable_cells if FREE_THRESHOLD <= belief_map[r, c] <= OBSTACLE_THRESHOLD)
        # incentivize verifying the areas that drone has scanned
        verification_bonus = sum(0.3 for r, c in observable_cells if (0.35 <= belief_map[r, c] < FREE_THRESHOLD or OBSTACLE_THRESHOLD < belief_map[r, c] <= 0.65))
        total_info += verification_bonus
        
    distance = np.hypot(current_pos[0] - frontier[0], current_pos[1] - frontier[1])
    distance = max(distance / steps_per_action, 1.0) # normalize distance based on how many steps an agent can take
    eps = 1e-2
    
    score = (total_info + eps) / distance
    
    # penalize frontiers close to where the next agent is going
    if secondary_target is not None:
        overlap_dist = np.hypot(frontier[0] - secondary_target[0], frontier[1] - secondary_target[1])
        overlap_penalty = 1.0 / (1.0 + overlap_dist)
        score *= (1.0 - 0.5 * overlap_penalty)
    
    return score
    
def top_k_frontiers(belief_map, frontiers: list[tuple[int, int]], current_pos: tuple[int, int], radius: int, agent: str, secondary_target: tuple[int, int] | None, steps_per_action) -> list[tuple[tuple[int, int], int]]:
    """Returns the top K frontiers from the belief map given the agent's current position and radius"""    
    k = NUM_FRONTIERS    
    frontier_scores = []
    for frontier in frontiers:
        score = nbv_score(belief_map, frontier, current_pos, radius, agent, secondary_target, steps_per_action)
        frontier_scores.append((frontier, score))
    frontier_scores.sort(key=lambda x: x[1], reverse=True)
    
    return frontier_scores[:k]

def pack_frontiers(frontiers):
    """Packs top K frontier coordinates and normalized scores into a flat vector for observation"""
    k = NUM_FRONTIERS    
    best_scores = frontiers[0][1] if frontiers else 1.0
    best_score = max(best_scores, 1e-6)
    
    packed = []
    for i in range(k):
        if i < len(frontiers):
            frontier_pos, score = frontiers[i]
            normalized_frontier_row, normalized_frontier_col = normalize_pos(frontier_pos)
            normalized_score = score/best_score
            packed.extend([normalized_frontier_row, normalized_frontier_col, normalized_score])
        else:
            packed.extend([-1.0, -1.0, 0]) # pad the list in a case where there are less than k frontiers
    return np.array(packed, dtype=np.float32)