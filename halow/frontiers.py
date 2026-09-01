import numpy as np
from halow.helpers import get_observable_cells, normalize_pos
from halow.constants import NUM_FRONTIERS, FREE_THRESHOLD, OBSTACLE_THRESHOLD

def next_best_view_score(belief_map: np.ndarray, frontier: tuple[int, int], current_pos: tuple[int, int], radius: int):
    frontier_row, frontier_col = frontier 
    observable_cells = get_observable_cells(belief_map, current_pos, radius)    
    total_info = sum(1 for r, c in observable_cells if FREE_THRESHOLD <= belief_map[r, c] <= OBSTACLE_THRESHOLD)
    distance = np.hypot(current_pos[0] - frontier_row, current_pos[1] - frontier_col)
    distance = max(distance, 1.0)
    eps = 1e-2
    score = (total_info + eps) / distance
    return score

def top_k_frontiers(belief_map, frontiers: list[tuple[int, int]], current_pos: tuple[int, int], radius: int):
    """Returns the top K frontiers from the belief map given the agent's current position and radius"""    
    k = NUM_FRONTIERS    
    frontier_scores = []
    for frontier in frontiers:
        score = next_best_view_score(belief_map, frontier, current_pos, radius)
        frontier_scores.append((frontier, score))
    frontier_scores.sort(key=lambda x: x[1], reverse=True)
    
    return frontier_scores[:k]

def pack_frontiers(frontiers):
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