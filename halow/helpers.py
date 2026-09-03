import numpy as np
import heapq
from PIL import Image, ImageSequence
from halow.constants import neighboring_cells, HEIGHT, WIDTH, PATH_RESOLUTION, FREE, OCCUPIED
import random
from noise import pnoise2
import os
import csv

def get_observable_cells(belief_map: np.ndarray, current_pos: tuple[int, int], radius: int):
    """Get agent's observable cells given its radius"""
    
    observable_cells = []
    
    height, width = belief_map.shape 
    for r in range(-radius, radius+1):
        for c in range(-radius, radius+1):
            if r**2 + c**2 <= radius**2:
                nr, nc = current_pos[0] + r, current_pos[1] + c
                
                if 0 <= nr < height and 0 <= nc < width:
                    observable_cells.append((nr, nc))
    
    return observable_cells

def astar_search(grid: np.ndarray, start_pos, target_pos, threshold=0.6) -> list[tuple[int, int]] | None:
    """A* search for a viable path in a 2D grid given a start position and a target"""
    
    height, width = grid.shape
    
    def neighbors(position):
        r, c = position
        
        for neighbor in neighboring_cells:
            nr, nc = r + neighbor[0], c + neighbor[1]
            
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] < threshold:
                yield(nr, nc)
                
    openset = []
    visited = set()
    
    heapq.heappush(openset, (0, start_pos))
    
    parent = dict()
    g_score = dict({start_pos: 0})
    
    path = []
    
    while openset:
        _, current_pos = heapq.heappop(openset)
        if current_pos in visited:
            continue
        
        visited.add(current_pos)
        
        if current_pos == target_pos:
            while current_pos in parent:
                path.append(current_pos)
                current_pos = parent[current_pos]
            path.reverse()
            
            return path
        
        for neighbor in neighbors(current_pos):
            if neighbor in visited:
                continue
            
            distance = np.hypot(neighbor[0] - current_pos[0], neighbor[1] - current_pos[1])
            
            tentative_gcost = g_score[current_pos] + distance
            
            if tentative_gcost < g_score.get(neighbor, float('inf')):
                parent[neighbor] = current_pos
                g_score[neighbor] = tentative_gcost
                
                h_cost = np.hypot(neighbor[0] - target_pos[0], neighbor[1] - target_pos[1])
                f_cost = tentative_gcost + h_cost
                heapq.heappush(openset, (f_cost, neighbor))
    
    return None

def interpolate_path(start_pos, target_pos):
    start_row, start_col = start_pos
    target_row, target_col = target_pos
    
    distance = np.hypot(target_col - start_col, target_row - start_row)
    num_steps = max(int(distance/PATH_RESOLUTION), 2)
    
    t = np.linspace(0, 1, num_steps)
    
    row_segment = (start_row + (target_row - start_row) * t)[1:]
    col_segment = (start_col + (target_col - start_col) * t)[1:]

    return list(zip(row_segment, col_segment))

def normalize_pos(pos: tuple[int, int]) -> np.ndarray:
    row, col = pos
    return np.array([row/HEIGHT, col/WIDTH], dtype=np.float32)

def get_animation_frames(path: str):
    try:
        img = Image.open(path)
    
        frames = []
        for frame in ImageSequence.Iterator(img):
            frames.append(np.array(frame.convert('RGBA')))
        
        return frames
    except FileNotFoundError:
        print(f"image path {path} does not exist")

def generate_map(height, width, seed=None, scale=0.15, threshold=0.2):
    grid = np.full((height, width), FREE, dtype=np.float32)
    
    if seed is None:
        seed = random.randint(0, 1000)
        
    for r in range(height):
        for c in range(width):
            value = pnoise2(c * scale + seed, r * scale + seed)
            
            if value > threshold:
                grid[r, c] = OCCUPIED
    
    return grid

def setup_logger(path: str):
        if os.path.exists(path):
            with open(path, "w", newline="") as f:
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

def log_run(log_path: str, run, episodes, critic_losses, drone_actor_losses, rover_actor_losses, drone_entropies, rover_entropies, drone_grad_norm, rover_grad_norm, critic_grad_norm):
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
    
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)