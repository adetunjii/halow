from noise import pnoise2
import numpy as np
import random
from constants import UNKNOWN, FREE, OCCUPIED

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