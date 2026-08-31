from mappo.actor import Actor

class Mappo:
    def __init__(self):
        self.drone_actor = Actor
        
    def _init_hyperparameters(self):
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.max_timesteps_per_episode = 200
        self.total_timesteps_per_batch = 2048