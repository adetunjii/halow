from pettingzoo import make, parallel_registry, register
from pettingzoo.test import parallel_api_test
from halow.environment import CustomEnvironment
from halow.mappo.mappo import Mappo
import matplotlib.pyplot as plt
import cProfile
import pstats

if __name__ == "__main__":
    # env = CustomEnvironment()
    # parallel_api_test(env, num_cycles=1_000_000)
    register("parallel", "thesis/CustomGridv0", CustomEnvironment)
    assert "thesis/CustomGridv0" in parallel_registry
    # parallel_api_test(env, num_cycles=1_000_000)

    customenv = CustomEnvironment()

    env = make("parallel", "thesis/CustomGridv0")

    _, inf = customenv.reset()
    policy = Mappo(customenv)
    policy.train(runs=50, resume=True)
    