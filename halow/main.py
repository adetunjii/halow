from pettingzoo import make, parallel_registry, register
from pettingzoo.test import parallel_api_test
from halow.environment import CustomEnvironment
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # env = CustomEnvironment()
    # parallel_api_test(env, num_cycles=1_000_000)
    register("parallel", "thesis/CustomGridv0", CustomEnvironment)
    assert "thesis/CustomGridv0" in parallel_registry
    # parallel_api_test(env, num_cycles=1_000_000)

    customenv = CustomEnvironment()
    print(customenv.action_space("drone").shape)

    env = make("parallel", "thesis/CustomGridv0")
    observations, infos = env.reset(seed=42)
    
    done = False
    
    for _ in range(1000):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        env.render() 
    env.close()
    plt.show(block=True)