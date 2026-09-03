import os
import torch
import numpy as np
from halow.environment import CustomEnvironment
from halow.mappo.actor import Actor
from halow.constants import COVERAGE_TARGET

root = os.path.dirname(__file__)

def evaluate_policy(num_episodes=50, render=False):
    env = CustomEnvironment()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    obs_space = env.observation_space("drone") 
    assert obs_space is not None

    obs_dim = obs_space.shape[0] # type: ignore
    action_dim = int(env.action_space("drone").n)

    # 1. Initialize actors and load trained weights
    drone_actor = Actor(obs_dim, action_dim, hidden_dim=64, num_layers=3).to(device)
    rover_actor = Actor(obs_dim, action_dim, hidden_dim=64, num_layers=3).to(device)

    checkpoint = torch.load(os.path.join(root, "mappo/weights/policy.pt"), map_location=device)
    drone_actor.load_state_dict(checkpoint["drone_actor"])
    rover_actor.load_state_dict(checkpoint["rover_actor"])

    drone_actor.eval()
    rover_actor.eval()

    eval_returns_drone, eval_returns_rover = [], []
    eval_coverages, eval_lengths, successes = [], [], []

    # 2. Test across unseen seeds outside the [0, 10000) training set
    test_seeds = range(10000, 10001)

    for seed in test_seeds:
        obs, infos = env.reset(seed=seed)
        terminated, truncated = False, False
        ep_return_drone, ep_return_rover = 0.0, 0.0
        steps = 0

        while not terminated and not truncated:
            with torch.no_grad():
                drone_obs = torch.tensor(obs["drone"], device=device)
                rover_obs = torch.tensor(obs["rover"], device=device)
                drone_mask = torch.tensor(infos["drone"]["action_mask"], device=device).bool()
                rover_mask = torch.tensor(infos["rover"]["action_mask"], device=device).bool()

                # Inference
                drone_action, _ = drone_actor.act(drone_obs, action_mask=drone_mask)
                rover_action, _ = rover_actor.act(rover_obs, action_mask=rover_mask)

                actions = {
                    "drone": drone_action.cpu().numpy(),
                    "rover": rover_action.cpu().numpy(),
                }

            obs, rewards, term, trunc, infos = env.step(actions)
            ep_return_drone += rewards["drone"]
            ep_return_rover += rewards["rover"]
            steps += 1

            if render:
                env.render()

            terminated = any(term.values())
            truncated = any(trunc.values())

        # Record metrics
        final_coverage = env._count_resolved_cells() / (env.height * env.width)
        eval_coverages.append(final_coverage)
        eval_returns_drone.append(ep_return_drone)
        eval_returns_rover.append(ep_return_rover)
        eval_lengths.append(steps)
        successes.append(final_coverage >= COVERAGE_TARGET)

    # 3. Summary metrics
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS ({num_episodes} Unseen Maps)")
    print("=" * 50)
    print(f"Success Rate:    {np.mean(successes) * 100:.1f}%")
    print(f"Mean Coverage:   {np.mean(eval_coverages) * 100:.2f}% ± {np.std(eval_coverages) * 100:.2f}%")
    print(f"Episode Length:  {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f} steps")
    print(f"Drone Return:    {np.mean(eval_returns_drone):.3f} ± {np.std(eval_returns_drone):.3f}")
    print(f"Rover Return:    {np.mean(eval_returns_rover):.3f} ± {np.std(eval_returns_rover):.3f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_policy(num_episodes=50, render=True)