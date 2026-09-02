import unittest
import torch
import numpy as np
from halow.mappo.buffer import RolloutBuffer  # Adjust import path accordingly


class TestRolloutBuffer(unittest.TestCase):
    """Unit tests for the RolloutBuffer class."""

    def setUp(self):
        self.num_agents = 2
        self.action_space = 5         
        self.observation_space = 423 
        self.global_state_space = 100 
        self.gamma = 0.99
        self.td_lambda = 0.95
        self.num_episodes = 3

        self.buffer = RolloutBuffer(
            action_space=self.action_space,
            observation_space=self.observation_space,
            global_state_space=self.global_state_space,
            gamma=self.gamma,
            td_lambda=self.td_lambda,
            num_episodes=self.num_episodes,
            num_agents=self.num_agents,
        )
        self.buffer.device = "cpu"

    def _mock_episode(self, length: int = 10):
        N = self.num_agents
        obs_dim = self.observation_space
        action_dim = self.action_space
        state_dim = self.global_state_space

        return {
            "observations": [np.random.randn(N, obs_dim).astype(np.float32) for _ in range(length)],
            "action_masks": [np.random.randint(0, 1, (N, action_dim)).astype(bool) for _ in range(length)],
            "actions": [np.random.randint(0, action_dim, (N,)).astype(np.int64) for _ in range(length)],
            "rewards": [np.random.randn(N,).astype(np.float32) for _ in range(length)],
            "values": [np.random.randn(1,).astype(np.float32) for _ in range(length)],
            "global_state": [np.random.randn(state_dim,).astype(np.float32) for _ in range(length)],
            "terminated": [False] * (length - 1) + [True],
            "final_value": [0.0]
        }

    def test_rollout_to_tensor(self):
        mock_episode = self._mock_episode(length=10)
        eps_dict = self.buffer.rollout_to_tensor(mock_episode)

        self.assertIn("observations", eps_dict)
        self.assertIsInstance(eps_dict["observations"], torch.Tensor)
        self.assertEqual(eps_dict["observations"].shape, (10, self.num_agents, self.observation_space))
        self.assertEqual(eps_dict["observations"].dtype, torch.float32)
        self.assertEqual(eps_dict["actions"].dtype, torch.int64)
        self.assertEqual(eps_dict["action_masks"].dtype, torch.bool)
        self.assertEqual(eps_dict["rewards"].shape, (10, self.num_agents))

    def test_compute_advantages_terminated(self):
        """Test that a terminated episode bootstraps to 0 at the last step."""
        T = 3
        episode = {
            "observations": torch.randn(T, self.num_agents, self.observation_space),
            "rewards": torch.ones(T, self.num_agents),
            "values": torch.arange(T, dtype=torch.float32).unsqueeze(1).expand(-1, self.num_agents),
            "terminated": [False, False, True],
            "final_value": torch.tensor(0.0) 
        }
        last_reward = episode["rewards"][-1]
        self.buffer.compute_advantages(episode)
        self.assertIn("returns", episode)
        self.assertIn("advantages", episode)
        self.assertEqual(episode["returns"].shape, (T, self.num_agents))
        self.assertTrue(torch.allclose(episode["returns"][-1], last_reward))

    def test_compute_advantages_truncated(self):
        T = 3
        N = self.num_agents
        episode = {
            "observations": torch.randn(T, N, self.observation_space),
            "rewards": torch.ones(T, N) * 1.0,
            "values": torch.full((T, N), 5.0),
            "terminated": [False, False, False],
            "final_value": torch.tensor(5.0)
        }

        last_reward = episode["rewards"][-1]
        last_value = episode["values"][-1]

        self.buffer.compute_advantages(episode)
        expected = last_reward + self.gamma * ((self.td_lambda * last_value) + (1-self.td_lambda) * last_value)
        self.assertTrue(torch.allclose(episode["returns"][-1], expected))

    def test_add_episode(self):
        """Test that adding an episode computes advantages and stores it correctly."""
        dummy = self._mock_episode(length=10)

        self.assertEqual(self.buffer.episode_idx, 0)
        self.buffer.add(dummy)
        self.assertEqual(self.buffer.episode_idx, 1)
        self.assertIsNotNone(self.buffer.episodes[0])
        self.assertIn("returns", self.buffer.episodes[0]) # type: ignore
        self.assertIn("advantages", self.buffer.episodes[0]) # type: ignore
        self.assertNotIn("rewards", self.buffer.episodes[0]) # type: ignore

    def test_batchify_single_episode(self):
        mock_episode =  self._mock_episode(length=10)
        self.buffer.add(mock_episode)

        obs, masks, actions, returns, advantages, states = self.buffer.batchify()

        T = 10
        obs_dim = self.observation_space
        act_dim = self.action_space
        state_dim = self.global_state_space

        self.assertEqual(obs.shape, (T * self.num_agents, obs_dim))
        self.assertEqual(masks.shape, (T * self.num_agents, act_dim))
        self.assertEqual(actions.shape, (T * self.num_agents,))
        self.assertEqual(returns.shape, (T * self.num_agents,))
        self.assertEqual(advantages.shape, (T * self.num_agents,))
        self.assertEqual(states.shape, (T, state_dim))

    def test_batchify_multiple_episodes(self):
        dummy1 = self._mock_episode(length=10)
        dummy2 = self._mock_episode(length=5)

        self.buffer.add(dummy1)
        self.buffer.add(dummy2)

        obs, masks, actions, returns, advantages, states = self.buffer.batchify()

        self.assertEqual(obs.shape, (15 * self.num_agents, self.observation_space))
        self.assertEqual(states.shape, (15, self.global_state_space))
        self.assertEqual(self.buffer.episode_idx, 0)
        self.assertTrue(all(e is None for e in self.buffer.episodes))

    def test_batchify_empty_buffer(self):
        obs, masks, actions, returns, advantages, states = self.buffer.batchify()

        self.assertEqual(obs.shape, (0, self.observation_space))
        self.assertEqual(masks.shape, (0, self.action_space))
        self.assertEqual(actions.shape, (0,))
        self.assertEqual(returns.shape, (0,))
        self.assertEqual(advantages.shape, (0,))
        self.assertEqual(states.shape, (0, self.global_state_space))

    def test_dtype_preservation(self):
        dummy = self._mock_episode(length=5)
        self.buffer.add(dummy)

        obs, masks, actions, returns, advantages, states = self.buffer.batchify()

        self.assertEqual(actions.dtype, torch.int32)
        self.assertEqual(masks.dtype, torch.bool)
        self.assertEqual(obs.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)
        self.assertEqual(advantages.dtype, torch.float32)

if __name__ == "__main__":
    unittest.main()