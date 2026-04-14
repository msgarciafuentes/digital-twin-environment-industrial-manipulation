"""Tests for the ManipulationEnv Gymnasium environment."""

import numpy as np
import pytest

from envs.manipulation_env import ManipulationEnv


class TestManipulationEnv:
    """Validate the Gymnasium environment interface."""

    def test_env_creation(self) -> None:
        env = ManipulationEnv()
        assert env is not None
        env.close()

    def test_reset_returns_observation(self) -> None:
        env = ManipulationEnv()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_step_returns_correct_tuple(self) -> None:
        env = ManipulationEnv()
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_observation_space(self) -> None:
        env = ManipulationEnv()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        env.close()

    def test_multiple_steps(self) -> None:
        env = ManipulationEnv()
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            assert env.observation_space.contains(obs)
        env.close()
