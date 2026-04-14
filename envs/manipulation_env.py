"""Custom MuJoCo Gymnasium environment for industrial manipulation."""

import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class ManipulationEnv(gym.Env):
    """A Gymnasium environment wrapping a MuJoCo industrial manipulation scene.

    This environment loads a MuJoCo MJCF model and exposes it through the
    standard Gymnasium API for reinforcement learning and simulation.

    Attributes:
        metadata: Gymnasium metadata specifying supported render modes.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "assets",
                "models",
                "scene.xml",
            )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Observation: all qpos and qvel
        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Action: one entry per actuator.  When the model has no actuators
        # yet (nu == 0), default to a single no-op action dimension so that
        # the action space is always valid for sampling.
        num_actuators = max(self.model.nu, 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_actuators,), dtype=np.float64
        )

        self.renderer: Any = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the simulation to its initial state."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the simulation by one timestep."""
        if self.model.nu > 0:
            self.data.ctrl[:] = action[: self.model.nu]

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the environment."""
        if self.render_mode is None:
            return None

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)

        self.renderer.update_scene(self.data)
        pixels = self.renderer.render()

        if self.render_mode == "rgb_array":
            return pixels
        return None

    def close(self) -> None:
        """Clean up renderer resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return the current observation vector."""
        return np.concatenate([self.data.qpos, self.data.qvel])
