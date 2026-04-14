"""Tests for the MuJoCo MJCF model files."""

import os

import mujoco
import pytest

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
SCENE_MODEL = os.path.join(ASSETS_DIR, "models", "scene.xml")


class TestSceneModel:
    """Validate that the default scene model loads and simulates correctly."""

    def test_model_loads(self) -> None:
        model = mujoco.MjModel.from_xml_path(SCENE_MODEL)
        assert model is not None

    def test_data_creation(self) -> None:
        model = mujoco.MjModel.from_xml_path(SCENE_MODEL)
        data = mujoco.MjData(model)
        assert data is not None

    def test_forward_step(self) -> None:
        model = mujoco.MjModel.from_xml_path(SCENE_MODEL)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        assert data.time > 0

    def test_model_has_expected_bodies(self) -> None:
        model = mujoco.MjModel.from_xml_path(SCENE_MODEL)
        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
        ]
        assert "table" in body_names
        assert "target_object" in body_names
