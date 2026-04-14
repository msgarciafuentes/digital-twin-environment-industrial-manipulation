# Digital Twin Environment — Industrial Manipulation

A custom MuJoCo simulated environment for industrial manipulation tasks,
designed as a foundation for reinforcement learning and digital twin research.

## Project Structure

```
├── assets/                 # MuJoCo simulation assets
│   ├── models/             # MJCF XML model files
│   │   └── scene.xml       # Default scene with table and target object
│   ├── meshes/             # 3D mesh files (STL, OBJ)
│   └── textures/           # Texture files (PNG, JPG)
├── config/                 # Configuration files
│   └── environment.yaml    # Environment and simulation parameters
├── envs/                   # Custom Gymnasium environments
│   ├── __init__.py
│   └── manipulation_env.py # ManipulationEnv — core Gym wrapper
├── scripts/                # Utility and runner scripts
│   ├── simulate.py         # Launch the MuJoCo interactive viewer
│   └── validate_model.py   # Validate MJCF model files
├── tests/                  # Test suite
│   ├── test_model.py       # Tests for MuJoCo model loading
│   └── test_env.py         # Tests for the Gymnasium environment
└── requirements.txt        # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.9+
- A working MuJoCo installation (bundled with the `mujoco` Python package)

### Installation

```bash
pip install -r requirements.txt
```

### Validate the Scene Model

```bash
python scripts/validate_model.py
```

### Launch the Interactive Viewer

```bash
python scripts/simulate.py
```

### Run the Test Suite

```bash
pytest tests/
```

## Usage

```python
from envs import ManipulationEnv

env = ManipulationEnv(render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```