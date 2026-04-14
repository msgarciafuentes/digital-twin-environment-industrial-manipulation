#!/usr/bin/env python3
"""Launch the MuJoCo viewer with the default scene model."""

import argparse
import os
import sys

import mujoco
import mujoco.viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch MuJoCo viewer for the industrial manipulation scene."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "assets", "models", "scene.xml"
        ),
        help="Path to the MJCF XML model file.",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"Loaded model: {model_path}")
    print(f"  bodies : {model.nbody}")
    print(f"  joints : {model.njnt}")
    print(f"  DOFs   : {model.nv}")
    print(f"  actuators: {model.nu}")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
