#!/usr/bin/env python3
"""Validate that a MuJoCo MJCF model loads without errors."""

import argparse
import os
import sys

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a MuJoCo MJCF XML model file."
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

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as exc:
        print(f"FAILED — {exc}", file=sys.stderr)
        sys.exit(1)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Model OK: {model_path}")
    print(f"  bodies    : {model.nbody}")
    print(f"  joints    : {model.njnt}")
    print(f"  DOFs      : {model.nv}")
    print(f"  actuators : {model.nu}")
    print(f"  geoms     : {model.ngeom}")
    print(f"  meshes    : {model.nmesh}")
    print(f"  textures  : {model.ntex}")


if __name__ == "__main__":
    main()
