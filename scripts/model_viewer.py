import time
import mujoco.viewer
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type = str,
        help = "Path towards your xml file",
        required = True
    )
    
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    xml_path = args.path

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    n_steps = 5

    quit_flag = {"exit": False}   # mutable so the callback can modify it


    def keyboard_callback(keycode):
        """Called when a key is pressed."""
        if keycode in (ord('c'), ord('C')):
            cam = viewer.cam
            print("\n========== CAMERA CAPTURE ==========")
            print(f"viewer.cam.azimuth   = {cam.azimuth:.3f}")
            print(f"viewer.cam.elevation = {cam.elevation:.3f}")
            print(f"viewer.cam.distance  = {cam.distance:.3f}")
            print(f"viewer.cam.lookat    = [{cam.lookat[0]:.4f}, {cam.lookat[1]:.4f}, {cam.lookat[2]:.4f}]")
            print("====================================\n")
            quit_flag["exit"] = True


    # viewer shows frame of environment every n_steps
    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0.5, 0.0, 0.05]
        start = time.time()
        while True and not quit_flag["exit"]:
            step_start = time.time()
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)