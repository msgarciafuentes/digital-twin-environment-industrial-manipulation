import time
import argparse

import mujoco
import mujoco.viewer
import numpy as np


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path towards your xml file",
        required=True,
    )
    return parser


def get_id(model, obj_type, name, required=True):
    """Return MuJoCo object id by name."""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1 and required:
        raise ValueError(f"MuJoCo object not found: {name}")
    return obj_id


def set_geom_rgba(model, geom_id, rgba):
    """Set geom color if the geom exists."""
    if geom_id != -1:
        model.geom_rgba[geom_id] = rgba


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    xml_path = args.path

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    n_steps = 5
    quit_flag = {"exit": False}

    # =========================================================
    # COLORS
    # =========================================================
    GREEN_ON = np.array([0.0, 1.0, 0.0, 1.0])
    RED_ON = np.array([1.0, 0.0, 0.0, 1.0])
    YELLOW_ON = np.array([1.0, 1.0, 0.0, 1.0])
    WHITE_ON = np.array([1.0, 1.0, 1.0, 1.0])
    LED_OFF = np.array([0.1, 0.1, 0.1, 1.0])

    # =========================================================
    # OBJECT IDS FROM XML
    # =========================================================
    # Switch actuator sliders in the viewer
    switch_1_actuator_id = get_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "switch_motor_1"
    )
    push_button_actuator_id = get_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "switch_motor_2"
    )

    # Switch joint ids
    switch_1_joint_id = get_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "panel_switch_hinge_1"
    )
    push_button_joint_id = get_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "panel_switch_hinge_2"
    )

    # Linear actuator
    linear_actuator_id = get_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "linear_actuator_piston_motor"
    )

    # Wire control actuators
    wire_left_right_actuator_id = get_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "demo_wire_left_right_motor"
    )
    wire_front_back_actuator_id = get_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "demo_wire_front_back_motor"
    )
    wire_rotate_actuator_id = get_id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "demo_bnc_connector1_rotate_motor"
    )

    # LEDs
    green_led_geom_id = get_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "panel_led_green"
    )
    red_led_geom_id = get_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "panel_led_red"
    )
    yellow_led_geom_id = get_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "panel_led_yellow"
    )
    white_led_geom_id = get_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "panel_led_white"
    )

    # Joint state addresses
    switch_1_qpos_addr = model.jnt_qposadr[switch_1_joint_id]
    push_button_qpos_addr = model.jnt_qposadr[push_button_joint_id]
    push_button_qvel_addr = model.jnt_dofadr[push_button_joint_id]

    # =========================================================
    # LOGIC STATE VARIABLES
    # =========================================================
    # Requirement 1:
    # Button press -> immediately reset button -> wait 5 seconds -> green LED on for 5 seconds.
    push_button_pressed_time = None
    push_button_cycle_active = False

    # Requirement 2:
    # Toggle switch ON -> slowly increase piston target to 1.
    # Toggle switch OFF -> slowly decrease piston target to 0.
    linear_actuator_target = 0.0
    linear_speed_per_second = 1  # 0 -> 1 takes around 6.7 seconds

    # Thresholds. Your UI sliders are expected to be set to 0 or 1.
    ON_THRESHOLD = 0.99
    
    wire_red_timer_start = None
    wire_red_delay_seconds = 3.0

    # =========================================================
    # KEYBOARD CALLBACK
    # =========================================================
    def keyboard_callback(keycode):
        """Press C to capture camera settings and exit."""
        if keycode in (ord("c"), ord("C")):
            cam = viewer.cam
            print("\n========== CAMERA CAPTURE ==========")
            print(f"viewer.cam.azimuth   = {cam.azimuth:.3f}")
            print(f"viewer.cam.elevation = {cam.elevation:.3f}")
            print(f"viewer.cam.distance  = {cam.distance:.3f}")
            print(
                "viewer.cam.lookat    = "
                f"[{cam.lookat[0]:.4f}, {cam.lookat[1]:.4f}, {cam.lookat[2]:.4f}]"
            )
            print("====================================\n")
            quit_flag["exit"] = True

    # =========================================================
    # VIEWER
    # =========================================================
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=keyboard_callback,
    ) as viewer:

        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0.5, 0.0, 0.05]

        # Start with LEDs off
        set_geom_rgba(model, green_led_geom_id, LED_OFF)
        set_geom_rgba(model, red_led_geom_id, LED_OFF)
        set_geom_rgba(model, yellow_led_geom_id, LED_OFF)
        set_geom_rgba(model, white_led_geom_id, LED_OFF)

        while not quit_flag["exit"]:
            step_start = time.time()

            # Run simulation
            for _ in range(n_steps):
                mujoco.mj_step(model, data)

            dt = model.opt.timestep * n_steps
            current_time = data.time

            # =================================================
            # 1. PUSH BUTTON -> DELAYED GREEN LED
            # =================================================
            push_button_ctrl = data.ctrl[push_button_actuator_id]
            push_button_joint_pos = data.qpos[push_button_qpos_addr]

            # In your XML, panel_switch_hinge_2 has range 0 to 0.05,
            # so qpos near 0.05 means the physical push button is pressed.
            push_button_is_pressed = (
                push_button_ctrl >= ON_THRESHOLD
                or push_button_joint_pos >= 0.045
            )

            # Start the cycle only once when the button is pressed.
            # Do NOT reset the button yet.
            if push_button_is_pressed and not push_button_cycle_active:
                push_button_pressed_time = current_time
                push_button_cycle_active = True

            green_led_should_be_on = False

            if push_button_cycle_active and push_button_pressed_time is not None:
                elapsed_after_press = current_time - push_button_pressed_time

                # Wait 5 seconds first
                if elapsed_after_press < 5.0:
                    green_led_should_be_on = False

                # Then green LED ON for 5 seconds
                elif 5.0 <= elapsed_after_press < 10.0:
                    green_led_should_be_on = True

                # After green LED turns off, reset the button
                else:
                    green_led_should_be_on = False

                    # Reset actuator slider/control to 0 only after green turns off
                    data.ctrl[push_button_actuator_id] = 0.0

                    # Reset physical button joint state to 0
                    data.qpos[push_button_qpos_addr] = 0.0
                    data.qvel[push_button_qvel_addr] = 0.0
                    mujoco.mj_forward(model, data)

                    # Reset cycle state
                    push_button_pressed_time = None
                    push_button_cycle_active = False

            if green_led_should_be_on:
                set_geom_rgba(model, green_led_geom_id, GREEN_ON)
            else:
                set_geom_rgba(model, green_led_geom_id, LED_OFF)

            # =================================================
            # 2. TOGGLE SWITCH -> LINEAR ACTUATOR UP/DOWN
            # =================================================
            toggle_switch_ctrl = data.ctrl[switch_1_actuator_id]
            toggle_switch_joint_pos = data.qpos[switch_1_qpos_addr]

            # In your XML, panel_switch_hinge_1 has range -0.6 to 0.6.
            # If the joint is physically moved near the positive limit,
            # treat it as ON too.
            toggle_switch_is_on = (
                toggle_switch_ctrl >= ON_THRESHOLD
                or toggle_switch_joint_pos >= 0.5
            )

            if toggle_switch_is_on:
                # Slowly open / move upward
                linear_actuator_target += linear_speed_per_second * dt
            else:
                # Slowly close / move downward
                linear_actuator_target -= linear_speed_per_second * dt

            linear_actuator_target = np.clip(linear_actuator_target, 0.0, 1.0)
            data.ctrl[linear_actuator_id] = linear_actuator_target

            # When piston reaches 1, turn ON yellow and white LEDs.
            # When piston reaches 0, turn them OFF.
            if linear_actuator_target >= 0.999:
                set_geom_rgba(model, yellow_led_geom_id, YELLOW_ON)
                set_geom_rgba(model, white_led_geom_id, WHITE_ON)

            elif linear_actuator_target <= 0.001:
                set_geom_rgba(model, yellow_led_geom_id, LED_OFF)
                set_geom_rgba(model, white_led_geom_id, LED_OFF)

            # =================================================
            # 3. WIRE SLIDERS + WIRE ROTATION -> RED LED WITH 3-SECOND DELAY
            # =================================================
            wire_left_right_ctrl = data.ctrl[wire_left_right_actuator_id]
            wire_front_back_ctrl = data.ctrl[wire_front_back_actuator_id]
            wire_rotate_ctrl = data.ctrl[wire_rotate_actuator_id]

            wire_conditions_are_on = (
                wire_left_right_ctrl >= ON_THRESHOLD
                and wire_front_back_ctrl >= ON_THRESHOLD
                and wire_rotate_ctrl >= ON_THRESHOLD
            )

            if wire_conditions_are_on:
                # Start timer only once when all three controls become 1
                if wire_red_timer_start is None:
                    wire_red_timer_start = current_time

                elapsed_wire_time = current_time - wire_red_timer_start

                # Red LED turns on only after 3 seconds
                if elapsed_wire_time >= wire_red_delay_seconds:
                    set_geom_rgba(model, red_led_geom_id, RED_ON)
                else:
                    set_geom_rgba(model, red_led_geom_id, LED_OFF)

            else:
                # If any condition is not 1, reset timer and turn red LED off
                wire_red_timer_start = None
                set_geom_rgba(model, red_led_geom_id, LED_OFF)

            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)