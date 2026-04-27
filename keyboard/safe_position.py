"""
XLeRobot Safe Position Configuration

Edit SAFE_POSITION below to define where the robot returns before shutdown.
All values are in degrees (matching the robot's normalized range).
"""

import time


# ============================================================
#  Safe/home position (auto-recorded by record_safe_position.py)
# ============================================================
SAFE_POSITION = {
    "left_arm_shoulder_pan.pos":    -0.36,
    "left_arm_shoulder_lift.pos":   -99.23,
    "left_arm_elbow_flex.pos":   100.00,
    "left_arm_wrist_flex.pos":    -7.59,
    "left_arm_wrist_roll.pos":   -56.64,
    "left_arm_gripper.pos":     1.05,
    "right_arm_shoulder_pan.pos":    -5.34,
    "right_arm_shoulder_lift.pos":   -99.57,
    "right_arm_elbow_flex.pos":   100.00,
    "right_arm_wrist_flex.pos":     4.31,
    "right_arm_wrist_roll.pos":    99.95,
    "right_arm_gripper.pos":     0.28,
}
# ============================================================


def return_to_safe_position(robot, duration=3.0, fps=50):
    """
    Smoothly move robot to SAFE_POSITION over `duration` seconds,
    then disable torque.

    Uses cosine ease-in-out interpolation.
    """
    import math
    print(f"[SAFE] Returning to safe position over {duration:.1f}s...")

    try:
        obs = robot.get_observation()
    except Exception as e:
        print(f"[SAFE] Cannot read current position: {e}, disabling torque directly.")
        _disable_torque(robot)
        return

    start_pos = {}
    for key in SAFE_POSITION:
        start_pos[key] = obs.get(key, 0.0)

    steps = int(duration * fps)
    vel_keys = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    for step in range(steps + 1):
        t = step / steps
        alpha = 0.5 * (1.0 - math.cos(t * math.pi))

        action = {}
        for key in SAFE_POSITION:
            action[key] = start_pos[key] + alpha * (SAFE_POSITION[key] - start_pos[key])
        action.update(vel_keys)

        try:
            robot.send_action(action)
        except Exception as e:
            print(f"[SAFE] Send error at step {step}: {e}")
            break

        time.sleep(1.0 / fps)

    print("[SAFE] Safe position reached.")
    _disable_torque(robot)


def _disable_torque(robot):
    """Disable torque on all motors."""
    try:
        robot.bus1.disable_torque()
        print("[SAFE] Left arm torque disabled")
    except Exception:
        pass
    try:
        robot.bus2.disable_torque()
        print("[SAFE] Right arm torque disabled")
    except Exception:
        pass
