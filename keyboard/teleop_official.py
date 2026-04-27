"""
XLeRobot Official Keyboard Teleoperation (adapted for dual-arm only, no head/base)

Based on: XLeRobot/software/examples/4_xlerobot_teleop_keyboard.py

Controls:
  Left Arm:
    Q/E       Shoulder pan +/-
    W/S       End-effector X +/- (forward/back)
    A/D       End-effector Y +/- (up/down)
    Z/X       Pitch +/-
    R/F       Wrist roll +/-
    T/G       Gripper open/close
    C         Reset left arm to zero

  Right Arm:
    U/O       Shoulder pan +/-
    I/K       End-effector X +/-
    J/L       End-effector Y +/-
    N/M       Pitch +/-
    P/;       Wrist roll +/-
    [/]       Gripper open/close
    B         Reset right arm to zero

  ESC / `    Quit (return to safe position, then disable torque)
"""

import time
import numpy as np
import math

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
from lerobot.utils.robot_utils import precise_sleep
from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from safe_position import return_to_safe_position

# Try to import rerun (optional)
try:
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False
    print("[INFO] rerun not installed, visualization disabled. Install with: pip install rerun-sdk")

# Keymaps
LEFT_KEYMAP = {
    'shoulder_pan+': 'q', 'shoulder_pan-': 'e',
    'wrist_roll+': 'r', 'wrist_roll-': 'f',
    'gripper+': 't', 'gripper-': 'g',
    'x+': 'w', 'x-': 's', 'y+': 'a', 'y-': 'd',
    'pitch+': 'z', 'pitch-': 'x',
    'reset': 'c',
}
# Right arm uses IJKL cluster (mirrors WASD)
RIGHT_KEYMAP = {
    'shoulder_pan+': 'u', 'shoulder_pan-': 'o',
    'wrist_roll+': 'p', 'wrist_roll-': ';',
    'gripper+': '[', 'gripper-': ']',
    'x+': 'i', 'x-': 'k', 'y+': 'j', 'y-': 'l',
    'pitch+': 'n', 'pitch-': 'm',
    'reset': 'b',
}

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}


class SimpleTeleopArm:
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=0.81):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        self.degree_step = 3
        self.xy_step = 0.0081
        self.target_positions = {
            "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
            "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 0.0,
        }
        self.zero_pos = {
            "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
            "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 0.0,
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position...")
        self.target_positions = self.zero_pos.copy()
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        self.target_positions["wrist_flex"] = 0.0
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_keys(self, key_state):
        if key_state.get('shoulder_pan+'):
            self.target_positions["shoulder_pan"] += self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('shoulder_pan-'):
            self.target_positions["shoulder_pan"] -= self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('wrist_roll+'):
            self.target_positions["wrist_roll"] += self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('wrist_roll-'):
            self.target_positions["wrist_roll"] -= self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('gripper+'):
            self.target_positions["gripper"] += self.degree_step
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('gripper-'):
            self.target_positions["gripper"] -= self.degree_step
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('pitch+'):
            self.pitch += self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")
        if key_state.get('pitch-'):
            self.pitch -= self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")

        moved = False
        if key_state.get('x+'):
            self.current_x += self.xy_step; moved = True
        if key_state.get('x-'):
            self.current_x -= self.xy_step; moved = True
        if key_state.get('y+'):
            self.current_y += self.xy_step; moved = True
        if key_state.get('y-'):
            self.current_y -= self.xy_step; moved = True
        if moved:
            joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = joint2
            self.target_positions["elbow_flex"] = joint3
            print(f"[{self.prefix}] x={self.current_x:.4f}, y={self.current_y:.4f}")

        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            - self.target_positions["elbow_flex"]
            + self.pitch
        )

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action


def main():
    FPS = 50

    robot_config = XLerobotConfig()
    robot = XLerobot(robot_config)

    try:
        robot.connect()
        print("[MAIN] Connected to robot!")
    except Exception as e:
        print(f"[MAIN] Failed to connect: {e}")
        return

    if HAS_RERUN:
        init_rerun(session_name="xlerobot_teleop")

    # Init keyboard
    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)
    keyboard.connect()

    # Init arms
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(kin_left, LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(kin_right, RIGHT_JOINT_MAP, obs, prefix="right")

    # Move to zero position
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)

    print("\n[MAIN] Teleop active! Controls:")
    print("  Left arm:  WASD=move, QE=shoulder, ZX=pitch, RF=wrist, TG=gripper, C=reset")
    print("  Right arm: IJKL=move, UO=shoulder, NM=pitch, P;=wrist, []=gripper, B=reset")
    print("  ` (backtick) = quit safely | Ctrl+C = emergency quit\n")

    try:
        while True:
            pressed_keys = set(keyboard.get_action().keys())
            left_key_state = {action: (key in pressed_keys) for action, key in LEFT_KEYMAP.items()}
            right_key_state = {action: (key in pressed_keys) for action, key in RIGHT_KEYMAP.items()}

            # Quit key: backtick (`)
            if '`' in pressed_keys:
                print("\n[MAIN] Quit requested, returning to safe position...")
                break

            # Reset
            if left_key_state.get('reset'):
                left_arm.move_to_zero_position(robot)
                continue
            if right_key_state.get('reset'):
                right_arm.move_to_zero_position(robot)
                continue

            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)

            # No base/head action needed
            action = {**left_action, **right_action, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
            robot.send_action(action)

            obs = robot.get_observation()
            if HAS_RERUN:
                log_rerun_data(obs, action)

    except KeyboardInterrupt:
        print("\n[MAIN] Emergency stop!")
    finally:
        return_to_safe_position(robot)
        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            keyboard.disconnect()
        except Exception:
            pass
        print("Teleoperation ended.")


if __name__ == "__main__":
    main()
