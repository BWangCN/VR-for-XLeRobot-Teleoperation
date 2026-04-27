"""
Record Home Pose — manually position the robot arms, then press Enter to save.

Usage:
    conda activate xlerobot
    python record_home_pose.py --port1 COM4 --port2 COM3

Steps:
    1. Run this script (motors will go limp so you can move them by hand)
    2. Physically move both arms to the desired home/rest position
    3. Press Enter to save the pose to home_pose.json
    4. The teleop script will auto-load this file as the home position
"""

import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Record robot home pose")
    parser.add_argument("--port1", default="COM4", help="Left arm serial port")
    parser.add_argument("--port2", default="COM3", help="Right arm serial port")
    parser.add_argument("--output", default="home_pose.json", help="Output file")
    args = parser.parse_args()

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import MotorCalibration

    norm = MotorNormMode.DEGREES
    grip_norm = MotorNormMode.RANGE_0_100

    bus_left = FeetechMotorsBus(
        port=args.port1,
        motors={
            "left_arm_shoulder_pan":  Motor(1, "sts3215", norm),
            "left_arm_shoulder_lift": Motor(2, "sts3215", norm),
            "left_arm_elbow_flex":    Motor(3, "sts3215", norm),
            "left_arm_wrist_flex":    Motor(4, "sts3215", norm),
            "left_arm_wrist_roll":    Motor(5, "sts3215", norm),
            "left_arm_gripper":       Motor(6, "sts3215", grip_norm),
        },
    )
    bus_right = FeetechMotorsBus(
        port=args.port2,
        motors={
            "right_arm_shoulder_pan":  Motor(1, "sts3215", norm),
            "right_arm_shoulder_lift": Motor(2, "sts3215", norm),
            "right_arm_elbow_flex":    Motor(3, "sts3215", norm),
            "right_arm_wrist_flex":    Motor(4, "sts3215", norm),
            "right_arm_wrist_roll":    Motor(5, "sts3215", norm),
            "right_arm_gripper":       Motor(6, "sts3215", grip_norm),
        },
    )

    # Load calibration
    calib_path = os.path.join(
        os.path.expanduser("~"),
        ".cache", "huggingface", "lerobot", "calibration",
        "robots", "xlerobot", "None.json"
    )
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            raw = json.load(f)
        left_calib, right_calib = {}, {}
        for name, data in raw.items():
            mc = MotorCalibration(**data)
            if name.startswith("left_arm_"):
                left_calib[name] = mc
            elif name.startswith("right_arm_"):
                right_calib[name] = mc
        bus_left.calibration = left_calib
        bus_right.calibration = right_calib
        print(f"[OK] Calibration loaded")
    else:
        print(f"[WARNING] No calibration file found at {calib_path}")

    bus_left.connect()
    bus_right.connect()

    # Disable torque so user can move arms freely
    left_names = [f"left_arm_{j}" for j in
        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]
    right_names = [f"right_arm_{j}" for j in
        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]

    for name in left_names:
        bus_left.sync_write("Torque_Enable", {name: 0})
    for name in right_names:
        bus_right.sync_write("Torque_Enable", {name: 0})

    print()
    print("=" * 50)
    print("  Motors are now FREE — move arms by hand")
    print("  Position both arms in the desired HOME pose")
    print("=" * 50)
    print()

    while True:
        # Read and display current positions
        left_pos = bus_left.sync_read("Present_Position", left_names)
        right_pos = bus_right.sync_read("Present_Position", right_names)

        print("\r  L: ", end="")
        for name in left_names:
            short = name.replace("left_arm_", "")
            val = left_pos.get(name, 0)
            print(f"{short}={val:+6.1f} ", end="")
        print("| R: ", end="")
        for name in right_names:
            short = name.replace("right_arm_", "")
            val = right_pos.get(name, 0)
            print(f"{short}={val:+6.1f} ", end="")
        print("    ", end="", flush=True)

        # Check for Enter key (non-blocking on Windows)
        import msvcrt
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key in (b'\r', b'\n'):
                break

        import time
        time.sleep(0.1)

    # Save pose
    pose = {}
    left_pos = bus_left.sync_read("Present_Position", left_names)
    right_pos = bus_right.sync_read("Present_Position", right_names)
    pose.update(left_pos)
    pose.update(right_pos)

    # Convert numpy values to float for JSON
    pose = {k: float(v) for k, v in pose.items()}

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(out_path, "w") as f:
        json.dump(pose, f, indent=2)

    print()
    print()
    print(f"[SAVED] Home pose written to: {out_path}")
    print()
    for k, v in pose.items():
        print(f"  {k}: {v:+.1f}")

    bus_left.disconnect()
    bus_right.disconnect()


if __name__ == "__main__":
    main()
