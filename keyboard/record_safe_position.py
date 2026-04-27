"""
Record Safe Position for XLeRobot

Usage:
  1. Run this script
  2. Motors will go limp - manually move arms to your desired safe position
  3. Press ENTER to record
  4. Motors lock to verify the position
  5. Press ENTER to confirm and save, or 'r' to redo

The recorded position is saved to safe_position.py automatically.
"""

from lerobot.robots.xlerobot import XLerobot, XLerobotConfig

JOINTS = [
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
    "left_arm_gripper",
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
]

SAFE_POS_FILE = __file__.replace("record_safe_position.py", "safe_position.py")


def main():
    print("=" * 60)
    print("  XLeRobot Safe Position Recorder")
    print("=" * 60)

    print("\nConnecting to robot...")
    robot = XLerobot(XLerobotConfig())
    robot.connect()
    print("Connected!")

    while True:
        # Disable torque so user can move arms freely
        print("\n[1/3] Motors are now LIMP. Move both arms to your desired safe position.")
        robot.bus1.disable_torque()
        robot.bus2.disable_torque()

        input("     Press ENTER when ready to record...")

        # Read current positions
        obs = robot.get_observation()
        recorded = {}
        for j in JOINTS:
            recorded[j] = obs.get(f"{j}.pos", 0.0)

        print("\n[2/3] Recorded position:")
        for j in JOINTS:
            side = "L" if j.startswith("left") else "R"
            name = j.split("_", 2)[2]
            print(f"  {side} {name:20s} = {recorded[j]:>8.2f}")

        # Re-enable torque and move to recorded position to verify
        print("\n[3/3] Locking motors at recorded position for verification...")
        from lerobot.motors.feetech import OperatingMode
        for name in robot.left_arm_motors:
            robot.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
        for name in robot.right_arm_motors:
            robot.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)
        robot.bus1.enable_torque()
        robot.bus2.enable_torque()

        action = {f"{j}.pos": recorded[j] for j in JOINTS}
        action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        robot.send_action(action)

        choice = input("     Looks good? Press ENTER to save, or 'r' + ENTER to redo: ").strip().lower()
        if choice != 'r':
            break

    # Write to safe_position.py
    lines = [
        '"""',
        'XLeRobot Safe Position Configuration',
        '',
        'Edit SAFE_POSITION below to define where the robot returns before shutdown.',
        'All values are in degrees (matching the robot\'s normalized range).',
        '"""',
        '',
        'import time',
        '',
        '',
        '# ============================================================',
        '#  Safe/home position (auto-recorded by record_safe_position.py)',
        '# ============================================================',
        'SAFE_POSITION = {',
    ]
    for j in JOINTS:
        lines.append(f'    "{j}.pos": {recorded[j]:>8.2f},')
    lines += [
        '}',
        '# ============================================================',
        '',
        '',
        'def return_to_safe_position(robot, duration=3.0, fps=50):',
        '    """',
        '    Smoothly move robot to SAFE_POSITION over `duration` seconds,',
        '    then disable torque.',
        '',
        '    Uses cosine ease-in-out interpolation.',
        '    """',
        '    import math',
        '    print(f"[SAFE] Returning to safe position over {duration:.1f}s...")',
        '',
        '    try:',
        '        obs = robot.get_observation()',
        '    except Exception as e:',
        '        print(f"[SAFE] Cannot read current position: {e}, disabling torque directly.")',
        '        _disable_torque(robot)',
        '        return',
        '',
        '    start_pos = {}',
        '    for key in SAFE_POSITION:',
        '        start_pos[key] = obs.get(key, 0.0)',
        '',
        '    steps = int(duration * fps)',
        '    vel_keys = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}',
        '',
        '    for step in range(steps + 1):',
        '        t = step / steps',
        '        alpha = 0.5 * (1.0 - math.cos(t * math.pi))',
        '',
        '        action = {}',
        '        for key in SAFE_POSITION:',
        '            action[key] = start_pos[key] + alpha * (SAFE_POSITION[key] - start_pos[key])',
        '        action.update(vel_keys)',
        '',
        '        try:',
        '            robot.send_action(action)',
        '        except Exception as e:',
        '            print(f"[SAFE] Send error at step {step}: {e}")',
        '            break',
        '',
        '        time.sleep(1.0 / fps)',
        '',
        '    print("[SAFE] Safe position reached.")',
        '    _disable_torque(robot)',
        '',
        '',
        'def _disable_torque(robot):',
        '    """Disable torque on all motors."""',
        '    try:',
        '        robot.bus1.disable_torque()',
        '        print("[SAFE] Left arm torque disabled")',
        '    except Exception:',
        '        pass',
        '    try:',
        '        robot.bus2.disable_torque()',
        '        print("[SAFE] Right arm torque disabled")',
        '    except Exception:',
        '        pass',
        '',
    ]

    with open(SAFE_POS_FILE, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nSaved to: {SAFE_POS_FILE}")
    print("This position will be used by teleop scripts on exit.")

    # Cleanup
    try:
        robot.bus1.disable_torque()
    except Exception:
        pass
    try:
        robot.bus2.disable_torque()
    except Exception:
        pass
    try:
        robot.disconnect()
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
