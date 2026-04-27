"""
XLeRobot Keyboard Teleoperation (no leader arms needed)

Controls:
  1-6      Select left arm joint
  7-0/-/=  Select right arm joint (7=shoulder_pan ... 0=wrist_roll, -=gripper... mapped below)
  Shift+1-6  Select right arm joint
  UP/DOWN  Move selected joint
  +/-      Increase/decrease step size
  SPACE    Toggle torque on selected joint (for manual positioning)
  R        Reset all joints to current position (stop movement)
  Q/ESC    Quit
"""

import pygame
import time
import sys
import numpy as np

from lerobot.robots.xlerobot import XLerobot, XLerobotConfig

# Joint names in order
LEFT_JOINTS = [
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
    "left_arm_gripper",
]

RIGHT_JOINTS = [
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
]

ALL_JOINTS = LEFT_JOINTS + RIGHT_JOINTS

JOINT_LABELS = [
    "L1: Shoulder Pan",
    "L2: Shoulder Lift",
    "L3: Elbow Flex",
    "L4: Wrist Flex",
    "L5: Wrist Roll",
    "L6: Gripper",
    "R1: Shoulder Pan",
    "R2: Shoulder Lift",
    "R3: Elbow Flex",
    "R4: Wrist Flex",
    "R5: Wrist Roll",
    "R6: Gripper",
]


def main():
    # Connect robot
    print("Connecting to robot...")
    robot = XLerobot(XLerobotConfig())
    robot.connect()
    print("Connected! Reading initial positions...")

    # Read current positions as initial targets
    obs = robot.get_observation()
    targets = {}
    for j in ALL_JOINTS:
        key = f"{j}.pos"
        targets[key] = obs.get(key, 0.0)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((700, 520))
    pygame.display.set_caption("XLeRobot Keyboard Teleop")
    font = pygame.font.SysFont("Consolas", 20)
    font_small = pygame.font.SysFont("Consolas", 16)

    selected = 0  # Currently selected joint index
    step_size = 2.0  # Step size per keypress
    running = True
    loop_hz = 30

    print("Ready! Use arrow keys to move joints.")

    while running:
        t0 = time.time()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                shift = mods & pygame.KMOD_SHIFT

                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                # Select joint: 1-6 for left arm, Shift+1-6 for right arm
                elif event.key == pygame.K_1:
                    selected = 6 if shift else 0
                elif event.key == pygame.K_2:
                    selected = 7 if shift else 1
                elif event.key == pygame.K_3:
                    selected = 8 if shift else 2
                elif event.key == pygame.K_4:
                    selected = 9 if shift else 3
                elif event.key == pygame.K_5:
                    selected = 10 if shift else 4
                elif event.key == pygame.K_6:
                    selected = 11 if shift else 5

                # Step size
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                    step_size = min(20.0, step_size + 0.5)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    step_size = max(0.5, step_size - 0.5)

                # Reset targets to current positions
                elif event.key == pygame.K_r:
                    obs = robot.get_observation()
                    for j in ALL_JOINTS:
                        key = f"{j}.pos"
                        targets[key] = obs.get(key, 0.0)

        # Continuous movement while key held
        keys = pygame.key.get_pressed()
        joint_key = f"{ALL_JOINTS[selected]}.pos"
        if keys[pygame.K_UP]:
            targets[joint_key] += step_size * 0.5
        if keys[pygame.K_DOWN]:
            targets[joint_key] -= step_size * 0.5

        # Clamp targets to valid range
        for k in targets:
            targets[k] = max(-100.0, min(100.0, targets[k]))

        # Send action
        action = {**targets, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        try:
            robot.send_action(action)
        except Exception as e:
            print(f"Send error: {e}")

        # Read observation
        try:
            obs = robot.get_observation()
        except Exception as e:
            print(f"Read error: {e}")
            obs = targets.copy()

        # Draw UI
        screen.fill((20, 20, 30))

        # Title
        surf = font.render("XLeRobot Keyboard Teleop", True, (100, 200, 255))
        screen.blit(surf, (15, 10))

        # Column headers
        surf = font_small.render("Joint                Target   Current", True, (180, 180, 180))
        screen.blit(surf, (15, 45))

        # Joint list
        for i, (jname, label) in enumerate(zip(ALL_JOINTS, JOINT_LABELS)):
            key = f"{jname}.pos"
            target_val = targets.get(key, 0.0)
            current_val = obs.get(key, 0.0)

            # Highlight selected
            if i == selected:
                pygame.draw.rect(screen, (40, 40, 80), (10, 68 + i * 28, 680, 26))
                color = (255, 255, 100)
                marker = ">"
            elif i < 6:
                color = (200, 200, 200)
                marker = " "
            else:
                color = (180, 220, 180)
                marker = " "

            text = f"{marker} {label:<22} {target_val:>7.1f}   {current_val:>7.1f}"
            surf = font_small.render(text, True, color)
            screen.blit(surf, (15, 70 + i * 28))

        # Separator
        y_base = 70 + 12 * 28 + 10

        # Controls info
        controls = [
            f"Step size: {step_size:.1f}  (+/- to change)",
            "1-6: select left joint | Shift+1-6: select right joint",
            "UP/DOWN: move joint | R: reset | Q: quit",
        ]
        for i, text in enumerate(controls):
            surf = font_small.render(text, True, (150, 150, 180))
            screen.blit(surf, (15, y_base + i * 22))

        pygame.display.flip()

        # Rate limit
        dt = time.time() - t0
        if dt < 1.0 / loop_hz:
            time.sleep(1.0 / loop_hz - dt)

    # Cleanup
    pygame.quit()
    try:
        robot.disconnect()
    except:
        pass
    print("Disconnected.")


if __name__ == "__main__":
    main()
