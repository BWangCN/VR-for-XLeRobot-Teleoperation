"""
Dual-arm keyboard teleop for XLErobot (two SO101 arms).
Uses pygame window for input — click the window first, then use keys.

Controls:
  TAB       : switch between LEFT and RIGHT arm
  1-6       : select joint on active arm
  UP/DOWN   : hold to move selected joint
  +/-       : increase/decrease step size
  R         : reset active arm to current physical position
  Q         : quit
"""

import time
import sys

try:
    import pygame
except ImportError:
    print("pygame not found. Run: pip install pygame")
    sys.exit(1)

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ImportError:
    print("Could not import SO101Follower. Make sure lerobot[feetech] is installed.")
    sys.exit(1)


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
OBS_KEYS    = [f"{j}.pos" for j in JOINT_NAMES]
ACTION_KEYS = [f"{j}.pos" for j in JOINT_NAMES]

WIN_W, WIN_H = 820, 360
BG_COLOR     = (20, 20, 30)
HL_LEFT      = (60, 120, 220)   # blue for left arm
HL_RIGHT     = (180, 60, 220)   # purple for right arm
TEXT_COLOR   = (220, 220, 220)
DIM_COLOR    = (120, 120, 140)
GOOD_COLOR   = (80, 200, 120)
WARN_COLOR   = (220, 80, 80)
LABEL_LEFT   = (100, 160, 255)
LABEL_RIGHT  = (200, 100, 255)


def connect_arm(port, arm_id):
    print(f"Connecting to {arm_id} arm on {port}...")
    config = SO101FollowerConfig(port=port, id=arm_id)
    robot  = SO101Follower(config)
    robot.connect()
    obs     = robot.get_observation()
    targets = [obs.get(k, 0.0) for k in OBS_KEYS]
    print(f"  {arm_id} connected. Positions: {[f'{v:.1f}' for v in targets]}")
    return robot, targets


def main():
    left_port  = input("Enter COM port for LEFT arm  (default COM7): ").strip() or "COM7"
    right_port = input("Enter COM port for RIGHT arm (default COM6): ").strip() or "COM6"

    try:
        left_robot,  left_targets  = connect_arm(left_port,  "left")
        right_robot, right_targets = connect_arm(right_port, "right")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    print("Both arms connected! Opening control window...")

    pygame.init()
    pygame.key.set_repeat(200, 30)  # smooth hold-to-move

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("XLErobot Dual-Arm Teleop  |  Click here first!")
    font_title = pygame.font.SysFont("Consolas", 17, bold=True)
    font_hdr   = pygame.font.SysFont("Consolas", 14, bold=True)
    font_small = pygame.font.SysFont("Consolas", 14)

    active_arm     = 0        # 0 = left, 1 = right
    selected_joint = 0        # 0-5
    step_size      = 2.0      # degrees per key event
    loop_hz        = 20

    left_current  = list(left_targets)
    right_current = list(right_targets)
    left_ok  = True
    right_ok = True

    running = True
    while running:
        t0 = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                # switch active arm
                if event.key == pygame.K_TAB:
                    active_arm = 1 - active_arm
                    selected_joint = 0

                # joint selection
                for i, k in enumerate([pygame.K_1, pygame.K_2, pygame.K_3,
                                        pygame.K_4, pygame.K_5, pygame.K_6]):
                    if event.key == k:
                        selected_joint = i

                # move active arm's selected joint
                delta = 0
                if event.key == pygame.K_UP:
                    delta = +step_size
                if event.key == pygame.K_DOWN:
                    delta = -step_size

                if delta != 0:
                    if active_arm == 0:
                        left_targets[selected_joint] += delta
                    else:
                        right_targets[selected_joint] += delta

                # step size
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    step_size = min(step_size + 0.5, 30.0)
                if event.key == pygame.K_MINUS:
                    step_size = max(step_size - 0.5, 0.5)

                # reset active arm targets to its physical position
                if event.key == pygame.K_r:
                    if active_arm == 0:
                        left_targets = list(left_current)
                    else:
                        right_targets = list(right_current)

        # send to left arm
        try:
            action = {k: left_targets[i] for i, k in enumerate(ACTION_KEYS)}
            left_robot.send_action(action)
            obs = left_robot.get_observation()
            left_current = [obs.get(k, 0.0) for k in OBS_KEYS]
            left_ok = True
        except Exception as e:
            left_ok = False

        # send to right arm
        try:
            action = {k: right_targets[i] for i, k in enumerate(ACTION_KEYS)}
            right_robot.send_action(action)
            obs = right_robot.get_observation()
            right_current = [obs.get(k, 0.0) for k in OBS_KEYS]
            right_ok = True
        except Exception as e:
            right_ok = False

        # ── draw UI ─────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Title bar
        active_label = "LEFT" if active_arm == 0 else "RIGHT"
        active_color = LABEL_LEFT if active_arm == 0 else LABEL_RIGHT
        title = f"Active arm: "
        screen.blit(font_title.render(title, True, TEXT_COLOR), (15, 10))
        screen.blit(font_title.render(active_arm and "RIGHT" or "LEFT", True, active_color),
                    (15 + font_title.size(title)[0], 10))
        screen.blit(font_title.render("   (TAB to switch)", True, DIM_COLOR),
                    (15 + font_title.size(title + active_label)[0], 10))

        col_w = (WIN_W - 30) // 2

        # Column headers
        for col, (label, ok, color) in enumerate([
            ("LEFT  " + left_port,  left_ok,  LABEL_LEFT),
            ("RIGHT " + right_port, right_ok, LABEL_RIGHT),
        ]):
            x = 15 + col * (col_w + 10)
            status = "CONNECTED" if ok else "DISCONNECTED"
            sc = GOOD_COLOR if ok else WARN_COLOR
            screen.blit(font_hdr.render(label, True, color), (x, 38))
            screen.blit(font_hdr.render(status, True, sc), (x + col_w - 105, 38))

            hdr = f"  {'Joint':<17} {'Tgt':>6} {'Cur':>6}"
            screen.blit(font_small.render(hdr, True, DIM_COLOR), (x, 60))

        # Joint rows
        for i, label in enumerate(JOINT_NAMES):
            for col, (targets, current, ok) in enumerate([
                (left_targets,  left_current,  left_ok),
                (right_targets, right_current, right_ok),
            ]):
                x = 15 + col * (col_w + 10)
                y = 80 + i * 26

                is_active_col = (col == active_arm)
                is_selected   = is_active_col and (i == selected_joint)

                if is_selected:
                    hl = HL_LEFT if col == 0 else HL_RIGHT
                    pygame.draw.rect(screen, hl, (x, y - 2, col_w, 22), border_radius=3)
                    marker = ">>"
                elif is_active_col:
                    marker = "  "
                else:
                    marker = "  "

                tgt = targets[i] if ok else 0.0
                cur = current[i] if ok else 0.0
                text = f"{marker} {label:<17} {tgt:>6.1f} {cur:>6.1f}"
                surf = font_small.render(text, True, TEXT_COLOR)
                screen.blit(surf, (x + 4, y))

        # Controls footer
        footer_y = 80 + 6 * 26 + 10
        lines = [
            f"Step: {step_size:.1f} deg  (+/- to change)     TAB: switch arm     R: reset arm     Q: quit",
            "1-6: select joint                          HOLD UP/DOWN: move joint",
        ]
        for i, line in enumerate(lines):
            screen.blit(font_small.render(line, True, DIM_COLOR), (15, footer_y + i * 20))

        pygame.display.flip()

        dt = time.time() - t0
        if dt < 1.0 / loop_hz:
            time.sleep(1.0 / loop_hz - dt)

    pygame.quit()
    for robot in (left_robot, right_robot):
        try:
            robot.disconnect()
        except:
            pass
    print("Disconnected.")


if __name__ == "__main__":
    main()
