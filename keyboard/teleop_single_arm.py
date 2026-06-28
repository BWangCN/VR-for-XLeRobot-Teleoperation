"""
Single-arm keyboard teleop for XLErobot SO101 arm.
Uses pygame window for input — click the window first, then use keys.

Controls:
  1-6     : select joint
  UP/DOWN : hold to move continuously
  +/-     : increase/decrease step size
  R       : reset to current physical position
  Q       : quit
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

WIN_W, WIN_H = 520, 320
BG_COLOR   = (20, 20, 30)
HL_COLOR   = (60, 120, 220)
TEXT_COLOR = (220, 220, 220)
DIM_COLOR  = (120, 120, 140)
GOOD_COLOR = (80, 200, 120)
WARN_COLOR = (220, 160, 60)


def get_port():
    port = input("Enter COM port for the arm (e.g. COM6 or COM7): ").strip()
    if not port:
        port = "COM6"
    return port


def main():
    port = get_port()

    print(f"Connecting to arm on {port}...")
    config = SO101FollowerConfig(port=port, id="arm")
    robot = SO101Follower(config)

    try:
        robot.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    # Seed targets from actual current motor positions so arm doesn't jump on start
    try:
        obs = robot.get_observation()
        targets = [obs.get(k, 0.0) for k in OBS_KEYS]
        print(f"Starting positions: {targets}")
    except Exception:
        targets = [0.0] * 6

    print("Connected! Opening control window...")

    pygame.init()
    # Enable key repeat: 200ms delay before repeat starts, then every 30ms
    # This is what makes holding a key feel smooth
    pygame.key.set_repeat(200, 30)

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("XLErobot Single-Arm Teleop  |  Click here first!")
    font_big   = pygame.font.SysFont("Consolas", 18, bold=True)
    font_small = pygame.font.SysFont("Consolas", 15)

    selected_joint = 0
    step_size = 2.0  # smaller step for smoother hold-to-move feel
    loop_hz   = 20
    current   = list(targets)

    running = True
    while running:
        t0 = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                # joint selection
                for i, k in enumerate([pygame.K_1, pygame.K_2, pygame.K_3,
                                        pygame.K_4, pygame.K_5, pygame.K_6]):
                    if event.key == k:
                        selected_joint = i

                # move joint — fires repeatedly while held due to set_repeat
                if event.key == pygame.K_UP:
                    targets[selected_joint] += step_size
                if event.key == pygame.K_DOWN:
                    targets[selected_joint] -= step_size

                # step size adjustment
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    step_size = min(step_size + 0.5, 30.0)
                if event.key == pygame.K_MINUS:
                    step_size = max(step_size - 0.5, 0.5)

                # reset targets to where the arm physically is right now
                if event.key == pygame.K_r:
                    targets = list(current)

        # send commands and read back
        connected = False
        try:
            action = {k: targets[i] for i, k in enumerate(ACTION_KEYS)}
            robot.send_action(action)
            obs = robot.get_observation()
            current = [obs.get(k, 0.0) for k in OBS_KEYS]
            connected = True
        except Exception as e:
            print(f"Send error: {e}")

        # draw UI
        screen.fill(BG_COLOR)

        status_color = GOOD_COLOR if connected else WARN_COLOR
        status_text  = f"  CONNECTED  port={port}" if connected else "  DISCONNECTED"
        screen.blit(font_big.render(status_text, True, status_color), (15, 10))

        header = f"  {'Joint':<20} {'Target':>7}   {'Current':>7}"
        screen.blit(font_small.render(header, True, DIM_COLOR), (15, 45))

        for i, label in enumerate(JOINT_NAMES):
            marker = "  "
            if i == selected_joint:
                pygame.draw.rect(screen, HL_COLOR, (8, 65 + i*28, WIN_W-16, 24), border_radius=4)
                marker = ">>"
            text = f"{marker} {label:<20} {targets[i]:>7.1f}   {current[i]:>7.1f}"
            surf = font_small.render(text, True, TEXT_COLOR)
            screen.blit(surf, (15, 70 + i * 28))

        y_base = 70 + 6 * 28 + 10
        controls = [
            f"Step size: {step_size:.1f} deg  (+/- to change)",
            "1-6: select joint  |  HOLD UP/DOWN: move  |  R: reset  |  Q: quit",
        ]
        for i, text in enumerate(controls):
            surf = font_small.render(text, True, (150, 150, 180))
            screen.blit(surf, (15, y_base + i * 22))

        pygame.display.flip()

        dt = time.time() - t0
        if dt < 1.0 / loop_hz:
            time.sleep(1.0 / loop_hz - dt)

    pygame.quit()
    try:
        robot.disconnect()
    except:
        pass
    print("Disconnected.")


if __name__ == "__main__":
    main()
