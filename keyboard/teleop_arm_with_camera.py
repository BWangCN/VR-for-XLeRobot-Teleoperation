"""
Single-arm keyboard teleop WITH live camera feed, all in one pygame window.

Left side  = camera feed
Right side = joint control panel

Controls (click the window first):
  1-6     : select joint
  UP/DOWN : hold to move selected joint
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
    import cv2
except ImportError:
    print("OpenCV not found. Run: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy not found. Run: pip install numpy")
    sys.exit(1)

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ImportError:
    print("Could not import SO101Follower. Make sure lerobot[feetech] is installed.")
    sys.exit(1)


JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
OBS_KEYS    = [f"{j}.pos" for j in JOINT_NAMES]
ACTION_KEYS = [f"{j}.pos" for j in JOINT_NAMES]

# Layout
CAM_W, CAM_H = 480, 360
PANEL_W      = 360
WIN_W        = CAM_W + PANEL_W
WIN_H        = max(CAM_H, 380)

BG_COLOR   = (20, 20, 30)
HL_COLOR   = (60, 120, 220)
TEXT_COLOR = (220, 220, 220)
DIM_COLOR  = (120, 120, 140)
GOOD_COLOR = (80, 200, 120)
WARN_COLOR = (220, 160, 60)


def get_inputs():
    port = input("Enter COM port for the arm (e.g. COM7): ").strip() or "COM7"
    cam  = input("Enter camera index (e.g. 1 for arm cam): ").strip() or "1"
    try:
        cam = int(cam)
    except ValueError:
        cam = 1
    return port, cam


def main():
    port, cam_index = get_inputs()

    # connect arm
    print(f"Connecting to arm on {port}...")
    config = SO101FollowerConfig(port=port, id="arm")
    robot = SO101Follower(config)
    try:
        robot.connect()
    except Exception as e:
        print(f"Failed to connect to arm: {e}")
        sys.exit(1)

    try:
        obs = robot.get_observation()
        targets = [obs.get(k, 0.0) for k in OBS_KEYS]
    except Exception:
        targets = [0.0] * 6

    # open camera (DSHOW backend for Windows reliability)
    print(f"Opening camera {cam_index}...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cam_ok = cap.isOpened()
    if not cam_ok:
        print(f"Warning: could not open camera {cam_index}. Continuing without video.")

    print("Connected! Opening control window...")

    pygame.init()
    pygame.key.set_repeat(200, 30)
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("XLErobot Arm + Camera  |  Click here first!")
    font_big   = pygame.font.SysFont("Consolas", 17, bold=True)
    font_small = pygame.font.SysFont("Consolas", 15)

    selected_joint = 0
    step_size = 2.0
    loop_hz = 30
    current = list(targets)

    running = True
    while running:
        t0 = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                for i, k in enumerate([pygame.K_1, pygame.K_2, pygame.K_3,
                                        pygame.K_4, pygame.K_5, pygame.K_6]):
                    if event.key == k:
                        selected_joint = i
                # UP increases, DOWN decreases (matches on-screen direction)
                if event.key == pygame.K_UP:
                    targets[selected_joint] -= step_size
                if event.key == pygame.K_DOWN:
                    targets[selected_joint] += step_size
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    step_size = min(step_size + 0.5, 30.0)
                if event.key == pygame.K_MINUS:
                    step_size = max(step_size - 0.5, 0.5)
                if event.key == pygame.K_r:
                    targets = list(current)

        # send to arm + read back
        connected = False
        try:
            action = {k: targets[i] for i, k in enumerate(ACTION_KEYS)}
            robot.send_action(action)
            obs = robot.get_observation()
            current = [obs.get(k, 0.0) for k in OBS_KEYS]
            connected = True
        except Exception:
            pass

        screen.fill(BG_COLOR)

        # camera feed on the left
        if cam_ok:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (CAM_W, CAM_H))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Build the surface from the raw frame (H, W, 3), then let pygame
                # handle the orientation. make_surface expects (W, H) so we transpose
                # ONCE here. No cv2.flip — the earlier flip+swap combo double-mirrored.
                surf = pygame.image.frombuffer(frame.tobytes(), (CAM_W, CAM_H), "RGB")
                screen.blit(surf, (0, 0))
            else:
                screen.blit(font_small.render("No camera frame", True, WARN_COLOR), (15, 15))
        else:
            screen.blit(font_small.render("Camera not available", True, WARN_COLOR), (15, 15))

        # control panel on the right
        px = CAM_W + 15
        status_color = GOOD_COLOR if connected else WARN_COLOR
        status_text  = f"CONNECTED {port}" if connected else "DISCONNECTED"
        screen.blit(font_big.render(status_text, True, status_color), (px, 12))

        header = f"{'Joint':<14}{'Tgt':>6}{'Cur':>7}"
        screen.blit(font_small.render(header, True, DIM_COLOR), (px, 42))

        for i, label in enumerate(JOINT_NAMES):
            marker = "  "
            if i == selected_joint:
                pygame.draw.rect(screen, HL_COLOR, (px - 6, 62 + i*30, PANEL_W - 18, 26), border_radius=4)
                marker = ">>"
            text = f"{marker}{label:<13}{targets[i]:>6.1f}{current[i]:>7.1f}"
            screen.blit(font_small.render(text, True, TEXT_COLOR), (px, 66 + i*30))

        y = 66 + 6*30 + 12
        for line in [
            f"Step: {step_size:.1f} deg  (+/-)",
            "1-6: joint   HOLD UP/DOWN: move",
            "R: reset     Q: quit",
        ]:
            screen.blit(font_small.render(line, True, (150,150,180)), (px, y))
            y += 24

        pygame.display.flip()

        dt = time.time() - t0
        if dt < 1.0 / loop_hz:
            time.sleep(1.0 / loop_hz - dt)

    if cam_ok:
        cap.release()
    pygame.quit()
    try:
        robot.disconnect()
    except:
        pass
    print("Disconnected.")


if __name__ == "__main__":
    main()
