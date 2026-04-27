"""
PICO 4 Ultra VR  →  Real XLeRobot (Arms Only)

Controls:
  Left arm  (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
  Right arm (same 6 joints)

Does NOT control: base, head

Usage:
    conda activate xlerobot
    python demo_pico_vr_real.py --port1 COM4 --port2 COM3
"""

import math
import os
import time
import numpy as np
import pygame

from dataclasses import dataclass
from typing import Optional

from pico_vr_bridge import PicoVRBridge

# ═══════════════════════════════════════════════════════════════════════════
# IK  (same geometry as SO101Kinematics, returns DEGREES)
# ═══════════════════════════════════════════════════════════════════════════

def ik_degrees(x, y, l1=0.1159, l2=0.1350):
    """2-link planar IK → (shoulder_lift_deg, elbow_flex_deg) in motor-degree convention."""
    t1_off = math.atan2(0.028, 0.11257)
    t2_off = math.atan2(0.0052, 0.1349) + t1_off
    r = math.sqrt(x*x + y*y)
    r_max, r_min = l1 + l2, abs(l1 - l2)
    if r > r_max:
        s = r_max / r; x *= s; y *= s; r = r_max
    if r < r_min:
        if r > 1e-6:
            s = r_min / r; x *= s; y *= s
        else:
            x = r_min; y = 0.0   # degenerate: default to forward
    c2 = max(-1.0, min(1.0, -(r*r - l1*l1 - l2*l2) / (2*l1*l2)))
    t2 = math.pi - math.acos(c2)
    t1 = math.atan2(y, x) + math.atan2(l2*math.sin(t2), l1 + l2*math.cos(t2))
    j2 = max(-0.1, min(3.50, t1 + t1_off))       # sl ∈ [-110.5°, 95.7°]
    j3 = max(-0.2, min(3.32, t2 + t2_off))       # ef ∈ [-101.5°, 100.3°]
    # Convert to motor-degree convention (same as SO101Kinematics)
    return 90.0 - math.degrees(j2), math.degrees(j3) - 90.0


def fk_from_degrees(sl_deg, ef_deg, l1=0.1159, l2=0.1350):
    """Forward kinematics: (shoulder_lift_deg, elbow_flex_deg) → (reach, height)."""
    t1_off = math.atan2(0.028, 0.11257)
    t2_off = math.atan2(0.0052, 0.1349) + t1_off
    # Reverse SO101 degree convention → IK internal angles
    j2 = math.radians(90.0 - sl_deg)
    j3 = math.radians(ef_deg + 90.0)
    t1 = j2 - t1_off
    t2 = j3 - t2_off
    # 2-link FK
    r = math.sqrt(l1**2 + l2**2 + 2*l1*l2*math.cos(t2))
    alpha = math.atan2(l2*math.sin(t2), l1 + l2*math.cos(t2))
    theta = t1 - alpha
    return r * math.cos(theta), r * math.sin(theta)


# ═══════════════════════════════════════════════════════════════════════════
# Robot wrapper  (thin layer over lerobot FeetechMotorsBus)
# ═══════════════════════════════════════════════════════════════════════════

class DualArmRobot:
    """Minimal interface: connect, read, write – arms only."""

    JOINTS = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(self, port1: str, port2: str):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        norm = MotorNormMode.DEGREES
        grip_norm = MotorNormMode.RANGE_0_100

        self.bus_left = FeetechMotorsBus(
            port=port1,
            motors={
                "left_arm_shoulder_pan":  Motor(1, "sts3215", norm),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm),
                "left_arm_elbow_flex":    Motor(3, "sts3215", norm),
                "left_arm_wrist_flex":    Motor(4, "sts3215", norm),
                "left_arm_wrist_roll":    Motor(5, "sts3215", norm),
                "left_arm_gripper":       Motor(6, "sts3215", grip_norm),
            },
        )
        self.bus_right = FeetechMotorsBus(
            port=port2,
            motors={
                "right_arm_shoulder_pan":  Motor(1, "sts3215", norm),
                "right_arm_shoulder_lift": Motor(2, "sts3215", norm),
                "right_arm_elbow_flex":    Motor(3, "sts3215", norm),
                "right_arm_wrist_flex":    Motor(4, "sts3215", norm),
                "right_arm_wrist_roll":    Motor(5, "sts3215", norm),
                "right_arm_gripper":       Motor(6, "sts3215", grip_norm),
            },
        )
        self._left_names = [f"left_arm_{j}" for j in self.JOINTS]
        self._right_names = [f"right_arm_{j}" for j in self.JOINTS]

    def connect(self):
        self.bus_left.connect()
        self.bus_right.connect()

        # Load calibration from existing XLeRobot calibration file
        import json
        from lerobot.motors.motors_bus import MotorCalibration
        calib_path = os.path.join(
            os.path.expanduser("~"),
            ".cache", "huggingface", "lerobot", "calibration",
            "robots", "xlerobot", "None.json"
        )
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                raw = json.load(f)
            left_calib = {}
            right_calib = {}
            for name, data in raw.items():
                mc = MotorCalibration(**data)
                if name.startswith("left_arm_"):
                    left_calib[name] = mc
                elif name.startswith("right_arm_"):
                    right_calib[name] = mc
            self.bus_left.calibration = left_calib
            self.bus_right.calibration = right_calib
            print(f"[Robot] Calibration loaded from {calib_path}")
        else:
            print(f"[Robot] WARNING: calibration file not found: {calib_path}")

        print("[Robot] Both buses connected")

    def disable_torque(self):
        """Disable torque on all motors so arms go limp."""
        try:
            self.bus_left.disable_torque()
            self.bus_right.disable_torque()
            print("[Robot] Torque disabled")
        except Exception as e:
            print(f"[Robot] Torque disable error: {e}")

    def disconnect(self):
        self.disable_torque()
        self.bus_left.disconnect()
        self.bus_right.disconnect()
        print("[Robot] Disconnected")

    def read(self):
        """Return dict  {'left_arm_shoulder_pan': deg, ...}"""
        left = self.bus_left.sync_read("Present_Position", self._left_names)
        right = self.bus_right.sync_read("Present_Position", self._right_names)
        return {**left, **right}

    def write(self, goals: dict):
        """goals: {'left_arm_shoulder_pan': deg, 'right_arm_gripper': pct, ...}"""
        left = {k: v for k, v in goals.items() if k.startswith("left_arm_")}
        right = {k: v for k, v in goals.items() if k.startswith("right_arm_")}
        if left:
            self.bus_left.sync_write("Goal_Position", left)
        if right:
            self.bus_right.sync_write("Goal_Position", right)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Args:
    port1: str = "COM4"      # left arm serial port
    port2: str = "COM3"      # right arm serial port
    pico_port: int = 9876
    vr_scale: float = 1.0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port1", default="COM4", help="Left arm serial port")
    parser.add_argument("--port2", default="COM3", help="Right arm serial port")
    parser.add_argument("--pico-port", type=int, default=9876)
    args = parser.parse_args()

    # ── PICO bridge ──────────────────────────────────────────────────────
    bridge = PicoVRBridge(port=args.pico_port)
    bridge.start()
    print(f"\n[VR] Listening on port {args.pico_port} …")

    # ── Robot ─────────────────────────────────────────────────────────────
    robot = DualArmRobot(port1=args.port1, port2=args.port2)
    robot.connect()

    # ── Pygame ────────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("XLeRobot PICO VR → Real Robot")
    font     = pygame.font.SysFont(None, 22)
    font_big = pygame.font.SysFont(None, 28)

    # ── Control state ─────────────────────────────────────────────────────
    TIP = 0.108
    # EE_HOME will be computed from recorded home pose (or default)
    EE_HOME_DEFAULT = np.array([0.162, 0.0, 0.118])
    EE_HOME = EE_HOME_DEFAULT.copy()
    pitch_l = 0.0;  pitch_r = 0.0

    left_origin  = None
    right_origin = None

    VR_SCALE_LAT  = 1.2
    VR_SCALE_VERT = 1.0
    VR_SCALE_FWD  = 0.8

    GRIP_MIN   = 0.0       # fully closed (0 %)
    GRIP_MAX   = 100.0     # fully open  (100 %)
    GRIP_SPEED = 2.0       # per frame

    # Target positions (degrees, except gripper = 0-100 %)
    targets = {
        "left_arm_shoulder_pan": 0.0,
        "left_arm_shoulder_lift": 0.0,
        "left_arm_elbow_flex": 0.0,
        "left_arm_wrist_flex": 0.0,
        "left_arm_wrist_roll": 0.0,
        "left_arm_gripper": 0.0,
        "right_arm_shoulder_pan": 0.0,
        "right_arm_shoulder_lift": 0.0,
        "right_arm_elbow_flex": 0.0,
        "right_arm_wrist_flex": 0.0,
        "right_arm_wrist_roll": 0.0,
        "right_arm_gripper": 0.0,
    }

    # P-gain for each joint
    KP = 0.5

    # Read current positions as initial targets (avoid jump to zeros on startup)
    try:
        init_pos = robot.read()
        for key in targets:
            if key in init_pos:
                targets[key] = init_pos[key]
        print(f"[Robot] Initial positions read OK")
    except Exception as e:
        print(f"[Robot] WARNING: could not read initial positions: {e}")

    # ── Safety limits ──────────────────────────────────────────────────
    # Absolute joint limits (degrees for arm joints, percent for gripper)
    JOINT_LIMITS = {
        "shoulder_pan":  (-90.0, 90.0),
        "shoulder_lift": (-110.0, 95.0),
        "elbow_flex":    (-100.0, 100.0),
        "wrist_flex":    (-100.0, 100.0),
        "wrist_roll":    (-100.0, 100.0),
        "gripper":       (0.0, 100.0),
    }
    MAX_STEP_DEG  = 6.0   # max degrees per control step (~300°/s at 50Hz)
    MAX_STEP_GRIP = 8.0   # max percent per step for gripper

    def safe_p_step(obs, tgts, kp):
        """P-control with joint limits and step-size clamping. Returns (goals, max_err)."""
        goals = {}
        max_err = 0.0
        for key, tgt in tgts.items():
            joint_name = key.split("_", 2)[2]  # e.g. "shoulder_pan"
            lo, hi = JOINT_LIMITS.get(joint_name, (-180.0, 180.0))
            tgt = max(lo, min(hi, tgt))         # clamp target to safe range
            cur = obs.get(key, tgt)              # if read fails, stay put (not 0!)
            err = abs(tgt - cur)
            if err > max_err:
                max_err = err
            delta = kp * (tgt - cur)
            max_d = MAX_STEP_GRIP if joint_name == "gripper" else MAX_STEP_DEG
            delta = max(-max_d, min(max_d, delta))  # clamp step size
            goals[key] = cur + delta
        return goals, max_err

    # ── State machine ─────────────────────────────────────────────────────
    STATE    = "WAITING"
    phase_t0 = 0.0
    PREP_DUR = 3.0
    CALIB_DUR = 5.0
    calib_buf_l = []
    calib_buf_r = []
    step = 0
    WARM = 20

    # ── helpers ───────────────────────────────────────────────────────────
    _y = 0
    def draw(txt, color=(200,200,200), big=False):
        nonlocal _y
        f = font_big if big else font
        screen.blit(f.render(str(txt), True, color), (15, _y))
        _y += 26 if big else 21
    def gap():
        nonlocal _y; _y += 8

    # Load saved home pose if available, and compute EE_HOME from it
    home_pose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "home_pose.json")
    HOME_POSE = None
    if os.path.exists(home_pose_path):
        import json as _json
        with open(home_pose_path) as _f:
            HOME_POSE = _json.load(_f)
        print(f"[Robot] Home pose loaded from {home_pose_path}")

        # Compute EE_HOME from home pose joint angles (average of left/right)
        sl_l = HOME_POSE.get("left_arm_shoulder_lift", 0.0)
        ef_l = HOME_POSE.get("left_arm_elbow_flex", 0.0)
        sl_r = HOME_POSE.get("right_arm_shoulder_lift", 0.0)
        ef_r = HOME_POSE.get("right_arm_elbow_flex", 0.0)
        reach_l, ht_l = fk_from_degrees(sl_l, ef_l)
        reach_r, ht_r = fk_from_degrees(sl_r, ef_r)
        EE_HOME[0] = (reach_l + reach_r) / 2.0
        EE_HOME[2] = (ht_l + ht_r) / 2.0
        print(f"[Robot] EE_HOME from pose: reach={EE_HOME[0]:.4f}m  height={EE_HOME[2]:.4f}m")
        print(f"[Robot] Home joints: L sl={sl_l:.1f}° ef={ef_l:.1f}° | R sl={sl_r:.1f}° ef={ef_r:.1f}°")
    else:
        print("[Robot] No home_pose.json found — using default EE_HOME")

    def set_home():
        """Set arm targets to saved home pose, or all-zeros fallback."""
        if HOME_POSE:
            for key in targets:
                if key in HOME_POSE:
                    targets[key] = HOME_POSE[key]
        else:
            for key in targets:
                targets[key] = 0.0

    def enter_prep():
        nonlocal STATE, phase_t0, left_origin, right_origin, pitch_l, pitch_r
        STATE = "PREP"; phase_t0 = time.time()
        left_origin = None; right_origin = None
        pitch_l = pitch_r = 0.0
        set_home()
        bridge.reset_origins()
        print("[VR] Arms homing — position your hands …")

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════
    try:
        while True:

            # ── events ──────────────────────────────────────────────────
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_r:
                        enter_prep()
                    elif ev.key == pygame.K_x:
                        enter_prep()
                    elif ev.key == pygame.K_q:
                        raise KeyboardInterrupt

            # ── read VR ─────────────────────────────────────────────────
            raw = bridge.get_latest_goal_nowait()
            lg = raw.get("left")    if isinstance(raw, dict) else None
            rg = raw.get("right")   if isinstance(raw, dict) else None
            has = (raw.get("has_left", False) or raw.get("has_right", False)) \
                  if isinstance(raw, dict) else False

            elapsed = time.time() - phase_t0

            # ════════════════════════════════════════════════════════════
            #  WAITING
            # ════════════════════════════════════════════════════════════
            if STATE == "WAITING":
                if has and step >= WARM:
                    enter_prep()

            # ════════════════════════════════════════════════════════════
            #  PREP
            # ════════════════════════════════════════════════════════════
            elif STATE == "PREP":
                set_home()
                if elapsed >= PREP_DUR:
                    STATE = "CALIB"; phase_t0 = time.time()
                    calib_buf_l.clear(); calib_buf_r.clear()
                    bridge.reset_origins()
                    print("[VR] Calibrating — hold still for 5 s …")

            # ════════════════════════════════════════════════════════════
            #  CALIB
            # ════════════════════════════════════════════════════════════
            elif STATE == "CALIB":
                if lg and lg.target_position is not None:
                    calib_buf_l.append(lg.target_position.copy())
                if rg and rg.target_position is not None:
                    calib_buf_r.append(rg.target_position.copy())
                if elapsed >= CALIB_DUR:
                    if calib_buf_l:
                        left_origin = np.mean(calib_buf_l, axis=0)
                        print(f"[VR] Left  origin: {left_origin}")
                    if calib_buf_r:
                        right_origin = np.mean(calib_buf_r, axis=0)
                        print(f"[VR] Right origin: {right_origin}")
                    bridge.reset_origins()
                    # Re-read current motor positions so targets start from
                    # the actual physical state (avoids wrist_roll getting
                    # stuck at the HOME_POSE value)
                    try:
                        cur = robot.read()
                        for key in targets:
                            if key in cur:
                                targets[key] = cur[key]
                    except Exception:
                        pass
                    STATE = "ACTIVE"
                    print("[VR] ACTIVE — R = re-calibrate | Q = quit")

            # ════════════════════════════════════════════════════════════
            #  ACTIVE
            # ════════════════════════════════════════════════════════════
            elif STATE == "ACTIVE" and step >= WARM:

                # ─── ROBOT LEFT ARM ← USER LEFT HAND (lg) ─────────────
                if (lg and lg.target_position is not None
                        and left_origin is not None):
                    d = lg.target_position - left_origin

                    ee_reach = EE_HOME[0] - d[2] * VR_SCALE_FWD
                    ee_lat   = d[0] * VR_SCALE_LAT
                    ee_ht    = EE_HOME[2] + d[1] * VR_SCALE_VERT

                    # Pan uses |reach| so left/right never flips with reach sign
                    pan_rad = math.atan2(ee_lat, max(abs(ee_reach), 0.01))
                    # Signed reach for IK (positive=forward, negative=backward)
                    if abs(ee_reach) >= 0.01:
                        sagittal_r = math.copysign(
                            math.sqrt(ee_reach**2 + ee_lat**2), ee_reach)
                    else:
                        sagittal_r = ee_reach
                    sagittal_r = np.clip(sagittal_r, -0.20, 0.24)
                    ee_ht      = np.clip(ee_ht, -0.20, 0.25)

                    if lg.wrist_flex_deg is not None:
                        pitch_l = lg.wrist_flex_deg * 0.5
                    if lg.wrist_roll_deg is not None:
                        targets["left_arm_wrist_roll"] = lg.wrist_roll_deg

                    cy = ee_ht + TIP * math.sin(math.radians(pitch_l))
                    sl, ef = ik_degrees(sagittal_r, cy)

                    # Debug: print VR delta and IK output every ~1s
                    if step % 50 == 0:
                        print(f"[IK-L] d=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) "
                              f"reach={sagittal_r:+.3f} ht={ee_ht:.3f} "
                              f"→ sl={sl:.1f}° ef={ef:.1f}°")

                    targets["left_arm_shoulder_pan"]  = math.degrees(pan_rad)
                    targets["left_arm_shoulder_lift"] = sl
                    targets["left_arm_elbow_flex"]    = ef
                    targets["left_arm_wrist_flex"]    = -(sl + ef) + pitch_l

                    # Gripper: A = open, B = close
                    meta_l = lg.metadata or {}
                    if meta_l.get("primaryButton", False):
                        targets["left_arm_gripper"] = min(
                            GRIP_MAX, targets["left_arm_gripper"] + GRIP_SPEED)
                    elif meta_l.get("secondaryButton", False):
                        targets["left_arm_gripper"] = max(
                            GRIP_MIN, targets["left_arm_gripper"] - GRIP_SPEED)

                # ─── ROBOT RIGHT ARM ← USER RIGHT HAND (rg) ───────────
                if (rg and rg.target_position is not None
                        and right_origin is not None):
                    d = rg.target_position - right_origin

                    ee_reach = EE_HOME[0] - d[2] * VR_SCALE_FWD
                    ee_lat   = d[0] * VR_SCALE_LAT
                    ee_ht    = EE_HOME[2] + d[1] * VR_SCALE_VERT

                    # Pan uses |reach| so left/right never flips with reach sign
                    pan_rad = math.atan2(ee_lat, max(abs(ee_reach), 0.01))
                    # Signed reach for IK (positive=forward, negative=backward)
                    if abs(ee_reach) >= 0.01:
                        sagittal_r = math.copysign(
                            math.sqrt(ee_reach**2 + ee_lat**2), ee_reach)
                    else:
                        sagittal_r = ee_reach
                    sagittal_r = np.clip(sagittal_r, -0.20, 0.24)
                    ee_ht      = np.clip(ee_ht, -0.20, 0.25)

                    if rg.wrist_flex_deg is not None:
                        pitch_r = rg.wrist_flex_deg * 0.5
                    if rg.wrist_roll_deg is not None:
                        targets["right_arm_wrist_roll"] = rg.wrist_roll_deg

                    cy = ee_ht + TIP * math.sin(math.radians(pitch_r))
                    sl, ef = ik_degrees(sagittal_r, cy)

                    targets["right_arm_shoulder_pan"]  = math.degrees(pan_rad)
                    targets["right_arm_shoulder_lift"] = sl
                    targets["right_arm_elbow_flex"]    = ef
                    targets["right_arm_wrist_flex"]    = -(sl + ef) + pitch_r

                    meta_r = rg.metadata or {}
                    if meta_r.get("primaryButton", False):
                        targets["right_arm_gripper"] = min(
                            GRIP_MAX, targets["right_arm_gripper"] + GRIP_SPEED)
                    elif meta_r.get("secondaryButton", False):
                        targets["right_arm_gripper"] = max(
                            GRIP_MIN, targets["right_arm_gripper"] - GRIP_SPEED)

            # ════════════════════════════════════════════════════════════
            #  SEND TO ROBOT (P-control with safety)
            # ════════════════════════════════════════════════════════════
            if step >= WARM and STATE != "WAITING":
                try:
                    obs = robot.read()
                    goals, _ = safe_p_step(obs, targets, KP)
                    robot.write(goals)
                except Exception as e:
                    print(f"[Robot] Error: {e}")

            # ════════════════════════════════════════════════════════════
            #  PYGAME UI
            # ════════════════════════════════════════════════════════════
            screen.fill((20, 20, 30))
            _y = 10

            if STATE == "WAITING":
                draw("PICO VR → Real XLeRobot (Arms)", (100,200,255), big=True); gap()
                draw(f"Waiting for PICO on port {args.pico_port} …", (255,100,100))
                draw(f"Step: {step}", (120,120,120))

            elif STATE == "PREP":
                remain = max(0, PREP_DUR - elapsed)
                draw("=== HOMING ===", (255,220,80), big=True); gap()
                draw("Arms returning to home …", (220,220,220))
                draw(f"Calibration in {remain:.1f} s", (100,255,100))

            elif STATE == "CALIB":
                remain = max(0, CALIB_DUR - elapsed)
                pct = min(100, int(100 * elapsed / CALIB_DUR))
                bar = "#" * (pct // 5)
                draw("=== CALIBRATING ===", (80,200,255), big=True); gap()
                draw("HOLD STILL", (255,255,100))
                draw(f"[{bar:<20}] {pct}%   {remain:.1f} s", (100,255,100))

            elif STATE == "ACTIVE":
                draw("ACTIVE", (80,255,120), big=True); gap()
                try:
                    obs = robot.read()
                    for prefix, label in [("left_arm", "L"), ("right_arm", "R")]:
                        sp = obs.get(f"{prefix}_shoulder_pan", 0)
                        sl = obs.get(f"{prefix}_shoulder_lift", 0)
                        ef = obs.get(f"{prefix}_elbow_flex", 0)
                        gr = obs.get(f"{prefix}_gripper", 0)
                        draw(f"{label}: pan={sp:+.1f} lift={sl:.1f} elbow={ef:.1f} grip={gr:.0f}",
                             (200,200,255))
                except Exception:
                    draw("(read error)", (255,100,100))
                gap()
                draw("R = re-calibrate | Q = quit", (130,130,160))

            pygame.display.flip()

            step += 1
            time.sleep(0.02)   # ~50 Hz

    except KeyboardInterrupt:
        print("\n[VR] Shutting down …")
    finally:
        # ── Return to home position before disconnecting ──────────────
        print("[VR] Returning arms to home position …")
        set_home()
        HOME_RETURN_SEC = 5.0
        HOME_KP = 0.2   # gentler gain for homing
        t0 = time.time()
        try:
            while time.time() - t0 < HOME_RETURN_SEC:
                obs = robot.read()
                goals, max_err = safe_p_step(obs, targets, HOME_KP)
                robot.write(goals)
                if max_err < 2.0:
                    print("[VR] Arms reached home position.")
                    break
                time.sleep(0.02)
            else:
                print("[VR] Home return timeout — arms may not be fully homed.")
        except Exception as e:
            print(f"[VR] Home return error: {e}")

        pygame.quit()
        bridge.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
