"""
XLeRobot ManiSkill Simulation — PICO 4 Ultra Full-Body VR Teleoperation

New features vs demo_pico_vr_sim.py:
  - T-pose calibration at startup (stand forward, arms at robot rest pose)
  - Arms track controller deltas in body frame (rotation-aware)
  - Headset acts as pelvis proxy → base forward/rotation
  - Head camera tracks head rotation *relative* to body via low-pass filter
  - Gripper: proportional trigger control (not just open/close)
  - Shoulder rotation from lateral hand movement

Usage:
    conda activate xlerobot_sim
    python demo_pico_vr_fullbody_sim.py
    python demo_pico_vr_fullbody_sim.py --shader="rt-fast"
"""

import gymnasium as gym
import numpy as np
import sapien
import pygame
import time
import math
import torch

from scipy.spatial.transform import Rotation as Rot

from mani_skill.envs.sapien_env import BaseEnv

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

from pico_vr_bridge import PicoVRBridge


# ═══════════════════════════════════════════════════════════════════════════
# CLI args
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "ReplicaCAD_SceneManipulation-v1"
    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = "xlerobot"
    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_delta_pos_dual_arm"
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "state"
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    render_mode: str = "human"
    shader: str = "default"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    reward_mode: Optional[str] = None
    record_dir: Optional[str] = None
    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    quiet: bool = False
    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    pico_port: int = 9876
    """TCP port for PICO VR bridge"""
    vr_scale: float = 1.0
    """Global scale factor for VR position"""


# ═══════════════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════════════

def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def extract_yaw(quat):
    """Yaw (Y-axis rotation) from [qx, qy, qz, qw]."""
    return Rot.from_quat(quat).as_euler('YXZ')[0]


def extract_pitch(quat):
    """Pitch (X-axis rotation) from [qx, qy, qz, qw]."""
    return Rot.from_quat(quat).as_euler('YXZ')[1]


def rotate_xz(dx, dz, angle):
    """Rotate a 2-D vector (dx, dz) by -angle around Y axis (body frame)."""
    c = math.cos(-angle)
    s = math.sin(-angle)
    return dx * c - dz * s, dx * s + dz * c


# ═══════════════════════════════════════════════════════════════════════════
# IK & joint mapping  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

def get_mapped_joints(robot):
    """16-element vector: [base_x, base_rot, arm1×5, arm2×5, grip1, grip2, head_pan, head_tilt]"""
    if robot is None:
        return np.zeros(16)
    full = robot.get_qpos()
    if hasattr(full, 'numpy'):
        full = full.numpy()
    if full.ndim > 1:
        full = full.squeeze()
    m = np.zeros(16)
    if len(full) >= 17:
        m[0]  = full[0]   # base X
        m[1]  = full[2]   # base rotation
        m[2]  = full[3]   # Arm1 Rotation
        m[3]  = full[6]   # Arm1 Pitch
        m[4]  = full[9]   # Arm1 Elbow
        m[5]  = full[11]  # Arm1 Wrist_Pitch
        m[6]  = full[13]  # Arm1 Wrist_Roll
        m[7]  = full[4]   # Arm2 Rotation
        m[8]  = full[7]   # Arm2 Pitch
        m[9]  = full[10]  # Arm2 Elbow
        m[10] = full[12]  # Arm2 Wrist_Pitch
        m[11] = full[14]  # Arm2 Wrist_Roll
        m[12] = full[15]  # Gripper 1
        m[13] = full[16]  # Gripper 2
        m[14] = full[5]   # Head pan
        m[15] = full[8]   # Head tilt
    return m


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """2-link planar IK for SO-101 arm → (shoulder_lift, elbow_flex)."""
    t1_off = math.atan2(0.028, 0.11257)
    t2_off = math.atan2(0.0052, 0.1349) + t1_off
    r = math.sqrt(x * x + y * y)
    r_max, r_min = l1 + l2, abs(l1 - l2)
    if r > r_max:
        s = r_max / r; x *= s; y *= s; r = r_max
    if r < r_min and r > 0:
        s = r_min / r; x *= s; y *= s; r = r_min
    c2 = -(r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = max(-1.0, min(1.0, c2))
    t2 = math.pi - math.acos(c2)
    t1 = math.atan2(y, x) + math.atan2(l2 * math.sin(t2), l1 + l2 * math.cos(t2))
    return (max(-0.1, min(3.45, t1 + t1_off)),
            max(-0.2, min(math.pi, t2 + t2_off)))


# ═══════════════════════════════════════════════════════════════════════════
# Calibration snapshot
# ═══════════════════════════════════════════════════════════════════════════

class Calibration:
    """Frozen reference frame captured at calibration moment."""
    def __init__(self, head_pos, head_quat, left_pos, left_quat, right_pos, right_quat):
        self.head_pos   = np.array(head_pos, dtype=float)
        self.head_quat  = np.array(head_quat, dtype=float)
        self.head_yaw   = extract_yaw(head_quat)
        self.head_pitch = extract_pitch(head_quat)
        self.left_pos   = np.array(left_pos, dtype=float)
        self.left_quat  = np.array(left_quat, dtype=float)
        self.right_pos  = np.array(right_pos, dtype=float)
        self.right_quat = np.array(right_quat, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args: Args):

    # ── PICO bridge ──────────────────────────────────────────────────────
    bridge = PicoVRBridge(port=args.pico_port, vr_to_robot_scale=args.vr_scale)
    bridge.start()
    print(f"\n[VR Teleop] Listening on port {args.pico_port} …")

    # ── Pygame panel ─────────────────────────────────────────────────────
    pygame.init()
    W, H = 640, 580
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("XLeRobot Full-Body VR Teleop")
    font      = pygame.font.SysFont(None, 22)
    font_big  = pygame.font.SysFont(None, 28)

    # ── ManiSkill env ────────────────────────────────────────────────────
    np.set_printoptions(suppress=True, precision=3)
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    env_kw = dict(
        obs_mode=args.obs_mode, reward_mode=args.reward_mode,
        control_mode=args.control_mode, render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs, sim_backend=args.sim_backend,
        enable_shadow=True, parallel_in_single_scene=False,
    )
    if args.robot_uids is not None:
        env_kw["robot_uids"] = args.robot_uids

    env: BaseEnv = gym.make(args.env_id, **env_kw)
    if not args.quiet:
        print("Action space", env.action_space)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        # First render can fail if the viewer window hasn't fully initialised yet
        try:
            env.render()
        except RuntimeError:
            time.sleep(0.5)
            env.render()

    robot = getattr(env.unwrapped, "agent", None)
    robot = robot.robot if robot else None

    # ── Control state ────────────────────────────────────────────────────
    action        = np.zeros(env.action_space.shape)
    target_joints = np.zeros(16)
    current_joints = get_mapped_joints(robot)

    TIP_LEN    = 0.108
    INIT_EE    = np.array([0.247, -0.023])   # (fwd reach, height) at rest
    ee1        = INIT_EE.copy()
    ee2        = INIT_EE.copy()
    pitch1     = 0.0
    pitch2     = 0.0

    # ── Tunable scales ───────────────────────────────────────────────────
    ARM_FWD    = 0.6     # VR fwd/back  → IK x
    ARM_VERT   = 1.0     # VR up/down   → IK y
    ARM_LAT    = 1.5     # VR left/right → shoulder rotation (rad / m)

    BASE_FWD_GAIN    = 1.5   # displacement → velocity
    BASE_FWD_MAX     = 0.5
    BASE_FWD_DEAD    = 0.05  # 5 cm dead-zone
    BASE_ROT_SCALE   = 1.0   # 1 : 1

    HEAD_PAN_SCALE   = 1.0
    HEAD_TILT_SCALE  = 1.0

    BODY_YAW_ALPHA   = 0.03  # low-pass: smaller = body follows head slower

    # P-gains  (action[0] is direct velocity, not P-controlled)
    pg = np.ones_like(action)
    pg[0]     = 0.0    # unused – base fwd is direct velocity
    pg[1]     = 0.8    # base rotation
    pg[2:7]   = 1.0    # arm 1
    pg[7:12]  = 1.0    # arm 2
    pg[12:14] = 0.1    # grippers
    pg[14:16] = 2.0    # head pan / tilt

    # ── State machine ────────────────────────────────────────────────────
    STATE     = "WAITING"        # WAITING → CALIBRATING → ACTIVE
    calib     = None             # Calibration object
    body_yaw  = 0.0              # low-pass filtered body yaw (world)

    step  = 0
    WARM  = 50                   # warmup steps (physics stabilisation)

    # ── Helper: draw one line ────────────────────────────────────────────
    _y = 0
    def draw(text, color=(200, 200, 200), big=False):
        nonlocal _y
        f = font_big if big else font
        screen.blit(f.render(str(text), True, color), (15, _y))
        _y += 26 if big else 21
    def gap():
        nonlocal _y; _y += 8

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════
    while True:
        # ── pygame events ────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); env.close(); bridge.stop(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    STATE = "CALIBRATING"; calib = None
                    ee1[:] = INIT_EE; ee2[:] = INIT_EE
                    pitch1 = pitch2 = 0.0
                    target_joints[:] = 0
                    bridge.reset_origins()
                    print("[VR Teleop] Re-calibrating …")
                elif ev.key == pygame.K_x:
                    ee1[:] = INIT_EE; ee2[:] = INIT_EE
                    pitch1 = pitch2 = 0.0
                    target_joints[:] = 0
                    print("[VR Teleop] Positions reset")

        # ── read VR ──────────────────────────────────────────────────────
        raw = bridge.get_latest_goal_nowait()
        lg  = raw.get("left")   if isinstance(raw, dict) else None
        rg  = raw.get("right")  if isinstance(raw, dict) else None
        hg  = raw.get("headset") if isinstance(raw, dict) else None
        has = (raw.get("has_left", False) or raw.get("has_right", False)) \
              if isinstance(raw, dict) else False

        # ══════════════════════════════════════════════════════════════════
        #  STATE: WAITING
        # ══════════════════════════════════════════════════════════════════
        if STATE == "WAITING":
            if has:
                STATE = "CALIBRATING"
                print("[VR Teleop] PICO connected → calibration mode")

        # ══════════════════════════════════════════════════════════════════
        #  STATE: CALIBRATING
        # ══════════════════════════════════════════════════════════════════
        elif STATE == "CALIBRATING" and step >= WARM:
            lt = lg.metadata.get("trigger", 0.0) if (lg and lg.metadata) else 0.0
            rt = rg.metadata.get("trigger", 0.0) if (rg and rg.metadata) else 0.0

            all_ready = (lt > 0.8 and rt > 0.8
                         and lg and lg.target_position is not None
                         and rg and rg.target_position is not None
                         and hg and hg.target_position is not None
                         and hg.metadata)

            if all_ready:
                hq = np.array(hg.metadata.get("quaternion", [0,0,0,1]), dtype=float)
                lq = np.array(lg.metadata.get("quaternion", [0,0,0,1]), dtype=float)
                rq = np.array(rg.metadata.get("quaternion", [0,0,0,1]), dtype=float)

                calib = Calibration(
                    head_pos=hg.target_position, head_quat=hq,
                    left_pos=lg.target_position, left_quat=lq,
                    right_pos=rg.target_position, right_quat=rq,
                )
                # sync bridge wrist origins to calibration quaternions
                bridge.reset_origins()
                bridge._left_origin_quat  = lq.copy()
                bridge._right_origin_quat = rq.copy()
                bridge._left_was_visible  = True
                bridge._right_was_visible = True

                body_yaw = calib.head_yaw
                STATE = "ACTIVE"
                print("[VR Teleop] ✓ Calibrated!  R = redo  |  X = reset arms")

        # ══════════════════════════════════════════════════════════════════
        #  STATE: ACTIVE
        # ══════════════════════════════════════════════════════════════════
        elif STATE == "ACTIVE" and step >= WARM and calib is not None:

            # ─── BODY (headset ≈ pelvis) → base ─────────────────────────
            if hg and hg.target_position is not None and hg.metadata:
                hp  = hg.target_position
                hq  = np.array(hg.metadata.get("quaternion", [0,0,0,1]), dtype=float)
                h_yaw   = extract_yaw(hq)
                h_pitch = extract_pitch(hq)

                # low-pass body yaw: slow turns = body, fast turns = head only
                body_yaw += BODY_YAW_ALPHA * normalize_angle(h_yaw - body_yaw)
                body_yaw_delta = normalize_angle(body_yaw - calib.head_yaw)

                # base rotation (P-controlled via target_joints[1])
                target_joints[1] = body_yaw_delta * BASE_ROT_SCALE

                # base forward velocity (headset displacement projected onto facing)
                dx = hp[0] - calib.head_pos[0]
                dz = hp[2] - calib.head_pos[2]
                fwd = dx * math.sin(body_yaw) + dz * math.cos(body_yaw)
                if abs(fwd) > BASE_FWD_DEAD:
                    action[0] = float(np.clip(fwd * BASE_FWD_GAIN,
                                              -BASE_FWD_MAX, BASE_FWD_MAX))
                else:
                    action[0] = 0.0

                # ─── HEAD CAMERA (relative to body) ─────────────────────
                head_rel_yaw   = normalize_angle(h_yaw - body_yaw)
                head_rel_pitch = h_pitch - calib.head_pitch
                target_joints[14] = head_rel_yaw   * HEAD_PAN_SCALE
                target_joints[15] = head_rel_pitch * HEAD_TILT_SCALE

            # ─── LEFT ARM ────────────────────────────────────────────────
            if lg and lg.target_position is not None:
                delta_room = lg.target_position - calib.left_pos
                byd = normalize_angle(body_yaw - calib.head_yaw)
                bx, bz = rotate_xz(delta_room[0], delta_room[2], byd)
                by = delta_room[1]

                ee1[0] = np.clip(INIT_EE[0] - bz * ARM_FWD,  0.02, 0.26)
                ee1[1] = np.clip(INIT_EE[1] + by * ARM_VERT, -0.18, 0.26)
                target_joints[2] = bx * ARM_LAT                       # shoulder pan

                if lg.wrist_flex_deg is not None:
                    pitch1 = math.radians(lg.wrist_flex_deg) * 0.5
                if lg.wrist_roll_deg is not None:
                    target_joints[6] = 1.57 + math.radians(lg.wrist_roll_deg)

                cy = ee1[1] + TIP_LEN * math.sin(pitch1)
                target_joints[3], target_joints[4] = inverse_kinematics(ee1[0], cy)
                target_joints[5] = target_joints[3] - target_joints[4] + pitch1

                # gripper: proportional (trigger 0→closed, 1→fully open)
                tr = lg.metadata.get("trigger", 0.0) if lg.metadata else 0.0
                target_joints[12] = 0.1 + tr * 2.4   # 0.1 … 2.5

            # ─── RIGHT ARM ───────────────────────────────────────────────
            if rg and rg.target_position is not None:
                delta_room = rg.target_position - calib.right_pos
                byd = normalize_angle(body_yaw - calib.head_yaw)
                bx, bz = rotate_xz(delta_room[0], delta_room[2], byd)
                by = delta_room[1]

                ee2[0] = np.clip(INIT_EE[0] - bz * ARM_FWD,  0.02, 0.26)
                ee2[1] = np.clip(INIT_EE[1] + by * ARM_VERT, -0.18, 0.26)
                target_joints[7] = -bx * ARM_LAT                      # mirrored

                if rg.wrist_flex_deg is not None:
                    pitch2 = math.radians(rg.wrist_flex_deg) * 0.5
                if rg.wrist_roll_deg is not None:
                    target_joints[11] = 1.57 + math.radians(rg.wrist_roll_deg)

                cy = ee2[1] + TIP_LEN * math.sin(pitch2)
                target_joints[8], target_joints[9] = inverse_kinematics(ee2[0], cy)
                target_joints[10] = target_joints[8] - target_joints[9] + pitch2

                tr = rg.metadata.get("trigger", 0.0) if rg.metadata else 0.0
                target_joints[13] = 0.1 + tr * 2.4

        # ══════════════════════════════════════════════════════════════════
        #  P-CONTROLLER
        # ══════════════════════════════════════════════════════════════════
        current_joints = get_mapped_joints(robot)
        if step < WARM:
            action = np.zeros_like(action)
        else:
            # action[0] already set (direct fwd velocity), P-control the rest
            for i in range(1, len(action)):
                action[i] = pg[i] * (target_joints[i] - current_joints[i])

        # ══════════════════════════════════════════════════════════════════
        #  PYGAME UI
        # ══════════════════════════════════════════════════════════════════
        screen.fill((20, 20, 30))
        _y = 10

        if STATE == "WAITING":
            draw("PICO 4 Ultra — Full-Body VR Teleop", (100, 200, 255), big=True)
            gap()
            draw(f"Waiting for PICO on port {args.pico_port} …", (255, 100, 100))
            draw(f"Step: {step}", (120, 120, 120))

        elif STATE == "CALIBRATING":
            draw("=== CALIBRATION ===", (255, 220, 80), big=True)
            gap()
            draw("1. Stand upright, face forward")
            draw("2. Hold both controllers in front of your waist,")
            draw("   arms slightly extended (robot rest pose)")
            draw("3. Pull BOTH TRIGGERS to lock calibration", (100, 255, 100))
            gap()
            lt = lg.metadata.get("trigger", 0.0) if (lg and lg.metadata) else 0.0
            rt = rg.metadata.get("trigger", 0.0) if (rg and rg.metadata) else 0.0
            lbar = "#" * int(lt * 20)
            rbar = "#" * int(rt * 20)
            draw(f"L trigger [{lbar:<20}] {lt:.2f}",
                 (0, 255, 0) if lt > 0.8 else (255, 255, 100))
            draw(f"R trigger [{rbar:<20}] {rt:.2f}",
                 (0, 255, 0) if rt > 0.8 else (255, 255, 100))
            if step < WARM:
                gap(); draw(f"Physics warmup {step}/{WARM} …", (120, 120, 120))

        elif STATE == "ACTIVE":
            byd = math.degrees(normalize_angle(body_yaw - calib.head_yaw)) \
                  if calib else 0.0
            draw("ACTIVE — Full-Body VR Teleop", (80, 255, 120), big=True)
            gap()
            draw(f"Body yaw: {byd:+6.1f}\u00b0   Fwd vel: {action[0]:+.3f}",
                 (255, 255, 100))
            draw(f"Head pan: {current_joints[14]:+.2f}   "
                 f"tilt: {current_joints[15]:+.2f}", (255, 255, 100))
            gap()
            draw(f"Arm1 EE ({ee1[0]:.3f}, {ee1[1]:.3f})  "
                 f"pitch {pitch1:+.2f}", (255, 200, 100))
            draw(f"Arm2 EE ({ee2[0]:.3f}, {ee2[1]:.3f})  "
                 f"pitch {pitch2:+.2f}", (255, 200, 100))
            gap()
            draw(f"Arm1 joints: {np.round(current_joints[2:7], 2)}", (180, 180, 255))
            draw(f"Arm2 joints: {np.round(current_joints[7:12], 2)}", (180, 180, 255))
            draw(f"Grippers:  L={current_joints[12]:.2f}  "
                 f"R={current_joints[13]:.2f}", (180, 180, 255))
            gap()
            draw("R = re-calibrate  |  X = reset arms", (130, 130, 160))

        pygame.display.flip()

        # ── step sim ─────────────────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        if args.render_mode is not None:
            try:
                env.render()
            except RuntimeError:
                pass  # skip frame if viewer is being resized
        time.sleep(0.01)

    pygame.quit()
    env.close()
    bridge.stop()


if __name__ == "__main__":
    main(tyro.cli(Args))
