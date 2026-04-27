"""
XLeRobot ManiSkill Simulation with PICO 4 Ultra VR Teleoperation

Features:
  - Calibration sequence: arms reset → 2s positioning → 5s capture → active
  - Delta-based arm mapping with shoulder pan, IK, wrist roll/flex
  - Proportional gripper (trigger analog)
  - V key toggles first-person (robot head camera) view

Usage:
    conda activate xlerobot_sim
    python demo_pico_vr_sim.py
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
    """Scale factor for VR position to robot position mapping"""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_mapped_joints(robot):
    """16-element vector from full qpos."""
    if robot is None:
        return np.zeros(16)
    full = robot.get_qpos()
    if hasattr(full, 'numpy'):
        full = full.numpy()
    if full.ndim > 1:
        full = full.squeeze()
    m = np.zeros(16)
    if len(full) >= 17:
        m[0]  = full[0];   m[1]  = full[2]
        m[2]  = full[3];   m[3]  = full[6];   m[4]  = full[9]
        m[5]  = full[11];  m[6]  = full[13]
        m[7]  = full[4];   m[8]  = full[7];   m[9]  = full[10]
        m[10] = full[12];  m[11] = full[14]
        m[12] = full[15];  m[13] = full[16]
        m[14] = full[5];   m[15] = full[8]
    return m


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    t1_off = math.atan2(0.028, 0.11257)
    t2_off = math.atan2(0.0052, 0.1349) + t1_off
    r = math.sqrt(x*x + y*y)
    r_max, r_min = l1+l2, abs(l1-l2)
    if r > r_max:
        s = r_max/r; x *= s; y *= s; r = r_max
    if r < r_min and r > 0:
        s = r_min/r; x *= s; y *= s
    c2 = max(-1.0, min(1.0, -(r*r - l1*l1 - l2*l2)/(2*l1*l2)))
    t2 = math.pi - math.acos(c2)
    t1 = math.atan2(y, x) + math.atan2(l2*math.sin(t2), l1+l2*math.cos(t2))
    return (max(-0.1, min(3.45, t1+t1_off)),
            max(-0.2, min(math.pi, t2+t2_off)))


def find_head_link(robot):
    """Find the head_camera_link for first-person view."""
    if robot is None:
        return None
    for link in robot.get_links():
        if "head_camera" in link.name:
            return link
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args: Args):

    # ── PICO bridge ──────────────────────────────────────────────────────
    bridge = PicoVRBridge(port=args.pico_port, vr_to_robot_scale=args.vr_scale)
    bridge.start()
    print(f"\n[VR] Listening on port {args.pico_port} …")

    # ── Pygame ───────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((640, 560))
    pygame.display.set_caption("XLeRobot PICO VR Teleop")
    font     = pygame.font.SysFont(None, 22)
    font_big = pygame.font.SysFont(None, 28)

    # ── ManiSkill ────────────────────────────────────────────────────────
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
    viewer = None
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        try:
            env.render()
        except RuntimeError:
            time.sleep(0.5)
            env.render()

    robot = getattr(env.unwrapped, "agent", None)
    robot = robot.robot if robot else None
    head_link = find_head_link(robot)

    # ── Control state ────────────────────────────────────────────────────
    action         = np.zeros(env.action_space.shape)
    target_joints  = np.zeros(16)
    current_joints = get_mapped_joints(robot)

    TIP   = 0.108
    # 3D EE home position: [reach, lateral, height] in arm cylindrical frame
    EE_HOME = np.array([0.162, 0.0, 0.118])
    pitch1 = 0.0;         pitch2 = 0.0

    left_origin  = None    # captured during calibration
    right_origin = None

    # VR-to-EE scale per axis (metres VR → metres robot)
    VR_SCALE_LAT = 1.2;  VR_SCALE_VERT = 0.45;  VR_SCALE_FWD = 0.3

    GRIP_MIN   = 0.1     # fully closed
    GRIP_MAX   = 2.5     # fully open
    GRIP_SPEED = 0.04    # per frame (~2.4 per sec at 60 fps)

    pg = np.ones_like(action)
    pg[0] = 1.5;  pg[1] = 0.8
    pg[2:7] = 1.0;  pg[7:12] = 1.0
    pg[12:14] = 0.1;  pg[14:16] = 2.0

    # ── State machine ────────────────────────────────────────────────────
    #   WAITING  → PICO connects
    #   PREP     → arms go home, 2 s for you to move hands
    #   CALIB    → 5 s capture, averaging VR positions
    #   ACTIVE   → normal control
    STATE       = "WAITING"
    phase_t0    = 0.0          # timestamp when current phase started
    PREP_DUR    = 2.0          # seconds
    CALIB_DUR   = 5.0          # seconds
    calib_buf_l = []           # list of np arrays during CALIB
    calib_buf_r = []
    step        = 0
    WARM        = 50

    # ── Camera ───────────────────────────────────────────────────────────
    first_person = True        # V key toggles; start in 1st-person
    FP_HEIGHT_OFFSET = 0.08   # metres above head_camera_link
    _fp_err_printed = False    # only print camera error once

    # ── draw helpers ─────────────────────────────────────────────────────
    _y = 0
    def draw(txt, color=(200,200,200), big=False):
        nonlocal _y
        f = font_big if big else font
        screen.blit(f.render(str(txt), True, color), (15, _y))
        _y += 26 if big else 21
    def gap():
        nonlocal _y; _y += 8

    def enter_prep():
        nonlocal STATE, phase_t0, left_origin, right_origin
        nonlocal pitch1, pitch2, target_joints
        STATE = "PREP";  phase_t0 = time.time()
        left_origin = None;  right_origin = None
        pitch1 = pitch2 = 0.0
        # Home position from EE_HOME (reach, height)
        target_joints[:] = 0
        t1, t2 = inverse_kinematics(EE_HOME[0], EE_HOME[2])
        target_joints[3] = t1;   target_joints[4] = t2     # arm1 lift/elbow
        target_joints[5] = t1 - t2                          # arm1 wrist pitch
        target_joints[6] = 1.57                              # arm1 wrist roll
        target_joints[8] = t1;   target_joints[9] = t2     # arm2 lift/elbow
        target_joints[10] = t1 - t2                          # arm2 wrist pitch
        target_joints[11] = 1.57                             # arm2 wrist roll
        bridge.reset_origins()
        print("[VR] Arms homing — position your hands …")

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════
    while True:

        # ── events ───────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); env.close(); bridge.stop(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    enter_prep()
                elif ev.key == pygame.K_x:
                    enter_prep()
                elif ev.key == pygame.K_v:
                    first_person = not first_person
                    print(f"[VR] Camera: {'FIRST-PERSON' if first_person else 'FREE'}")

        # ── read VR ──────────────────────────────────────────────────────
        raw = bridge.get_latest_goal_nowait()
        lg = raw.get("left")    if isinstance(raw, dict) else None
        rg = raw.get("right")   if isinstance(raw, dict) else None
        hg = raw.get("headset") if isinstance(raw, dict) else None
        has = (raw.get("has_left",False) or raw.get("has_right",False)) \
              if isinstance(raw, dict) else False

        elapsed = time.time() - phase_t0

        # ══════════════════════════════════════════════════════════════════
        #  WAITING
        # ══════════════════════════════════════════════════════════════════
        if STATE == "WAITING":
            if has and step >= WARM:
                enter_prep()

        # ══════════════════════════════════════════════════════════════════
        #  PREP — arms at home, wait PREP_DUR for user to position hands
        # ══════════════════════════════════════════════════════════════════
        elif STATE == "PREP":
            # keep target at zero → P-controller drives arms home
            target_joints[:] = 0
            if elapsed >= PREP_DUR:
                STATE = "CALIB";  phase_t0 = time.time()
                calib_buf_l.clear();  calib_buf_r.clear()
                bridge.reset_origins()
                print("[VR] Calibrating — hold still for 5 s …")

        # ══════════════════════════════════════════════════════════════════
        #  CALIB — collect VR samples for CALIB_DUR, then average
        # ══════════════════════════════════════════════════════════════════
        elif STATE == "CALIB":
            if lg and lg.target_position is not None:
                calib_buf_l.append(lg.target_position.copy())
            if rg and rg.target_position is not None:
                calib_buf_r.append(rg.target_position.copy())

            if elapsed >= CALIB_DUR:
                if calib_buf_l:
                    left_origin = np.mean(calib_buf_l, axis=0)
                    print(f"[VR] Left  origin ({len(calib_buf_l)} samples): {left_origin}")
                if calib_buf_r:
                    right_origin = np.mean(calib_buf_r, axis=0)
                    print(f"[VR] Right origin ({len(calib_buf_r)} samples): {right_origin}")

                # lock bridge wrist quaternion origins to current pose
                bridge.reset_origins()

                STATE = "ACTIVE"
                print("[VR] ACTIVE — R = re-calibrate | V = toggle view")

        # ══════════════════════════════════════════════════════════════════
        #  ACTIVE — normal control
        # ══════════════════════════════════════════════════════════════════
        elif STATE == "ACTIVE" and step >= WARM:

            # ─── ROBOT LEFT ARM = Arm2 [7-11,13] ← USER LEFT HAND (lg) ──
            if (lg and lg.target_position is not None
                    and left_origin is not None):
                d = lg.target_position - left_origin

                # 3D EE target: [reach, lateral, height]
                ee_reach = EE_HOME[0] - d[2] * VR_SCALE_FWD
                ee_lat   = -d[0] * VR_SCALE_LAT   # negate: VR +X=right, robot pan +Y=left
                ee_ht    = EE_HOME[2] + d[1] * VR_SCALE_VERT

                # Cylindrical decomposition
                target_joints[7] = math.atan2(ee_lat, max(ee_reach, 0.01))
                planar_r = math.sqrt(ee_reach**2 + ee_lat**2)
                planar_r = np.clip(planar_r, 0.02, 0.24)
                ee_ht    = np.clip(ee_ht, -0.18, 0.24)

                if lg.wrist_flex_deg is not None:
                    pitch2 = math.radians(lg.wrist_flex_deg) * 0.5
                if lg.wrist_roll_deg is not None:
                    target_joints[11] = 1.57 + math.radians(lg.wrist_roll_deg)

                cy = ee_ht + TIP * math.sin(pitch2)
                target_joints[8], target_joints[9] = inverse_kinematics(planar_r, cy)
                target_joints[10] = target_joints[8] - target_joints[9] + pitch2

                # Gripper: A = open, B = close
                meta_l = lg.metadata or {}
                if meta_l.get("primaryButton", False):
                    target_joints[13] = min(GRIP_MAX, target_joints[13] + GRIP_SPEED)
                elif meta_l.get("secondaryButton", False):
                    target_joints[13] = max(GRIP_MIN, target_joints[13] - GRIP_SPEED)

                # base from left thumbstick
                ts = meta_l.get("thumbstick", [0,0])
                if abs(ts[0]) > 0.15:
                    target_joints[1] -= ts[0] * 0.02
                if abs(ts[1]) > 0.15:
                    action[0] = ts[1] * 0.15

            # ─── ROBOT RIGHT ARM = Arm1 [2-6,12] ← USER RIGHT HAND (rg) ─
            if (rg and rg.target_position is not None
                    and right_origin is not None):
                d = rg.target_position - right_origin

                ee_reach = EE_HOME[0] - d[2] * VR_SCALE_FWD
                ee_lat   = -d[0] * VR_SCALE_LAT   # negate: VR +X=right, robot pan +Y=left
                ee_ht    = EE_HOME[2] + d[1] * VR_SCALE_VERT

                target_joints[2] = math.atan2(ee_lat, max(ee_reach, 0.01))
                planar_r = math.sqrt(ee_reach**2 + ee_lat**2)
                planar_r = np.clip(planar_r, 0.02, 0.24)
                ee_ht    = np.clip(ee_ht, -0.18, 0.24)

                if rg.wrist_flex_deg is not None:
                    pitch1 = math.radians(rg.wrist_flex_deg) * 0.5
                if rg.wrist_roll_deg is not None:
                    target_joints[6] = 1.57 + math.radians(rg.wrist_roll_deg)

                cy = ee_ht + TIP * math.sin(pitch1)
                target_joints[3], target_joints[4] = inverse_kinematics(planar_r, cy)
                target_joints[5] = target_joints[3] - target_joints[4] + pitch1

                # Gripper: A = open, B = close
                meta_r = rg.metadata or {}
                if meta_r.get("primaryButton", False):
                    target_joints[12] = min(GRIP_MAX, target_joints[12] + GRIP_SPEED)
                elif meta_r.get("secondaryButton", False):
                    target_joints[12] = max(GRIP_MIN, target_joints[12] - GRIP_SPEED)

            # ─── HEAD (disabled — absolute forward) ─────────────────────
            # Head joints stay at 0 → camera always faces robot forward
            target_joints[14] = 0.0
            target_joints[15] = 0.0

        # ══════════════════════════════════════════════════════════════════
        #  P-CONTROLLER
        # ══════════════════════════════════════════════════════════════════
        current_joints = get_mapped_joints(robot)
        if step < WARM:
            action = np.zeros_like(action)
        else:
            for i in range(1, len(action)):
                action[i] = pg[i] * (target_joints[i] - current_joints[i])

        # ══════════════════════════════════════════════════════════════════
        #  FIRST-PERSON CAMERA (V key)
        # ══════════════════════════════════════════════════════════════════
        if first_person and viewer is not None and head_link is not None:
            try:
                pose = head_link.get_pose()
                p = pose.p                          # [x, y, z]  numpy
                q = pose.q                          # [w, x, y, z] in SAPIEN
                # Offset slightly above head link (in head-local up direction)
                r = Rot.from_quat([q[1], q[2], q[3], q[0]])
                up = r.apply([0, 0, FP_HEIGHT_OFFSET])
                cam_pos = [float(p[0]+up[0]), float(p[1]+up[1]), float(p[2]+up[2])]
                # Build sapien.Pose directly (wxyz quaternion)
                cam_pose = sapien.Pose(p=cam_pos, q=[float(q[0]),float(q[1]),float(q[2]),float(q[3])])
                if hasattr(viewer, 'window') and hasattr(viewer.window, 'set_camera_pose'):
                    viewer.window.set_camera_pose(cam_pose)
                elif hasattr(viewer, 'set_camera_pose'):
                    viewer.set_camera_pose(cam_pose)
                else:
                    viewer.set_camera_xyz(*cam_pos)
                    rpy = r.as_euler('xyz')
                    viewer.set_camera_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            except Exception as e:
                if not _fp_err_printed:
                    print(f"[VR] First-person camera error: {e}")
                    _fp_err_printed = True

        # ══════════════════════════════════════════════════════════════════
        #  PYGAME UI
        # ══════════════════════════════════════════════════════════════════
        screen.fill((20, 20, 30))
        _y = 10

        if STATE == "WAITING":
            draw("PICO 4 Ultra VR Teleop", (100,200,255), big=True); gap()
            draw(f"Waiting for PICO on port {args.pico_port} …", (255,100,100))
            draw(f"Step: {step}", (120,120,120))

        elif STATE == "PREP":
            remain = max(0, PREP_DUR - elapsed)
            draw("=== RESET & POSITIONING ===", (255,220,80), big=True); gap()
            draw("Arms returning to home position …", (220,220,220))
            draw(f"Move your hands to match the robot rest pose", (220,220,220))
            draw(f"Calibration starts in {remain:.1f} s", (100,255,100))

        elif STATE == "CALIB":
            remain = max(0, CALIB_DUR - elapsed)
            pct = min(100, int(100 * elapsed / CALIB_DUR))
            bar = "#" * (pct // 5)
            draw("=== CALIBRATING ===", (80,200,255), big=True); gap()
            draw("HOLD STILL — capturing VR origin …", (255,255,100))
            draw(f"[{bar:<20}] {pct}%   {remain:.1f} s left", (100,255,100))
            gap()
            draw(f"L samples: {len(calib_buf_l)}   R samples: {len(calib_buf_r)}",
                 (180,180,180))

        elif STATE == "ACTIVE":
            # compute deltas for display
            if left_origin is not None and lg and lg.target_position is not None:
                dl = lg.target_position - left_origin
                ld = f"dX={dl[0]:+.3f} dY={dl[1]:+.3f} dZ={dl[2]:+.3f}"
            else:
                ld = "—"
            if right_origin is not None and rg and rg.target_position is not None:
                dr = rg.target_position - right_origin
                rd = f"dX={dr[0]:+.3f} dY={dr[1]:+.3f} dZ={dr[2]:+.3f}"
            else:
                rd = "—"

            cam = "1ST-PERSON" if first_person else "FREE"
            draw(f"ACTIVE   cam: {cam}", (80,255,120), big=True); gap()
            draw(f"L delta: {ld}", (0,255,0))
            draw(f"R delta: {rd}", (0,255,0)); gap()
            draw(f"Arm1 Pan:{target_joints[2]:+.2f}rad"
                 f"  Lift:{target_joints[3]:.2f}  Elbow:{target_joints[4]:.2f}"
                 f"  WrRoll:{target_joints[6]:.2f}", (255,200,100))
            draw(f"Arm2 Pan:{target_joints[7]:+.2f}rad"
                 f"  Lift:{target_joints[8]:.2f}  Elbow:{target_joints[9]:.2f}"
                 f"  WrRoll:{target_joints[11]:.2f}", (255,200,100))
            draw(f"Pitch: L={pitch1:+.3f}  R={pitch2:+.3f}", (255,200,100)); gap()
            draw(f"Arm1: {np.round(current_joints[2:7], 2)}", (200,200,255))
            draw(f"Arm2: {np.round(current_joints[7:12], 2)}", (200,200,255))
            draw(f"Grip: L={current_joints[12]:.2f} R={current_joints[13]:.2f}",
                 (200,200,255))
            draw(f"Base: [{current_joints[0]:.2f}, {current_joints[1]:.2f}]",
                 (200,200,255)); gap()
            draw("R = re-calibrate | X = reset | V = toggle camera",
                 (130,130,160))

        pygame.display.flip()

        # ── step sim ─────────────────────────────────────────────────────
        obs, rew, term, trunc, info = env.step(action)
        step += 1
        if args.render_mode is not None:
            try:
                env.render()
            except RuntimeError:
                pass
        time.sleep(0.01)

    pygame.quit(); env.close(); bridge.stop()


if __name__ == "__main__":
    main(tyro.cli(Args))
