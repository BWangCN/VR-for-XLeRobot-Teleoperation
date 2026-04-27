# XLeRobot PICO 4 Ultra VR Teleoperation

Low-latency VR teleoperation for the [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) dual-arm mobile robot using the **PICO 4 Ultra** headset.

A native Unity app on the headset streams controller, head, button and thumbstick data over TCP to a Python bridge running on a PC. The bridge converts the stream into kinematic commands for either the **ManiSkill simulation** or the **real robot** (Feetech servos via LeRobot).

## Architecture

```
PICO 4 Ultra (Unity)              PC (Windows / Linux)
+----------------------+         +-------------------------------+
| PicoXLeRobotStreamer |         | pico_vr_bridge.py             |
|  - Controller pos/rot|--TCP--->|  - Parses JSON                |
|  - Trigger / grip    | :9876   |  - Tracks origin quaternions  |
|  - Thumbstick / btns |         |  - Produces ControlGoal       |
|  - Head pose         |         |             |                 |
|  - Passthrough on    |         |             v                 |
+----------------------+         | demo_pico_vr_*.py             |
                                  |  - IK + P-controller          |
                                  |  - ManiSkill sim OR real bus  |
                                  |  - Pygame status panel        |
                                  +-------------------------------+
```

The transport is newline-delimited JSON over TCP at 60 Hz. The bridge exposes the same `ControlGoal` API as the upstream [XLeVR](https://github.com/Vector-Wangel/XLeRobot/tree/main/XLeVR) WebSocket system, so existing demos can be swapped over with a one-line import change.

## Features

| Demo | Script | Mode | Description |
|---|---|---|---|
| Basic sim | `demo_pico_vr_sim.py` | ManiSkill | Dual-arm IK from controller deltas, gripper from triggers, base from left thumbstick. First-person head-camera toggle. |
| Full-body sim | `demo_pico_vr_fullbody_sim.py` | ManiSkill | T-pose calibration, body-frame arm tracking, headset acts as pelvis (drives base translation), head rotation drives camera with low-pass-filtered body yaw. |
| Real robot | `demo_pico_vr_real.py` | Feetech bus | Drives physical arms via LeRobot `FeetechMotorsBus`. Safety-clamped P-control, recordable home pose, automatic safe-return on quit. |

## Requirements

### Hardware

- PICO 4 Ultra headset (also works on PICO Neo 3 / 4 with minor SDK changes)
- PC on the same Wi-Fi network (Windows 11 tested; Linux/macOS should work)
- For the real-robot demo: an XLeRobot with two SO-101 / SO-100 arms on Feetech `sts3215` servos, two USB-serial buses

### Software

- **Unity 2022.3 LTS** with the **PICO Unity Integration SDK 3.3.x** (drop the SDK into `Packages/` or import via Package Manager → Add from disk).
- **Python 3.11** for the simulation demos (`xlerobot_sim` env)
- **Python 3.12** for the real-robot demo (`xlerobot` env)

### Conda environments (recommended)

Two separate envs are recommended because ManiSkill / SAPIEN pin to Python 3.11 while LeRobot 0.4.5 needs Python 3.12.

**Simulation env** — `xlerobot_sim` (Python 3.11):

```bash
conda create -n xlerobot_sim python=3.11 -y
conda activate xlerobot_sim
pip install numpy scipy pygame tyro
pip install mani_skill          # pulls in sapien + gymnasium
```

**Real-robot env** — `xlerobot` (Python 3.12):

```bash
conda create -n xlerobot python=3.12 -y
conda activate xlerobot
pip install numpy scipy pygame
pip install "lerobot[feetech]>=0.4.5"
```

### Known environment gotchas

- **SAPIEN / Vulkan on Windows.** SAPIEN renders through Vulkan. If `env.render()` segfaults or shows a black window, install the latest GPU driver and the [Vulkan Runtime](https://vulkan.lunarg.com/sdk/home). On laptops, force the dedicated GPU (NVIDIA Control Panel → Manage 3D Settings → Program Settings → `python.exe` → High-performance NVIDIA processor).
- **SAPIEN ray-traced shaders.** `--shader rt-fast` and `--shader rt` need an RTX-class GPU and will silently fall back to a black screen on integrated GPUs. Use `--shader default` (already the default in the launch scripts) if unsure.
- **`gymnasium` version mismatch.** ManiSkill 3.x officially declares `gymnasium==0.29.1` but works fine with `gymnasium>=1.0`. If pip resolves to 1.x, ignore the warning.
- **SAPIEN warnings about `pinocchio` / `pkg_resources`.** Non-critical; they appear once at import time and can be ignored.
- **`pygame` window won't focus on Windows.** If keyboard hotkeys (`R`, `X`, `V`) don't register, click the pygame window once to focus it — the SAPIEN viewer steals focus on first render.
- **ReplicaCAD assets.** First run of `ReplicaCAD_SceneManipulation-v1` downloads ~2 GB into `~/.maniskill/data/`. Make sure that path has space.
- **LeRobot calibration file.** `demo_pico_vr_real.py` reads calibration from `~/.cache/huggingface/lerobot/calibration/robots/xlerobot/None.json`. If you have not yet calibrated through LeRobot's standard flow, the script prints a warning and runs with raw motor units — calibrate first via `lerobot` CLI.
- **Feetech serial ports.** On Windows the buses appear as `COM3` / `COM4` (use Device Manager to check); on Linux they are `/dev/ttyUSB0` / `/dev/ttyUSB1`. Pass them with `--port1` / `--port2`.
- **Firewall.** Windows Defender will prompt the first time the bridge binds to `0.0.0.0:9876` — allow it on **Private** networks. If you missed the prompt, manually add an inbound rule for TCP 9876.
- **PICO Integration SDK path.** The bundled `UnityProject/Packages/manifest.json` references the SDK with a local file path (`file:.../PICO-Unity-Integration-SDK`). When you reopen this project on a different machine, repoint that line in `manifest.json` to wherever you installed the SDK, or replace it with the registry version.

## Quick Start

### 1. Python bridge / demo

```bash
# Simulation (basic)
conda activate xlerobot_sim
cd pico_vr/server
python demo_pico_vr_sim.py

# Simulation (full-body)
python demo_pico_vr_fullbody_sim.py

# Real robot
conda activate xlerobot
python demo_pico_vr_real.py --port1 COM4 --port2 COM3
```

The console prints the local IP and port (default `9876`). Note them — Unity needs them.

### 2. Unity side (PICO)

1. Open your Unity project (or create one with the PICO Integration SDK installed and a passthrough-ready XR Origin in the scene).
2. Copy `pico_vr/unity/PicoXLeRobotStreamer.cs` into `Assets/Scripts/`.
3. Add an empty `GameObject`, attach `PicoXLeRobotStreamer`, and set:
   - **Pc Ip Address** = your PC's LAN IP (run `ipconfig` / `ip addr`)
   - **Port** = `9876`
   - **Send Rate** = `60` (Hz)
4. Build and deploy to the PICO 4 Ultra. Launch — it auto-connects and auto-reconnects.

A minimal Unity skeleton is included under `UnityProject/` for reference.

## TCP Protocol

Each frame is a single line of JSON terminated by `\n`:

```json
{
  "left": {
    "pos": [x, y, z],
    "rot": [qx, qy, qz, qw],
    "trigger": 0.0,
    "grip": 0.0,
    "thumbstick": [x, y],
    "primaryButton": false,
    "secondaryButton": false
  },
  "right": { "...": "same shape" },
  "head":  { "pos": [x, y, z], "rot": [qx, qy, qz, qw] },
  "timestamp": 1234567890
}
```

`pos` is in metres in the headset tracking frame. `rot` is `[x, y, z, w]`.

## Control Mapping

### Basic sim (`demo_pico_vr_sim.py`)

| Input | Action |
|---|---|
| Left controller position | Robot left arm end-effector (delta from calibration) |
| Right controller position | Robot right arm end-effector |
| Trigger | (analog, also drives gripper open/close in real demo) |
| A button | Open gripper (incremental) |
| B button | Close gripper (incremental) |
| Wrist twist (roll) | Wrist roll |
| Wrist tilt (pitch) | Wrist flex |
| Left thumbstick Y | Base forward / backward |
| Left thumbstick X | Base rotation |
| `R` (PC keyboard) | Re-enter calibration |
| `X` (PC) | Reset all positions |
| `V` (PC) | Toggle first-person head camera |

### Full-body sim (`demo_pico_vr_fullbody_sim.py`)

| Input | Action |
|---|---|
| Pull both triggers (T-pose) | Calibrate body / arm origins |
| Headset translation | Robot base forward + lateral via body-frame projection |
| Headset yaw | Drives base rotation via low-pass filter |
| Headset pitch / yaw (relative) | Robot head camera tilt / pan |
| Controller position (in body frame) | Arm IK target |
| Trigger (analog) | Proportional gripper |

### Real robot (`demo_pico_vr_real.py`)

Same arm mapping as the basic sim. **Base and head are not actuated.** Safety features:

- Joint-limit clamping (`JOINT_LIMITS` per joint)
- Per-step delta clamping (`MAX_STEP_DEG = 6°`)
- On quit (`Q` or window close): arms automatically P-controlled back to the recorded home pose, then torque is disabled.

To record a custom home pose:

```bash
python record_home_pose.py --port1 COM4 --port2 COM3
# Manually move both arms to the desired rest pose, press Enter.
# Saves home_pose.json next to the script; auto-loaded by the teleop demo.
```

## Calibration Workflow

All three demos share the same state machine:

1. **WAITING** — server up, no PICO connected.
2. **PREP** — PICO connected, arms drive to home pose; user has ~2 s to position hands.
3. **CALIB** — bridge averages controller positions over ~5 s to lock in an origin (the deltas during ACTIVE are computed against this origin, so absolute headset coordinates do not matter).
4. **ACTIVE** — normal control. Press `R` any time to re-enter PREP and re-calibrate.

## Tuning

The most useful knobs sit at the top of each demo's `main()`:

| Constant | Meaning |
|---|---|
| `VR_SCALE_LAT / VERT / FWD` | Metres of robot EE travel per metre of controller travel, per axis |
| `EE_HOME` | Default end-effector reach / height in arm cylindrical frame |
| `GRIP_MIN / MAX / SPEED` | Gripper range and slew rate |
| `BASE_FWD_GAIN / DEAD` | Body-translation → base velocity (full-body demo) |
| `BODY_YAW_ALPHA` | Low-pass for body-yaw following head (full-body demo) |
| `KP`, `MAX_STEP_DEG` | Real-robot P-gain and per-step clamp |

## Repository Layout

```
pico_vr/
  server/
    pico_vr_bridge.py             TCP server, JSON parser, ControlGoal producer
    demo_pico_vr_sim.py           ManiSkill basic teleop demo
    demo_pico_vr_fullbody_sim.py  ManiSkill full-body teleop demo
    demo_pico_vr_real.py          Real-robot teleop (LeRobot Feetech bus)
    record_home_pose.py           Hand-guided home-pose recorder
    home_pose.json                Saved real-robot home pose
  unity/
    PicoXLeRobotStreamer.cs       Drop-in Unity client script
  UnityProject/                   Minimal Unity scene skeleton (for reference)
  SETUP.txt                       Step-by-step setup notes
  README.md
```

## Drop-in Bridge API

`PicoVRBridge` mirrors the `VRMonitor` API from upstream XLeVR, so existing scripts written for the Quest 3 WebSocket pipeline work unchanged:

```python
from pico_vr_bridge import PicoVRBridge

bridge = PicoVRBridge(port=9876, vr_to_robot_scale=1.0)
bridge.start()

left  = bridge.get_left_goal_nowait()    # ControlGoal | None
right = bridge.get_right_goal_nowait()
all_  = bridge.get_latest_goal_nowait()  # dict with 'left', 'right', 'headset'

bridge.reset_origins()  # re-capture wrist quaternion origins
bridge.stop()
```

Each `ControlGoal` carries `target_position`, `wrist_roll_deg`, `wrist_flex_deg`, `gripper_closed`, and a `metadata` dict with raw trigger / grip / thumbstick / button values.

## Troubleshooting

- **PICO sits at "Disconnected"** — check the PC firewall allows inbound TCP `9876`, both devices are on the same subnet, and the IP in the Unity script matches `ipconfig` / `ip addr`.
- **Robot jumps on calibration** — make sure you are holding still during the 5 s CALIB window. Press `R` to redo it.
- **Wrist roll feels inverted** — flip the sign of `wrist_roll_deg` in `_process_controller`, or negate the `target_joints[*]` line in the demo.
- **Real robot moves wrong direction** — adjust `VR_SCALE_LAT` sign or swap left/right controller bindings in the demo.
- **First-person camera tearing** — disable with `V`; the SAPIEN viewer occasionally races on the very first frame and prints once.
- **Passthrough not visible on PICO** — confirm PICO Integration SDK 3.3+ is installed and the main camera's clear flags are `SolidColor` with alpha 0 (the script sets this on `Start()`).

## Related Projects

- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) — the dual-arm mobile robot platform this targets
- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face robot-learning library used for motor I/O on the real robot
- [ManiSkill](https://github.com/haosulab/ManiSkill) — simulation backend
- [XLeVR](https://github.com/Vector-Wangel/XLeRobot/tree/main/XLeVR) — Meta Quest 3 WebXR teleoperation (sibling system, same `ControlGoal` interface)

## License

Inherits from the parent XLeRobot project. See the upstream repository for details.
