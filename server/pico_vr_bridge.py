"""
PICO VR Bridge Server for XLeRobot Simulation

Receives controller data from PICO 4 Ultra via TCP,
converts to ControlGoal objects compatible with ManiSkill simulation.

Protocol: Simple JSON over TCP (newline-delimited)
Port: 9876 (configurable)
"""

import asyncio
import json
import socket
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Data structures (matching XLeVR ControlGoal for compatibility)
# ---------------------------------------------------------------------------

class ControlMode(Enum):
    IDLE = "idle"
    POSITION_CONTROL = "position_control"


@dataclass
class ControlGoal:
    arm: str  # "left", "right", "headset"
    mode: Optional[ControlMode] = None
    target_position: Optional[np.ndarray] = None
    wrist_roll_deg: Optional[float] = None
    wrist_flex_deg: Optional[float] = None
    gripper_closed: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def extract_roll_from_quaternion(current_quat, origin_quat):
    """Extract Z-axis (roll) rotation between two quaternions, in degrees."""
    origin_rot = R.from_quat(origin_quat)
    current_rot = R.from_quat(current_quat)
    relative = current_rot * origin_rot.inv()
    rotvec = relative.as_rotvec()
    return -np.degrees(rotvec[2])


def extract_pitch_from_quaternion(current_quat, origin_quat):
    """Extract X-axis (pitch) rotation between two quaternions, in degrees."""
    origin_rot = R.from_quat(origin_quat)
    current_rot = R.from_quat(current_quat)
    relative = current_rot * origin_rot.inv()
    rotvec = relative.as_rotvec()
    return np.degrees(rotvec[0])


# ---------------------------------------------------------------------------
# PICO VR Bridge
# ---------------------------------------------------------------------------

class PicoVRBridge:
    """
    TCP server that receives PICO controller data and produces ControlGoal objects.

    Expected JSON from PICO (newline-delimited):
    {
      "left": {
        "pos": [x, y, z],
        "rot": [qx, qy, qz, qw],
        "trigger": 0.0-1.0,
        "grip": 0.0-1.0,
        "thumbstick": [x, y],
        "primaryButton": false,
        "secondaryButton": false
      },
      "right": { ... same structure ... },
      "head": {
        "pos": [x, y, z],
        "rot": [qx, qy, qz, qw]
      },
      "timestamp": 1234567890
    }
    """

    def __init__(self, host="0.0.0.0", port=9876, vr_to_robot_scale=1.0):
        self.host = host
        self.port = port
        self.vr_to_robot_scale = vr_to_robot_scale

        self.is_running = False
        self._lock = threading.Lock()

        # Latest goals
        self.left_goal: Optional[ControlGoal] = None
        self.right_goal: Optional[ControlGoal] = None
        self.headset_goal: Optional[ControlGoal] = None

        # Origin quaternions (captured on first valid frame)
        self._left_origin_quat = None
        self._right_origin_quat = None
        self._left_was_visible = False
        self._right_was_visible = False

        # Server socket
        self._server_socket = None
        self._server_thread = None

    def start(self):
        """Start the TCP server in a background thread."""
        self.is_running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

    def stop(self):
        """Stop the server."""
        self.is_running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

    def _run_server(self):
        """Main server loop - accepts one client at a time."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)

        local_ip = self._get_local_ip()
        print(f"[PicoVRBridge] TCP server listening on {local_ip}:{self.port}")
        print(f"[PicoVRBridge] Configure PICO app to connect to: {local_ip}:{self.port}")

        while self.is_running:
            try:
                client, addr = self._server_socket.accept()
                print(f"[PicoVRBridge] PICO connected from {addr}")
                self._handle_client(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_client(self, client: socket.socket):
        """Handle a connected PICO client."""
        client.settimeout(2.0)
        buffer = ""
        _msg_count = 0

        while self.is_running:
            try:
                data = client.recv(8192)
                if not data:
                    print("[PicoVRBridge] PICO disconnected")
                    break

                buffer += data.decode("utf-8", errors="ignore")
                if _msg_count == 0:
                    print("[PicoVRBridge] First data received from PICO")

                # Process newline-delimited JSON messages
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        self._process_message(msg)
                        _msg_count += 1
                        if _msg_count in (1, 10, 100):
                            print(f"[PicoVRBridge] Processed {_msg_count} messages")
                    except json.JSONDecodeError:
                        pass  # skip malformed messages

            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError, OSError):
                print("[PicoVRBridge] PICO connection lost")
                break

        client.close()

    def _process_message(self, msg: dict):
        """Convert a PICO JSON message into ControlGoal objects."""

        # Process left controller
        if "left" in msg:
            goal = self._process_controller(msg["left"], "left")
            if goal:
                with self._lock:
                    self.left_goal = goal

        # Process right controller
        if "right" in msg:
            goal = self._process_controller(msg["right"], "right")
            if goal:
                with self._lock:
                    self.right_goal = goal

        # Process headset
        if "head" in msg:
            head = msg["head"]
            pos = head.get("pos", [0, 0, 0])
            rot = head.get("rot", [0, 0, 0, 1])
            goal = ControlGoal(
                arm="headset",
                target_position=np.array(pos) * self.vr_to_robot_scale,
                metadata={"quaternion": rot}
            )
            with self._lock:
                self.headset_goal = goal

    def _process_controller(self, ctrl: dict, hand: str) -> Optional[ControlGoal]:
        """Process a single controller's data into a ControlGoal."""
        pos = ctrl.get("pos")
        rot = ctrl.get("rot")
        trigger = ctrl.get("trigger", 0.0)
        grip = ctrl.get("grip", 0.0)
        thumbstick = ctrl.get("thumbstick", [0.0, 0.0])

        if pos is None or rot is None:
            return None

        pos_array = np.array(pos, dtype=float)
        quat = np.array(rot, dtype=float)  # [qx, qy, qz, qw]

        # Track origin quaternion for relative rotation
        if hand == "left":
            was_visible = self._left_was_visible
            self._left_was_visible = True
            if not was_visible or self._left_origin_quat is None:
                self._left_origin_quat = quat.copy()
                # Send reset signal on first appearance
                return ControlGoal(
                    arm="left",
                    mode=ControlMode.POSITION_CONTROL,
                    target_position=None,
                    metadata={"reset_target_to_current": True}
                )
            origin_quat = self._left_origin_quat
        else:
            was_visible = self._right_was_visible
            self._right_was_visible = True
            if not was_visible or self._right_origin_quat is None:
                self._right_origin_quat = quat.copy()
                return ControlGoal(
                    arm="right",
                    mode=ControlMode.POSITION_CONTROL,
                    target_position=None,
                    metadata={"reset_target_to_current": True}
                )
            origin_quat = self._right_origin_quat

        # Compute wrist rotations relative to origin
        wrist_roll = extract_roll_from_quaternion(quat, origin_quat)
        wrist_flex = extract_pitch_from_quaternion(quat, origin_quat)

        # Gripper: trigger > 0.5 → open, else closed (matching XLeVR convention)
        gripper_closed = trigger < 0.5

        scaled_pos = pos_array * self.vr_to_robot_scale

        return ControlGoal(
            arm=hand,
            mode=ControlMode.POSITION_CONTROL,
            target_position=scaled_pos,
            wrist_roll_deg=-wrist_roll,
            wrist_flex_deg=-wrist_flex,
            gripper_closed=gripper_closed,
            metadata={
                "source": "pico_vr",
                "trigger": trigger,
                "grip": grip,
                "thumbstick": thumbstick,
                "vr_position": pos,
                "quaternion": rot,
                "primaryButton": ctrl.get("primaryButton", False),
                "secondaryButton": ctrl.get("secondaryButton", False),
            }
        )

    # -----------------------------------------------------------------------
    # Public API (same interface as XLeVR VRMonitor for drop-in compatibility)
    # -----------------------------------------------------------------------

    def get_latest_goal_nowait(self, arm=None):
        with self._lock:
            if arm == "left":
                return self.left_goal
            elif arm == "right":
                return self.right_goal
            elif arm == "headset":
                return self.headset_goal
            else:
                return {
                    "left": self.left_goal,
                    "right": self.right_goal,
                    "headset": self.headset_goal,
                    "has_left": self.left_goal is not None,
                    "has_right": self.right_goal is not None,
                    "has_headset": self.headset_goal is not None,
                }

    def get_left_goal_nowait(self):
        return self.get_latest_goal_nowait("left")

    def get_right_goal_nowait(self):
        return self.get_latest_goal_nowait("right")

    def reset_origins(self):
        """Reset controller origin quaternions (recalibrate)."""
        with self._lock:
            self._left_origin_quat = None
            self._right_origin_quat = None
            self._left_was_visible = False
            self._right_was_visible = False
        print("[PicoVRBridge] Origins reset - will recalibrate on next frame")

    @staticmethod
    def _get_local_ip():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "localhost"


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PICO VR Bridge for XLeRobot Simulation")
    print("=" * 60)
    bridge = PicoVRBridge(port=9876)
    bridge.start()

    try:
        while True:
            goals = bridge.get_latest_goal_nowait()
            left = goals["left"]
            right = goals["right"]

            status = []
            if left and left.target_position is not None:
                pos = left.target_position
                status.append(f"L:[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]")
            if right and right.target_position is not None:
                pos = right.target_position
                status.append(f"R:[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]")

            if status:
                print(" | ".join(status), end="\r")

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...")
        bridge.stop()


if __name__ == "__main__":
    main()
