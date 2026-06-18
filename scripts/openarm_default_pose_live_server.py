#!/usr/bin/env python3
"""Standalone OpenArm default-pose Isaac mirror.

This script is intentionally separate from the hsi_pregrasp_refusal task code.
It spawns only the OpenArm robot in Isaac Sim, holds the robot at its default
joint pose, and listens for Jetson gripper or joint target commands.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


ISAACLAB_ROOT = Path("/home/chyanin/IsaacLab")
for path in [
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Standalone default-pose OpenArm Isaac mirror.")
parser.add_argument("--host", default="10.10.10.1", help="HTTP command bind host.")
parser.add_argument("--port", type=int, default=8765, help="HTTP command port.")
parser.add_argument("--token", default="", help="Optional token required in X-Bridge-Token or Bearer auth.")
parser.add_argument("--openarm_setup", choices=["bimanual", "unimanual"], default="bimanual")
parser.add_argument("--control_arm", choices=["right", "left"], default="right")
parser.add_argument("--default_gripper_deg", type=float, default=0.0, help="Default real gripper degree target.")
parser.add_argument(
    "--joint_limit_margin_deg",
    type=float,
    default=2.0,
    help="Stay this many degrees inside the real-vs-sim arm joint limits.",
)
parser.add_argument(
    "--gripper_limit_margin_deg",
    type=float,
    default=0.0,
    help="Stay this many degrees inside the real gripper limits. Default keeps exact -65..0 deg.",
)
parser.add_argument("--debug_interval", type=int, default=100, help="Print state every N sim steps. Use 0 to disable.")
parser.add_argument("--sim_dt", type=float, default=0.005)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG, OPENARM_UNI_HIGH_PD_CFG  # noqa: E402


STOP_COMMANDS = {"stop", "hold", "wait", "pause", "stay", "default", "home"}
REAL_ARM_LIMITS_DEG = {
    "right": {
        "joint1": (-75.0, 75.0),
        "joint2": (-9.0, 90.0),
        "joint3": (-85.0, 85.0),
        "joint4": (0.0, 135.0),
        "joint5": (-85.0, 85.0),
        "joint6": (-40.0, 40.0),
        "joint7": (-80.0, 80.0),
    },
    "left": {
        "joint1": (-75.0, 75.0),
        "joint2": (-90.0, 9.0),
        "joint3": (-85.0, 85.0),
        "joint4": (0.0, 135.0),
        "joint5": (-85.0, 85.0),
        "joint6": (-40.0, 40.0),
        "joint7": (-80.0, 80.0),
    },
}
REAL_GRIPPER_LIMIT_DEG = (-65.0, 0.0)


@dataclass
class MirrorCommand:
    command: str = "default"
    mode: str = "default"
    description: str = "hold OpenArm default joint pose"
    gripper_target_deg: float | None = None
    sim_finger_target: float | None = None
    joint_targets_deg: dict[str, float] = field(default_factory=dict)
    sequence: int = 0
    updated_at: float = 0.0
    id: str = ""


class MirrorCommandState:
    """Thread-safe latest-command store."""

    def __init__(self, default_deg: float) -> None:
        self.lock = threading.Lock()
        self.current = MirrorCommand()
        self.default_deg = default_deg
        self.set_command(f"gripper target {default_deg:.3f} deg")

    def set_command(self, command: str) -> MirrorCommand:
        normalized = normalize_command(command)
        mode, description, target_deg, sim_target, joint_targets = resolve_command(normalized, self.default_deg)
        with self.lock:
            self.current = MirrorCommand(
                command=normalized,
                mode=mode,
                description=description,
                gripper_target_deg=target_deg,
                sim_finger_target=sim_target,
                joint_targets_deg=joint_targets,
                sequence=self.current.sequence + 1,
                updated_at=time.time(),
                id=time.strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8],
            )
            return self.current

    def snapshot(self) -> MirrorCommand:
        with self.lock:
            return MirrorCommand(**asdict(self.current))


class RuntimeInfo:
    """Thread-safe runtime information exposed over HTTP."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.limits: dict[str, Any] = {}

    def set_limits(self, limits: dict[str, Any]) -> None:
        with self.lock:
            self.limits = limits

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.limits))


def normalize_command(command: str) -> str:
    normalized = " ".join(str(command).replace("\x00", "").strip().split())
    return normalized or "default"


def extract_gripper_target_deg(command: str) -> float | None:
    text = normalize_command(command).lower().replace("=", " ").replace(",", " ")
    if "gripper" not in text:
        return None
    tokens = text.split()
    for idx, token in enumerate(tokens):
        if token in {"target", "target-deg", "target_deg"}:
            for candidate in tokens[idx + 1 :]:
                try:
                    return float(candidate)
                except ValueError:
                    continue
        if token in {"deg", "degree", "degrees"}:
            for candidate in reversed(tokens[:idx]):
                try:
                    return float(candidate)
                except ValueError:
                    continue
    for token in tokens:
        try:
            return float(token)
        except ValueError:
            continue
    return None


def safe_gripper_limit_deg() -> tuple[float, float]:
    low, high = REAL_GRIPPER_LIMIT_DEG
    margin = max(0.0, float(args_cli.gripper_limit_margin_deg))
    if high - low > 2.0 * margin:
        return low + margin, high - margin
    return low, high


def clamp_gripper_target_deg(target_deg: float) -> float:
    low, high = safe_gripper_limit_deg()
    return min(high, max(low, float(target_deg)))


def real_deg_to_sim_finger_target(target_deg: float) -> float:
    """Map real OpenArm gripper degrees to one simulated finger joint target."""
    clipped = clamp_gripper_target_deg(target_deg)
    return (-clipped / 65.0) * 0.044


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
JOINT_ALIAS_RE = re.compile(
    rf"(?<![A-Za-z0-9_])"
    rf"(?P<alias>"
    rf"openarm_(?:left|right)_joint[1-7]|"
    rf"openarm_joint[1-7]|"
    rf"(?:left|right)[_\s-]*(?:joint|j)[_\s-]*[1-7]|"
    rf"(?:joint|j)[_\s-]*[1-7]"
    rf")"
    rf"(?![A-Za-z0-9_])\s*(?:=|:)?\s*(?P<value>{NUMBER_RE})",
    re.IGNORECASE,
)
JOINT_SIDE_FIRST_RE = re.compile(
    rf"\b(?:joint|j)\s+(?P<side>left|right)\s+(?P<index>[1-7])\s*(?:=|:)?\s*(?P<value>{NUMBER_RE})",
    re.IGNORECASE,
)


def normalize_joint_alias(alias: str) -> str:
    text = re.sub(r"[\s-]+", "_", alias.strip().lower())
    text = re.sub(r"_+", "_", text)

    if re.fullmatch(r"openarm_(?:left|right)_joint[1-7]", text) or re.fullmatch(r"openarm_joint[1-7]", text):
        return text

    match = re.fullmatch(r"(left|right)_?(?:joint|j)_?([1-7])", text)
    if match:
        return f"{match.group(1)}_joint{match.group(2)}"

    match = re.fullmatch(r"(?:joint|j)_?([1-7])", text)
    if match:
        return f"joint{match.group(1)}"

    return text


def parse_joint_targets_deg(command: str) -> dict[str, float]:
    """Extract arm joint degree targets from text commands."""
    text = normalize_command(command)
    targets: dict[str, float] = {}

    for match in JOINT_SIDE_FIRST_RE.finditer(text):
        alias = f"{match.group('side')}_joint{match.group('index')}"
        targets[normalize_joint_alias(alias)] = float(match.group("value"))

    for match in JOINT_ALIAS_RE.finditer(text):
        alias = normalize_joint_alias(match.group("alias"))
        targets[alias] = float(match.group("value"))

    if targets:
        return targets

    lowered = text.lower()
    if "whole body" not in lowered and "whole-body" not in lowered and "all joints" not in lowered:
        return targets

    numbers = [float(value) for value in re.findall(NUMBER_RE, lowered)]
    if len(numbers) == 7:
        side = ""
        if args_cli.openarm_setup == "bimanual":
            if re.search(r"\bleft\b", lowered):
                side = "left"
            elif re.search(r"\bright\b", lowered):
                side = "right"
            else:
                side = args_cli.control_arm
        for index, value in enumerate(numbers, start=1):
            alias = f"{side}_joint{index}" if side else f"joint{index}"
            targets[normalize_joint_alias(alias)] = value
    elif args_cli.openarm_setup == "bimanual" and len(numbers) == 14:
        for index, value in enumerate(numbers[:7], start=1):
            targets[f"left_joint{index}"] = value
        for index, value in enumerate(numbers[7:], start=1):
            targets[f"right_joint{index}"] = value

    return targets


def resolve_command(
    command: str,
    default_deg: float,
) -> tuple[str, str, float | None, float | None, dict[str, float]]:
    text = normalize_command(command).lower()
    if text in STOP_COMMANDS:
        sim_target = real_deg_to_sim_finger_target(default_deg)
        return "default", f"hold default pose and default gripper {default_deg:.3f} deg", default_deg, sim_target, {}

    joint_targets = parse_joint_targets_deg(command)
    target_deg = extract_gripper_target_deg(command)
    if joint_targets:
        if target_deg is not None:
            sim_target = real_deg_to_sim_finger_target(target_deg)
            return (
                "joint_and_gripper_targets",
                f"mirror {len(joint_targets)} joint target(s) and gripper target {target_deg:.3f} deg",
                target_deg,
                sim_target,
                joint_targets,
            )
        return (
            "joint_targets",
            f"mirror {len(joint_targets)} joint target(s)",
            None,
            None,
            joint_targets,
        )

    if target_deg is not None:
        sim_target = real_deg_to_sim_finger_target(target_deg)
        return "gripper_target", f"mirror gripper target {target_deg:.3f} deg", target_deg, sim_target, {}
    if "open" in text and "gripper" in text:
        sim_target = real_deg_to_sim_finger_target(-65.0)
        return "gripper_target", "open gripper", -65.0, sim_target, {}
    if "close" in text and "gripper" in text:
        sim_target = real_deg_to_sim_finger_target(0.0)
        return "gripper_target", "close gripper", 0.0, sim_target, {}
    sim_target = real_deg_to_sim_finger_target(default_deg)
    return "default", f"unknown command; hold default pose and default gripper {default_deg:.3f} deg", default_deg, sim_target, {}


def json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(command_state: MirrorCommandState):
    class DefaultPoseHandler(BaseHTTPRequestHandler):
        server_version = "OpenArmDefaultPoseMirror/1.0"

        def _authorized(self) -> bool:
            if not args_cli.token:
                return True
            auth = self.headers.get("Authorization", "")
            header = self.headers.get("X-Bridge-Token", "")
            return header == args_cli.token or auth == f"Bearer {args_cli.token}"

        def _reject_unauthorized(self) -> bool:
            if self._authorized():
                return False
            json_response(self, 401, {"ok": False, "error": "Missing or invalid bridge token."})
            return True

        def _read_payload(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(content_length) if content_length else b""
            if not raw:
                return {}
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return json.loads(raw.decode("utf-8"))
            parsed = parse_qs(raw.decode("utf-8"))
            return {key: values[-1] for key, values in parsed.items()}

        def do_GET(self) -> None:  # noqa: N802
            if self._reject_unauthorized():
                return
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/health"}:
                json_response(self, 200, {"ok": True, "service": "openarm-default-pose-mirror"})
                return
            if parsed.path == "/status":
                json_response(self, 200, {"ok": True, "live": True, "current_command": asdict(command_state.snapshot())})
                return
            if parsed.path == "/command":
                command = parse_qs(parsed.query).get("command", ["default"])[-1]
                current = command_state.set_command(command)
                json_response(self, 202, {"ok": True, "live": True, "accepted": asdict(current)})
                return
            json_response(self, 404, {"ok": False, "error": f"Unknown route: {parsed.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self._reject_unauthorized():
                return
            parsed = urlparse(self.path)
            if parsed.path == "/stop":
                current = command_state.set_command("default")
                json_response(self, 200, {"ok": True, "live": True, "accepted": asdict(current)})
                return
            if parsed.path == "/command":
                try:
                    payload = self._read_payload()
                except Exception as exc:
                    json_response(self, 400, {"ok": False, "error": f"Invalid request body: {exc}"})
                    return
                current = command_state.set_command(str(payload.get("command", "default")))
                json_response(self, 202, {"ok": True, "live": True, "accepted": asdict(current)})
                return
            json_response(self, 404, {"ok": False, "error": f"Unknown route: {parsed.path}"})

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return DefaultPoseHandler


def start_http_server(command_state: MirrorCommandState) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((args_cli.host, args_cli.port), make_handler(command_state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[DEFAULT OPENARM] listening on http://{args_cli.host}:{args_cli.port}", flush=True)
    return server


def openarm_robot_cfg() -> ArticulationCfg:
    if args_cli.openarm_setup == "bimanual":
        robot_cfg = OPENARM_BI_HIGH_PD_CFG.copy()
        robot_cfg.init_state.joint_pos = {
            "openarm_left_joint.*": 0.0,
            "openarm_right_joint.*": 0.0,
            "openarm_left_finger_joint.*": 0.0,
            "openarm_right_finger_joint.*": 0.0,
        }
    else:
        robot_cfg = OPENARM_UNI_HIGH_PD_CFG.copy()
        robot_cfg.init_state.joint_pos = {
            "openarm_joint1": 1.57,
            "openarm_joint2": 0.0,
            "openarm_joint3": -1.57,
            "openarm_joint4": 1.57,
            "openarm_joint5": 0.0,
            "openarm_joint6": 0.0,
            "openarm_joint7": 0.0,
            "openarm_finger_joint.*": 0.0,
        }
    return robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


ROBOT_CFG = openarm_robot_cfg()


@configclass
class DefaultOpenArmSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground, light, robot only."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = ROBOT_CFG


def controlled_finger_expr() -> str:
    if args_cli.openarm_setup == "bimanual":
        return f"openarm_{args_cli.control_arm}_finger_joint.*"
    return "openarm_finger_joint.*"


def canonical_joint_name(alias: str) -> str:
    normalized = normalize_joint_alias(alias)
    if re.fullmatch(r"openarm_(?:left|right)_joint[1-7]", normalized) or re.fullmatch(
        r"openarm_joint[1-7]", normalized
    ):
        return normalized

    match = re.fullmatch(r"(left|right)_joint([1-7])", normalized)
    if match:
        if args_cli.openarm_setup == "bimanual":
            return f"openarm_{match.group(1)}_joint{match.group(2)}"
        return f"openarm_joint{match.group(2)}"

    match = re.fullmatch(r"joint([1-7])", normalized)
    if match:
        if args_cli.openarm_setup == "bimanual":
            return f"openarm_{args_cli.control_arm}_joint{match.group(1)}"
        return f"openarm_joint{match.group(1)}"

    return normalized


def real_limit_key_for_joint(joint_name: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"openarm_(left|right)_joint([1-7])", joint_name)
    if match:
        return match.group(1), f"joint{match.group(2)}"
    match = re.fullmatch(r"openarm_joint([1-7])", joint_name)
    if match:
        return args_cli.control_arm, f"joint{match.group(1)}"
    return None


def safe_real_joint_limit_rad(joint_name: str) -> tuple[float, float] | None:
    key = real_limit_key_for_joint(joint_name)
    if key is None:
        return None
    side, joint_key = key
    low_deg, high_deg = REAL_ARM_LIMITS_DEG[side][joint_key]
    margin = max(0.0, float(args_cli.joint_limit_margin_deg))
    if high_deg - low_deg > 2.0 * margin:
        low_deg += margin
        high_deg -= margin
    return math.radians(low_deg), math.radians(high_deg)


def reset_default_joint_targets(
    robot,
    target_joint_pos: torch.Tensor,
    finger_ids: list[int],
    default_sim_target: float,
) -> None:
    target_joint_pos[:] = robot.data.default_joint_pos
    target_joint_pos[:, finger_ids] = default_sim_target


def apply_joint_targets_deg(robot, target_joint_pos: torch.Tensor, joint_targets_deg: dict[str, float]) -> tuple[dict, dict]:
    applied: dict[str, dict[str, float]] = {}
    skipped: dict[str, str] = {}
    limits = robot.data.soft_joint_pos_limits

    for alias, target_deg in sorted(joint_targets_deg.items()):
        joint_name = canonical_joint_name(alias)
        joint_ids, joint_names = robot.find_joints([joint_name])
        if len(joint_ids) == 0:
            skipped[alias] = f"joint not found: {joint_name}"
            continue

        joint_id = int(joint_ids[0])
        raw_target_rad = math.radians(float(target_deg))
        sim_low = float(limits[0, joint_id, 0].item())
        sim_high = float(limits[0, joint_id, 1].item())
        real_limit = safe_real_joint_limit_rad(joint_name)
        if real_limit is None:
            low, high = sim_low, sim_high
        else:
            real_low, real_high = real_limit
            low = max(sim_low, real_low)
            high = min(sim_high, real_high)
        if low > high:
            skipped[alias] = (
                f"no overlapping safe limit for {joint_name}: "
                f"sim {math.degrees(sim_low):.3f}..{math.degrees(sim_high):.3f} deg"
            )
            continue
        target_rad = min(max(raw_target_rad, low), high)
        target_joint_pos[:, joint_id] = target_rad

        applied[str(joint_names[0])] = {
            "requested_deg": float(target_deg),
            "target_deg": math.degrees(target_rad),
            "target_rad": target_rad,
            "safe_limit_deg": [math.degrees(low), math.degrees(high)],
        }

    return applied, skipped


def main() -> None:
    command_state = MirrorCommandState(args_cli.default_gripper_deg)
    http_server = start_http_server(command_state)

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    if args_cli.openarm_setup == "bimanual":
        sim.set_camera_view(eye=[2.1, -1.8, 1.3], target=[0.0, 0.0, 0.35])
    else:
        sim.set_camera_view(eye=[1.8, -1.4, 1.2], target=[0.0, 0.0, 0.35])

    scene = InteractiveScene(DefaultOpenArmSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    finger_ids, finger_names = robot.find_joints([controlled_finger_expr()])
    if len(finger_ids) == 0:
        raise RuntimeError(f"No finger joints found for expression {controlled_finger_expr()!r}.")
    finger_ids = [int(joint_id) for joint_id in finger_ids]

    target_joint_pos = robot.data.default_joint_pos.clone()
    target_joint_vel = robot.data.default_joint_vel.clone()
    default_sim_target = real_deg_to_sim_finger_target(args_cli.default_gripper_deg)
    reset_default_joint_targets(robot, target_joint_pos, finger_ids, default_sim_target)
    robot.write_joint_state_to_sim(target_joint_pos, target_joint_vel)
    scene.reset()

    print(
        "[DEFAULT OPENARM] viewer ready; "
        f"setup={args_cli.openarm_setup} control_arm={args_cli.control_arm}; "
        f"default_gripper_deg={args_cli.default_gripper_deg:+.3f}; "
        f"finger_joints={list(finger_names)}",
        flush=True,
    )

    last_sequence = -1
    step = 0
    sim_dt = sim.get_physics_dt()
    try:
        while simulation_app.is_running():
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.step()
                continue

            current = command_state.snapshot()
            if current.sequence != last_sequence:
                if current.mode == "default":
                    reset_default_joint_targets(robot, target_joint_pos, finger_ids, default_sim_target)

                if current.sim_finger_target is not None:
                    target_joint_pos[:, finger_ids] = float(current.sim_finger_target)

                applied, skipped = apply_joint_targets_deg(robot, target_joint_pos, current.joint_targets_deg)
                target_text = (
                    f"{current.gripper_target_deg:+.3f}" if current.gripper_target_deg is not None else "unchanged"
                )
                print(
                    "[DEFAULT OPENARM] "
                    f"accepted command={current.command!r} mode={current.mode} "
                    f"gripper_deg={target_text} "
                    f"sim_finger_target={float(current.sim_finger_target or target_joint_pos[0, finger_ids[0]].item()):.4f} "
                    f"applied_joints={applied} skipped_joints={skipped}",
                    flush=True,
                )
                last_sequence = current.sequence

            robot.set_joint_position_target(target_joint_pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            step += 1

            if args_cli.debug_interval > 0 and step % args_cli.debug_interval == 0:
                width = robot.data.joint_pos[:, finger_ids].sum(dim=-1)
                print(
                    "[DEFAULT OPENARM] "
                    f"step={step} mode={current.mode} gripper_width={float(width[0].item()):.4f}",
                    flush=True,
                )
    finally:
        http_server.shutdown()
        http_server.server_close()


if __name__ == "__main__":
    main()
    simulation_app.close()
