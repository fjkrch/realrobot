#!/usr/bin/env python3
"""Replay one saved upsampled OpenArm height episode on the real robot.

Default mode is dry-run. Real motor commands require:

  --no-dry-run --confirm-real-hardware --confirm-height

This runner never recollects data, never edits NPZ files, never interpolates or
clamps the saved trajectory, and never moves to the first pose for the operator.
The real robot must already be close to the first saved command before replay.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
from typing import Any
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]

JOINT_NAMES = [f"joint_{index}" for index in range(1, 8)]
ALL_MOTORS = JOINT_NAMES + ["gripper"]
SUPPORTED_HEIGHTS_10HZ = ("125", "122.5", "120", "117.5", "115")
SUPPORTED_HEIGHTS_20HZ400 = ("112.5", "110", "107.5", "105", "102.25", "100")
SUPPORTED_HEIGHTS = SUPPORTED_HEIGHTS_10HZ + SUPPORTED_HEIGHTS_20HZ400

HEIGHT_DATASET_DIR_ALIASES_10HZ = {
    "125": ("h125cm_upsampled",),
    # The prompt used underscore decimal names; the checked-in datasets use p.
    "122.5": ("h122p5cm_upsampled", "h122_5cm_upsampled"),
    "120": ("h120cm_upsampled",),
    "117.5": ("h117p5cm_upsampled", "h117_5cm_upsampled"),
    "115": ("h115cm_upsampled",),
}

HEIGHT_DATASET_DIR_ALIASES_20HZ400 = {
    "112.5": ("h112p5cm_upsampled", "h112_5cm_upsampled"),
    "110": ("h110cm_upsampled",),
    "107.5": ("h107p5cm_upsampled", "h107_5cm_upsampled"),
    "105": ("h105cm_upsampled",),
    "102.25": ("h102p25cm_upsampled", "h102_25cm_upsampled"),
    "100": ("h100cm_upsampled",),
}

DATASET_ROOT_10HZ = REPO_ROOT / "synthetic_smolvla" / "datasets" / "openarm_photo_clean_v1_one_per_height"
DATASET_ROOT_20HZ400 = (
    REPO_ROOT / "synthetic_smolvla" / "datasets" / "openarm_photo_clean_v1_one_per_height_20hz400"
)
# Backward-compatible alias used by tests and older callers.
DATASET_ROOT = DATASET_ROOT_10HZ

MAX_STEP_DEG = 2.0
MAX_SPEED_DEG_S = 20.0
DEFAULT_RATE_HZ = 10.0

MAX_STEP_DEG_20HZ400 = 1.5
MAX_SPEED_DEG_S_20HZ400 = 30.0
DEFAULT_RATE_HZ_20HZ400 = 20.0
EXPECTED_COMMANDS_20HZ400 = 400

KNOWN_REAL_LIMITS_DEG = {
    "right": {
        "joint_1": (-75.0, 75.0),
        "joint_2": (-9.0, 90.0),
        "joint_3": (-85.0, 85.0),
        "joint_4": (0.0, 135.0),
        "joint_5": (-85.0, 85.0),
        "joint_6": (-40.0, 40.0),
        "joint_7": (-80.0, 80.0),
        "gripper": (-65.0, 0.0),
    },
    "left": {
        "joint_1": (-75.0, 75.0),
        "joint_2": (-90.0, 9.0),
        "joint_3": (-85.0, 85.0),
        "joint_4": (0.0, 135.0),
        "joint_5": (-85.0, 85.0),
        "joint_6": (-40.0, 40.0),
        "joint_7": (-80.0, 80.0),
        "gripper": (-65.0, 0.0),
    },
}

STOP_REQUESTED = False


class ReplaySafetyError(RuntimeError):
    """Raised when replay must refuse or abort for safety."""


class EmergencyStop(RuntimeError):
    """Raised when the software stop path is requested."""


@dataclass(frozen=True)
class ReplayProfile:
    name: str
    dataset_root: Path
    height_aliases: dict[str, tuple[str, ...]]
    default_rate_hz: float
    max_step_deg: float
    max_speed_deg_s: float
    expected_commands: int | None = None


REPLAY_PROFILES = {
    "10hz": ReplayProfile(
        name="10hz",
        dataset_root=DATASET_ROOT_10HZ,
        height_aliases=HEIGHT_DATASET_DIR_ALIASES_10HZ,
        default_rate_hz=DEFAULT_RATE_HZ,
        max_step_deg=MAX_STEP_DEG,
        max_speed_deg_s=MAX_SPEED_DEG_S,
    ),
    "20hz400": ReplayProfile(
        name="20hz400",
        dataset_root=DATASET_ROOT_20HZ400,
        height_aliases=HEIGHT_DATASET_DIR_ALIASES_20HZ400,
        default_rate_hz=DEFAULT_RATE_HZ_20HZ400,
        max_step_deg=MAX_STEP_DEG_20HZ400,
        max_speed_deg_s=MAX_SPEED_DEG_S_20HZ400,
        expected_commands=EXPECTED_COMMANDS_20HZ400,
    ),
}


@dataclass(frozen=True)
class EpisodeAudit:
    episode: Path
    height: str
    dataset_family: str
    commands: int
    rate_hz: float
    max_step_deg: float
    max_speed_deg_s: float
    max_allowed_step_deg: float
    max_allowed_speed_deg_s: float
    expected_duration_sec: float
    expected_duration_10hz_sec: float


def _request_stop(signum: int, _frame: Any) -> None:
    del _frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    name = signal.Signals(signum).name
    print(f"\n[replay] {name} received; aborting and disconnecting.", file=sys.stderr, flush=True)


def install_stop_handlers() -> None:
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


def default_port_for_side(side: str) -> str:
    return "can0" if side == "right" else "can1"


def normalize_height(raw: str) -> str:
    text = str(raw).strip().replace("_", ".")
    for height in SUPPORTED_HEIGHTS:
        if text == height:
            return height
    try:
        value = float(text)
    except ValueError as exc:
        raise ReplaySafetyError(f"Unsupported --height {raw!r}; choose {', '.join(SUPPORTED_HEIGHTS)}.") from exc
    for height in SUPPORTED_HEIGHTS:
        if math.isclose(value, float(height), abs_tol=1e-9):
            return height
    raise ReplaySafetyError(f"Unsupported --height {raw!r}; choose {', '.join(SUPPORTED_HEIGHTS)}.")


def select_replay_profile(height: str, *, dataset_family: str = "auto", rate_hz: float | None = None) -> ReplayProfile:
    normalized = normalize_height(height)
    if dataset_family != "auto":
        if dataset_family not in REPLAY_PROFILES:
            raise ReplaySafetyError(f"Unsupported --dataset-family {dataset_family!r}.")
        profile = REPLAY_PROFILES[dataset_family]
        if normalized not in profile.height_aliases:
            supported = ", ".join(profile.height_aliases)
            raise ReplaySafetyError(
                f"--height {height} is not available in dataset family {dataset_family!r}; choose {supported}."
            )
        return profile

    candidates = [profile for profile in REPLAY_PROFILES.values() if normalized in profile.height_aliases]
    if not candidates:
        raise ReplaySafetyError(f"Unsupported --height {height!r}; choose {', '.join(SUPPORTED_HEIGHTS)}.")
    if len(candidates) == 1:
        return candidates[0]
    if rate_hz is not None:
        for profile in candidates:
            if math.isclose(float(rate_hz), profile.default_rate_hz, rel_tol=0.0, abs_tol=1e-9):
                return profile
    return REPLAY_PROFILES["10hz"]


def apply_profile_defaults(args: argparse.Namespace, profile: ReplayProfile) -> None:
    if args.rate_hz is None:
        args.rate_hz = profile.default_rate_hz
    if args.max_relative_target_deg is None:
        args.max_relative_target_deg = profile.max_step_deg


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def episode_path_for_height(
    height: str,
    *,
    dataset_root: str | Path | None = None,
    dataset_family: str = "10hz",
) -> Path:
    normalized = normalize_height(height)
    profile = select_replay_profile(normalized, dataset_family=dataset_family)
    root = resolve_path(dataset_root) if dataset_root is not None else profile.dataset_root
    candidates = [
        root / dirname / "episodes" / "episode_000000.npz"
        for dirname in profile.height_aliases[normalized]
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n  ".join(str(path) for path in candidates)
    raise ReplaySafetyError(f"No saved upsampled episode found for --height {height}. Tried:\n  {rendered}")


def require_upsampled_episode(path: Path) -> None:
    if path.name != "episode_000000.npz" or path.parent.name != "episodes":
        raise ReplaySafetyError(f"Refusing non-canonical episode path: {path}")
    if not path.parent.parent.name.endswith("_upsampled"):
        raise ReplaySafetyError(f"Refusing non-upsampled dataset path: {path}")


def require_can_interface(port: str) -> None:
    result = subprocess.run(
        ["ip", "link", "show", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise ReplaySafetyError(f"{port} does not exist. Start CAN first.")


def _load_action_with_numpy(path: Path) -> list[list[float]] | None:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    with np.load(path, allow_pickle=False) as data:
        if "action" not in data.files:
            raise ReplaySafetyError(f"{path} has no NPZ key 'action'.")
        action = np.asarray(data["action"], dtype=np.float64)
    if action.ndim != 2:
        raise ReplaySafetyError(f"action must be rank-2 [T, 8], got shape {tuple(action.shape)}.")
    return [[float(value) for value in row] for row in action.tolist()]


def _parse_npy_payload(payload: bytes) -> tuple[tuple[int, ...], list[float]]:
    if not payload.startswith(b"\x93NUMPY"):
        raise ReplaySafetyError("action.npy inside NPZ is not a valid NPY array.")
    major = payload[6]
    if major == 1:
        header_len = struct.unpack("<H", payload[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", payload[8:12])[0]
        header_start = 12
    else:
        raise ReplaySafetyError(f"Unsupported NPY version {major}.{payload[7]}.")
    header = ast.literal_eval(payload[header_start : header_start + header_len].decode("latin1"))
    if header.get("fortran_order"):
        raise ReplaySafetyError("action.npy is Fortran-ordered; refusing fallback parse.")
    shape = tuple(int(dim) for dim in header["shape"])
    descr = str(header["descr"])
    dtype_code = descr[-2:]
    if dtype_code not in {"f4", "f8", "i4", "i8"}:
        raise ReplaySafetyError(f"Unsupported action dtype {descr!r}; install numpy to load this file.")
    endian = "<" if descr[0] in {"<", "|"} else ">"
    fmt_by_code = {"f4": "f", "f8": "d", "i4": "i", "i8": "q"}
    count = math.prod(shape)
    fmt = endian + fmt_by_code[dtype_code] * count
    data_start = header_start + header_len
    needed = struct.calcsize(fmt)
    raw = payload[data_start : data_start + needed]
    if len(raw) != needed:
        raise ReplaySafetyError("action.npy ended before the declared array payload.")
    values = [float(value) for value in struct.unpack(fmt, raw)]
    return shape, values


def _load_action_without_numpy(path: Path) -> list[list[float]]:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        if "action.npy" not in names:
            raise ReplaySafetyError(f"{path} has no NPZ key 'action'.")
        shape, values = _parse_npy_payload(archive.read("action.npy"))
    if len(shape) != 2:
        raise ReplaySafetyError(f"action must be rank-2 [T, 8], got shape {shape}.")
    rows, cols = shape
    return [values[index * cols : (index + 1) * cols] for index in range(rows)]


def load_action_rows(path: Path) -> list[list[float]]:
    require_upsampled_episode(path)
    rows = _load_action_with_numpy(path)
    if rows is not None:
        return rows
    return _load_action_without_numpy(path)


def max_abs_step_deg(rows: list[list[float]]) -> float:
    if len(rows) < 2:
        return 0.0
    return max(
        abs(float(rows[index][col]) - float(rows[index - 1][col]))
        for index in range(1, len(rows))
        for col in range(8)
    )


def validate_limits(
    rows: list[list[float]],
    limits: dict[str, tuple[float, float]],
    *,
    label: str,
) -> None:
    missing = [name for name in ALL_MOTORS if name not in limits]
    if missing:
        raise ReplaySafetyError(f"{label} joint limits missing {missing}.")
    errors = []
    for row_index, row in enumerate(rows):
        for col, name in enumerate(ALL_MOTORS):
            low, high = limits[name]
            value = float(row[col])
            if value < low or value > high:
                errors.append(
                    f"row {row_index} {name}={value:.6f} outside {label} limit {low:.3f}..{high:.3f}"
                )
                if len(errors) >= 12:
                    break
        if len(errors) >= 12:
            break
    if errors:
        raise ReplaySafetyError("Refusing out-of-limit saved commands:\n  " + "\n  ".join(errors))


def validate_action_contract(
    rows: list[list[float]],
    *,
    episode: Path,
    height: str,
    profile: ReplayProfile | None = None,
    side: str,
    rate_hz: float,
    first_zero_tolerance_deg: float,
) -> EpisodeAudit:
    profile = profile or REPLAY_PROFILES["10hz"]
    if rate_hz <= 0.0:
        raise ReplaySafetyError("--rate-hz must be positive.")
    if not rows:
        raise ReplaySafetyError("action must contain at least one command.")
    if profile.expected_commands is not None and len(rows) != profile.expected_commands:
        raise ReplaySafetyError(
            f"{profile.name} replay requires exactly {profile.expected_commands} saved commands; "
            f"got {len(rows)}."
        )
    for row_index, row in enumerate(rows):
        if len(row) != 8:
            raise ReplaySafetyError(f"action.shape must be [T, 8]; row {row_index} has {len(row)} columns.")
        for col, value in enumerate(row):
            if not math.isfinite(float(value)):
                raise ReplaySafetyError(f"action[{row_index}, {col}] is not finite: {value!r}")

    first_max = max(abs(float(value)) for value in rows[0])
    if first_max > first_zero_tolerance_deg:
        raise ReplaySafetyError(
            f"First command is not near all zeros: max abs {first_max:.6f} deg "
            f"> tolerance {first_zero_tolerance_deg:.6f} deg."
        )

    observed_step = max_abs_step_deg(rows)
    if observed_step > profile.max_step_deg + 1e-6:
        raise ReplaySafetyError(
            f"Saved trajectory exceeds upsample slew cap: max(abs(diff(action)))="
            f"{observed_step:.6f} deg > {profile.max_step_deg:.6f} deg."
        )

    observed_speed = observed_step * rate_hz
    if observed_speed > profile.max_speed_deg_s + 1e-6:
        raise ReplaySafetyError(
            f"Saved trajectory exceeds {profile.max_speed_deg_s:.1f} deg/s at {rate_hz:.3f} Hz: "
            f"{observed_speed:.6f} deg/s."
        )

    validate_limits(rows, KNOWN_REAL_LIMITS_DEG[side], label=f"documented {side} real")
    commands = len(rows)
    return EpisodeAudit(
        episode=episode,
        height=height,
        dataset_family=profile.name,
        commands=commands,
        rate_hz=rate_hz,
        max_step_deg=observed_step,
        max_speed_deg_s=observed_speed,
        max_allowed_step_deg=profile.max_step_deg,
        max_allowed_speed_deg_s=profile.max_speed_deg_s,
        expected_duration_sec=commands / rate_hz,
        expected_duration_10hz_sec=commands / DEFAULT_RATE_HZ,
    )


def format_command(row: list[float]) -> str:
    values = ", ".join(f"{float(value):+.3f}" for value in row)
    return f"[{values}]"


def print_command_preview(rows: list[list[float]]) -> None:
    middle_index = len(rows) // 2
    for label, index in (("first", 0), ("middle", middle_index), ("last", len(rows) - 1)):
        print(f"[preview] {label} command row {index}: {format_command(rows[index])}", flush=True)


def print_audit(audit: EpisodeAudit, *, side: str, dry_run: bool) -> None:
    payload = {
        "ok": True,
        "mode": "dry-run" if dry_run else "real",
        "height_cm": audit.height,
        "dataset_family": audit.dataset_family,
        "episode": str(audit.episode),
        "side": side,
        "commands": audit.commands,
        "rate_hz": audit.rate_hz,
        "max_abs_diff_action_deg": round(audit.max_step_deg, 6),
        "max_speed_deg_s_at_rate": round(audit.max_speed_deg_s, 6),
        "max_allowed_step_deg": round(audit.max_allowed_step_deg, 6),
        "max_allowed_speed_deg_s": round(audit.max_allowed_speed_deg_s, 6),
        "expected_duration_sec": round(audit.expected_duration_sec, 6),
        "expected_duration_10hz_sec": round(audit.expected_duration_10hz_sec, 6),
        "uses_upsampled_dataset": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def close_enough_to_first(
    state: list[float],
    first: list[float],
    *,
    tolerance_deg: float,
) -> tuple[bool, dict[str, float]]:
    errors = {
        ALL_MOTORS[index]: round(abs(float(state[index]) - float(first[index])), 6)
        for index in range(8)
    }
    return max(errors.values()) <= tolerance_deg, errors


class DryRunSink:
    def send_arm(self, row_index: int, arm_deg: list[float]) -> None:
        print(
            f"[dry-run] row {row_index:04d} arm targets deg: "
            + ", ".join(f"{name}={float(value):+.3f}" for name, value in zip(JOINT_NAMES, arm_deg, strict=True)),
            flush=True,
        )

    def send_gripper(self, row_index: int, gripper_deg: float) -> None:
        print(f"[dry-run] row {row_index:04d} gripper target deg: {float(gripper_deg):+.3f}", flush=True)

    def close(self) -> None:
        return None


class RealOpenArmSink:
    def __init__(self, args: argparse.Namespace, *, port: str) -> None:
        self.args = args
        self.port = port
        self.robot = None

    def _build_robot(self):
        from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

        return OpenArmFollower(
            OpenArmFollowerConfig(
                port=self.port,
                side=self.args.side,
                id=self.args.id,
                # LeRobot checks dict max_relative_target keys against each
                # send_action() subset. A scalar applies the same cap to the
                # 7-joint arm send and the later gripper-only send.
                max_relative_target=float(self.args.max_relative_target_deg),
            )
        )

    def validate_live_limits_before_connect(self, rows: list[list[float]]) -> None:
        probe = self._build_robot()
        joint_limits = getattr(probe.config, "joint_limits", {}) or {}
        validate_limits(rows, joint_limits, label="live OpenArm")
        if probe.is_calibrated:
            raise ReplaySafetyError(f"Calibration exists for id={self.args.id!r}; use a fresh id.")

    def connect(self) -> None:
        last_exc = None
        for attempt in range(1, self.args.connect_retries + 1):
            robot = self._build_robot()
            try:
                robot.connect(calibrate=False)
                self.robot = robot
                if attempt > 1:
                    print(f"[real] connected on attempt {attempt}.", file=sys.stderr, flush=True)
                return
            except Exception as exc:  # noqa: BLE001 - hardware dependency exceptions vary
                last_exc = exc
                print(
                    f"[real] connect attempt {attempt}/{self.args.connect_retries} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    robot.disconnect()
                except Exception:
                    pass
                if attempt < self.args.connect_retries:
                    time.sleep(self.args.connect_retry_delay_sec)
        raise ReplaySafetyError(
            f"Could not connect after {self.args.connect_retries} attempts. Last error: {last_exc}"
        )

    def read_state_deg(self) -> list[float]:
        if self.robot is None:
            raise ReplaySafetyError("Robot is not connected.")
        obs = self.robot.get_observation()
        missing = [f"{name}.pos" for name in ALL_MOTORS if f"{name}.pos" not in obs]
        if missing:
            available = sorted(key for key in obs if key.endswith(".pos"))
            raise ReplaySafetyError(f"Observation missing {missing}. Available position keys: {available}")
        return [float(obs[f"{name}.pos"]) for name in ALL_MOTORS]

    def assert_current_pose_matches_first(self, first: list[float]) -> None:
        state = self.read_state_deg()
        ok, errors = close_enough_to_first(state, first, tolerance_deg=self.args.start_tolerance_deg)
        if not ok:
            worst = max(errors, key=errors.get)
            raise ReplaySafetyError(
                "Current real arm pose is not close to the first saved command; refusing to jump. "
                f"Tolerance={self.args.start_tolerance_deg:.3f} deg, worst={worst} "
                f"error={errors[worst]:.6f} deg, state={format_command(state)}, "
                f"first={format_command(first)}. Place the robot at the first command manually."
            )
        print(
            "[real] current pose matches first saved command within "
            f"{self.args.start_tolerance_deg:.3f} deg.",
            file=sys.stderr,
            flush=True,
        )

    def _check_connected(self) -> None:
        if self.robot is None:
            raise ReplaySafetyError("Robot is not connected.")

    def send_arm(self, row_index: int, arm_deg: list[float]) -> None:
        del row_index
        self._check_connected()
        action = {f"{name}.pos": float(arm_deg[index]) for index, name in enumerate(JOINT_NAMES)}
        self.robot.send_action(action)

    def send_gripper(self, row_index: int, gripper_deg: float) -> None:
        del row_index
        self._check_connected()
        self.robot.send_action({"gripper.pos": float(gripper_deg)})

    def close(self) -> None:
        if self.robot is not None:
            try:
                self.robot.disconnect()
                print("[real] disconnected; torque should be off.", file=sys.stderr, flush=True)
            finally:
                self.robot = None


def replay_rows(
    rows: list[list[float]],
    *,
    sink: DryRunSink | RealOpenArmSink,
    rate_hz: float,
    sleep_enabled: bool,
) -> float:
    period = 1.0 / rate_hz
    replay_start = time.perf_counter()
    for row_index, row in enumerate(rows):
        if STOP_REQUESTED:
            raise EmergencyStop("software stop requested before row send")
        command_start = time.perf_counter()
        sink.send_arm(row_index, [float(value) for value in row[:7]])
        if STOP_REQUESTED:
            raise EmergencyStop("software stop requested after arm send")
        sink.send_gripper(row_index, float(row[7]))
        remaining = period - (time.perf_counter() - command_start)
        if sleep_enabled and remaining > 0.0:
            time.sleep(remaining)
    return time.perf_counter() - replay_start


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", required=True, help="one of: " + ", ".join(SUPPORTED_HEIGHTS))
    parser.add_argument(
        "--dataset-family",
        choices=["auto", *REPLAY_PROFILES],
        default="auto",
        help="auto chooses 10hz for original heights and 20hz400 for lower-height 400-step episodes",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="default true; pass --no-dry-run only for real hardware replay",
    )
    parser.add_argument(
        "--confirm-real-hardware",
        action="store_true",
        help="required with --no-dry-run before any motor command can be sent",
    )
    parser.add_argument(
        "--confirm-height",
        action="store_true",
        help="required with --no-dry-run to confirm robot/table are set to --height",
    )
    parser.add_argument("--rate-hz", type=float, default=None, help="defaults to the selected dataset family rate")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="optional override root containing hTAG_upsampled/... folders; defaults to the selected dataset family",
    )
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--port", default=None, help="CAN interface; defaults to can1 for left, can0 for right")
    parser.add_argument("--id", default="saved_episode_replay_noncal", help="fresh OpenArm id with no calibration file")
    parser.add_argument("--start-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--first-command-zero-tolerance-deg", type=float, default=0.25)
    parser.add_argument("--max-relative-target-deg", type=float, default=None)
    parser.add_argument("--connect-retries", type=int, default=3)
    parser.add_argument("--connect-retry-delay-sec", type=float, default=1.5)
    parser.add_argument(
        "--dry-run-no-sleep",
        action="store_true",
        help="dry-run audit loop without wall-clock sleeps; refused with --no-dry-run",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the final interactive REPLAY prompt; confirmation flags are still required",
    )
    return parser


def validate_real_flags(args: argparse.Namespace, profile: ReplayProfile | None = None) -> None:
    profile = profile or REPLAY_PROFILES["10hz"]
    if args.dry_run:
        return
    if args.dry_run_no_sleep:
        raise ReplaySafetyError("--dry-run-no-sleep is only allowed in dry-run mode.")
    if not args.confirm_real_hardware:
        raise ReplaySafetyError("Refusing real replay. Add --confirm-real-hardware when physically at the robot.")
    if not args.confirm_height:
        raise ReplaySafetyError(
            f"Refusing real replay. Set the robot/table to {args.height} cm and add --confirm-height."
        )
    if not math.isclose(float(args.rate_hz), profile.default_rate_hz, rel_tol=0.0, abs_tol=1e-9):
        raise ReplaySafetyError(
            f"Real replay for {profile.name} is fixed at {profile.default_rate_hz:.1f} Hz to match Isaac timing; "
            f"got --rate-hz {float(args.rate_hz):.6f}."
        )
    if args.start_tolerance_deg <= 0.0:
        raise ReplaySafetyError("--start-tolerance-deg must be positive.")
    if args.max_relative_target_deg <= 0.0:
        raise ReplaySafetyError("--max-relative-target-deg must be positive.")
    if args.max_relative_target_deg > profile.max_step_deg + 1e-9:
        raise ReplaySafetyError(
            f"--max-relative-target-deg {args.max_relative_target_deg:.6f} exceeds "
            f"the {profile.name} saved-command cap {profile.max_step_deg:.6f} deg."
        )


def final_interactive_confirmation(args: argparse.Namespace, audit: EpisodeAudit) -> None:
    if args.yes:
        return
    expected = f"REPLAY {audit.height}"
    print(
        "Real replay is armed. Keep the physical e-stop ready. "
        f"Type {expected!r} to stream {audit.commands} saved commands at {audit.rate_hz:.3f} Hz: ",
        end="",
        flush=True,
    )
    typed = input().strip()
    if typed != expected:
        raise ReplaySafetyError("Operator cancelled before real movement.")


def main() -> int:
    args = build_arg_parser().parse_args()
    install_stop_handlers()

    try:
        args.height = normalize_height(args.height)
        profile = select_replay_profile(args.height, dataset_family=args.dataset_family, rate_hz=args.rate_hz)
        apply_profile_defaults(args, profile)
        episode = episode_path_for_height(
            args.height,
            dataset_root=args.dataset_root,
            dataset_family=profile.name,
        )
        rows = load_action_rows(episode)
        audit = validate_action_contract(
            rows,
            episode=episode,
            height=args.height,
            profile=profile,
            side=args.side,
            rate_hz=float(args.rate_hz),
            first_zero_tolerance_deg=float(args.first_command_zero_tolerance_deg),
        )
        print_audit(audit, side=args.side, dry_run=args.dry_run)
        print_command_preview(rows)

        if args.dry_run:
            print("[dry-run] no hardware will be touched.", file=sys.stderr, flush=True)
            sink = DryRunSink()
            try:
                actual = replay_rows(
                    rows,
                    sink=sink,
                    rate_hz=args.rate_hz,
                    sleep_enabled=not args.dry_run_no_sleep,
                )
            finally:
                sink.close()
            print(
                f"[dry-run] actual_wall_clock_duration_sec={actual:.6f} "
                f"expected_duration_sec={audit.expected_duration_sec:.6f} "
                f"expected_duration_10hz_sec={audit.expected_duration_10hz_sec:.6f}",
                flush=True,
            )
            return 0

        validate_real_flags(args, profile=profile)
        port = args.port or default_port_for_side(args.side)
        # The explicit hardware confirmation guard above must run before this CAN check.
        require_can_interface(port)
        real = RealOpenArmSink(args, port=port)
        try:
            real.validate_live_limits_before_connect(rows)
            final_interactive_confirmation(args, audit)
            real.connect()
            real.assert_current_pose_matches_first(rows[0])
            print(
                "[real] starting exact saved replay: one command every "
                f"{1.0 / args.rate_hz:.3f}s, no interpolation.",
                file=sys.stderr,
                flush=True,
            )
            actual = replay_rows(rows, sink=real, rate_hz=args.rate_hz, sleep_enabled=True)
            print(
                f"[real] actual_wall_clock_duration_sec={actual:.6f} "
                f"expected_duration_sec={audit.expected_duration_sec:.6f} "
                f"expected_duration_10hz_sec={audit.expected_duration_10hz_sec:.6f}",
                flush=True,
            )
            return 0
        finally:
            real.close()
    except EmergencyStop as exc:
        print(f"[replay] emergency stop: {exc}", file=sys.stderr, flush=True)
        return 130
    except (ReplaySafetyError, KeyboardInterrupt) as exc:
        print(f"[replay] refusing/aborting: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
