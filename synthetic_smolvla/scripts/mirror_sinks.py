#!/usr/bin/env python3
"""Mirror sinks for interactive SmolVLA Isaac commands.

The default path is simulation only.  Real robot mirroring is intentionally
opt-in and goes through a guarded JSONL helper on the Jetson.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Protocol

from sim_contract import (
    JOINT_NAMES,
    REPO_ROOT,
    SAFE_GRIPPER_LIMIT_DEG,
    clamp,
    clamp_joint_targets,
    load_yaml_config,
    normalize_side,
)


REQUIRED_REAL_CONFIRMATION = "I am at the robot with e-stop ready"
DEFAULT_REAL_HOST = os.environ.get("OPENARM_REAL_HOST", "10.10.10.2")
DEFAULT_REAL_USER = os.environ.get("OPENARM_REAL_USER", "arms")
DEFAULT_REAL_REPO = os.environ.get("OPENARM_REAL_REPO", "/home/arms/hsi-pre-grasp")
DEFAULT_REAL_HELPER = os.environ.get(
    "OPENARM_REAL_HELPER",
    "/home/arms/hsi-pre-grasp/scripts/openarm_safe_real_mirror.py",
)
DEFAULT_START_POSE_TOLERANCE_DEG = 2.0
DEFAULT_FIRST_TARGET_TOLERANCE_DEG = 15.0


class MirrorSafetyError(RuntimeError):
    """Raised when a real mirror guard refuses to continue."""


@dataclass(frozen=True)
class CommandContext:
    task_index: int | None = None
    step_index: int | None = None
    typed_task: str | None = None
    policy_instruction: str | None = None
    target_object: str | None = None


@dataclass(frozen=True)
class ClampResult:
    command_deg: list[float]
    clamp_events: int


class CommandSink(Protocol):
    def emit(self, command_deg: list[float], context: CommandContext) -> None:
        """Emit one already selected policy command."""

    def close(self) -> None:
        """Release sink resources."""


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _finite_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise MirrorSafetyError(f"{label} must be finite, got {value!r}")
    return number


def clamp_command_deg(side: str, command_deg: list[float] | tuple[float, ...]) -> ClampResult:
    """Validate and clamp an 8-D arm+gripper command in degrees."""

    side = normalize_side(side)
    if len(command_deg) != 8:
        raise MirrorSafetyError(f"Expected 8 command values, got {len(command_deg)}.")
    raw = [_finite_float(value, label=f"command[{index}]") for index, value in enumerate(command_deg)]
    arm_raw = {JOINT_NAMES[index]: raw[index] for index in range(7)}
    arm_clamped = clamp_joint_targets(side, arm_raw)
    grip = clamp(raw[7], *SAFE_GRIPPER_LIMIT_DEG)
    clamped = [arm_clamped[joint] for joint in JOINT_NAMES] + [grip]
    events = sum(abs(clamped[index] - raw[index]) > 1e-5 for index in range(8))
    return ClampResult(command_deg=[float(v) for v in clamped], clamp_events=int(events))


def command_arm_dict(command_deg: list[float]) -> dict[str, float]:
    if len(command_deg) < 7:
        raise MirrorSafetyError("Expected at least 7 arm joint values.")
    return {joint: float(command_deg[index]) for index, joint in enumerate(JOINT_NAMES)}


def start_pose_from_config(config_path: str | Path, side: str) -> list[float]:
    """Return the configured symmetric start pose for a real side."""

    config = load_yaml_config(config_path)
    side = normalize_side(side)
    reset_pose = config.get("robot", {}).get("reset_pose_deg", {})
    if side not in reset_pose:
        raise MirrorSafetyError(f"Scene config has no reset pose for side {side!r}.")
    pose = reset_pose[side]
    command = [float(pose[joint]) for joint in JOINT_NAMES] + [float(pose.get("gripper", -65.0))]
    return clamp_command_deg(side, command).command_deg


def max_abs_arm_delta_deg(a: list[float], b: list[float]) -> float:
    if len(a) < 7 or len(b) < 7:
        raise MirrorSafetyError("Need at least 7 arm joints to compare deltas.")
    return max(abs(float(a[index]) - float(b[index])) for index in range(7))


def max_abs_target_delta_deg(a: list[float], b: list[float], *, include_gripper: bool) -> float:
    count = 8 if include_gripper else 7
    if len(a) < count or len(b) < count:
        raise MirrorSafetyError(f"Need {count} target values to compare deltas.")
    return max(abs(float(a[index]) - float(b[index])) for index in range(count))


def arm_error_summary(target_deg: list[float], state_deg: list[float], *, include_gripper: bool) -> dict[str, Any]:
    count = 8 if include_gripper else 7
    names = list(JOINT_NAMES) + ["gripper"]
    errors = {
        names[index]: round(abs(float(state_deg[index]) - float(target_deg[index])), 5)
        for index in range(count)
    }
    return {"errors_deg": errors, "max_error_deg": max(errors.values()) if errors else 0.0}


class SimSink:
    """Current Isaac application path wrapped as a sink."""

    def __init__(self, apply_fn: Callable[[list[float]], None]) -> None:
        self._apply_fn = apply_fn

    def emit(self, command_deg: list[float], context: CommandContext) -> None:
        del context
        self._apply_fn(command_deg)

    def close(self) -> None:
        return None


class DryRunMirrorSink:
    """Write a timestamped JSONL trace of commands that would be mirrored."""

    def __init__(self, path: str | Path, *, side: str, disable_gripper_real: bool) -> None:
        self.path = _abs(path)
        self.side = normalize_side(side)
        self.disable_gripper_real = bool(disable_gripper_real)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._previous: list[float] | None = None
        self._sequence = 0

    def emit(self, command_deg: list[float], context: CommandContext) -> None:
        clamped = clamp_command_deg(self.side, command_deg)
        delta = None
        max_delta = None
        if self._previous is not None:
            delta = [round(clamped.command_deg[index] - self._previous[index], 6) for index in range(8)]
            max_delta = max(abs(value) for value in delta[:7])
        record = {
            "type": "command",
            "sequence": self._sequence,
            "timestamp_unix": round(time.time(), 6),
            "side": self.side,
            "task_index": context.task_index,
            "step_index": context.step_index,
            "typed_task": context.typed_task,
            "policy_instruction": context.policy_instruction,
            "target_object": context.target_object,
            "command_deg": [round(float(value), 6) for value in clamped.command_deg],
            "arm_command_deg": {joint: round(value, 6) for joint, value in command_arm_dict(clamped.command_deg).items()},
            "gripper_command_deg": round(float(clamped.command_deg[7]), 6),
            "clamp_events": clamped.clamp_events,
            "delta_from_previous_deg": delta,
            "max_arm_delta_from_previous_deg": None if max_delta is None else round(float(max_delta), 6),
            "real_gripper_disabled": self.disable_gripper_real,
            "gripper_sent_to_real": not self.disable_gripper_real,
        }
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()
        self._previous = clamped.command_deg
        self._sequence += 1

    def close(self) -> None:
        self._handle.close()


class CompositeSink:
    def __init__(self, sinks: list[CommandSink]) -> None:
        self.sinks = sinks

    def emit(self, command_deg: list[float], context: CommandContext) -> None:
        for sink in self.sinks:
            sink.emit(command_deg, context)

    def close(self) -> None:
        errors = []
        for sink in reversed(self.sinks):
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - best effort cleanup
                errors.append(exc)
        if errors:
            raise errors[0]


@dataclass
class RealMirrorConfig:
    side: str
    port: str
    confirm: str
    rate_hz: float
    max_joint_delta_deg: float
    watchdog_timeout_sec: float
    disable_gripper_real: bool = True
    helper_max_relative_target_deg: float | None = None
    start_pose_max_joint_delta_deg: float | None = None
    start_pose_gripper_max_delta_deg: float | None = None
    start_pose_rate_hz: float | None = None
    host: str = DEFAULT_REAL_HOST
    user: str = DEFAULT_REAL_USER
    repo: str = DEFAULT_REAL_REPO
    helper: str = DEFAULT_REAL_HELPER
    connect_timeout_sec: float = 5.0
    request_timeout_sec: float = 8.0
    start_pose_tolerance_deg: float = DEFAULT_START_POSE_TOLERANCE_DEG
    start_pose_timeout_sec: float = 25.0
    start_pose_hold_sec: float = 0.3
    start_pose_samples: int = 1
    start_pose_duration_sec: float | None = None
    first_target_tolerance_deg: float = DEFAULT_FIRST_TARGET_TOLERANCE_DEG
    connect_retries: int = 3
    connect_retry_delay_sec: float = 1.5
    use_ssh: bool = True
    hold_interval_sec: float = 0.2

    def __post_init__(self) -> None:
        self.side = normalize_side(self.side)
        if self.confirm != REQUIRED_REAL_CONFIRMATION:
            raise MirrorSafetyError(
                "Real mirror refused: pass --real-confirm "
                f"{REQUIRED_REAL_CONFIRMATION!r} only while physically at the robot."
            )
        if self.rate_hz <= 0:
            raise MirrorSafetyError("--mirror-rate-hz must be positive.")
        if self.max_joint_delta_deg <= 0:
            raise MirrorSafetyError("--max-joint-delta-deg must be positive.")
        if self.helper_max_relative_target_deg is not None and self.helper_max_relative_target_deg <= 0:
            raise MirrorSafetyError("--real-helper-max-rel-deg must be positive.")
        if self.start_pose_max_joint_delta_deg is not None and self.start_pose_max_joint_delta_deg <= 0:
            raise MirrorSafetyError("--start-pose-max-joint-delta-deg must be positive.")
        if self.start_pose_gripper_max_delta_deg is not None and self.start_pose_gripper_max_delta_deg <= 0:
            raise MirrorSafetyError("--start-pose-gripper-max-delta-deg must be positive.")
        if self.start_pose_rate_hz is not None and self.start_pose_rate_hz <= 0:
            raise MirrorSafetyError("--start-pose-rate-hz must be positive.")
        if self.start_pose_timeout_sec <= 0:
            raise MirrorSafetyError("--real-start-pose-timeout-sec must be positive.")
        if self.start_pose_hold_sec < 0:
            raise MirrorSafetyError("--real-start-pose-hold-sec cannot be negative.")
        if self.start_pose_samples < 1:
            raise MirrorSafetyError("--real-start-pose-samples must be at least 1.")
        if self.start_pose_duration_sec is not None and self.start_pose_duration_sec <= 0:
            raise MirrorSafetyError("--real-start-pose-duration-sec must be positive.")
        if self.watchdog_timeout_sec <= 0:
            raise MirrorSafetyError("--watchdog-timeout-sec must be positive.")
        if self.hold_interval_sec <= 0:
            raise MirrorSafetyError("--hold-interval-sec must be positive.")


class _JsonlProcess:
    def __init__(self, command: list[str], *, timeout_sec: float) -> None:
        self.command = command
        self.timeout_sec = timeout_sec
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.proc.stdout is None or self.proc.stdin is None:
            raise MirrorSafetyError("Failed to open JSONL helper pipes.")
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.proc.stdout, selectors.EVENT_READ)

    def request(self, payload: dict[str, Any], *, timeout_sec: float | None = None) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise MirrorSafetyError(f"Real helper exited early with code {self.proc.returncode}.")
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        self.proc.stdin.flush()
        return self.read(timeout_sec=timeout_sec or self.timeout_sec)

    def read(self, *, timeout_sec: float | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_sec or self.timeout_sec)
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            events = self._selector.select(timeout=min(0.25, remaining))
            if events:
                line = self.proc.stdout.readline()
                if not line:
                    break
                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MirrorSafetyError(f"Real helper returned non-JSON output: {line[:160]!r}") from exc
                if not response.get("ok", False):
                    raise MirrorSafetyError(str(response.get("error", response)))
                return response
            if self.proc.poll() is not None:
                break
        tail = ""
        if self.proc.poll() is not None and self.proc.stderr is not None:
            tail = self.proc.stderr.read()[-800:]
        detail = f" stderr tail: {tail}" if tail else ""
        raise MirrorSafetyError(f"Timed out waiting for real helper response.{detail}")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self.request({"op": "stop"}, timeout_sec=2.0)
                except Exception:
                    pass
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        finally:
            try:
                self._selector.close()
            except Exception:
                pass


class RealMirrorSink:
    """Guarded mirror sink that streams arm targets to the Jetson helper."""

    def __init__(self, config: RealMirrorConfig) -> None:
        self.config = config
        self._process: _JsonlProcess | None = None
        self._last_send_monotonic = 0.0
        self._last_target_deg: list[float] | None = None
        self._latest_state_deg: list[float] | None = None
        self._first_target_checked = False
        self._prepared = False
        self._lock = threading.RLock()
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self._last_request_monotonic = 0.0

    def _build_command(self) -> list[str]:
        helper_args = [
            "python",
            self.config.helper,
            "--side",
            self.config.side,
            "--port",
            self.config.port,
            "--real-confirm",
            self.config.confirm,
            "--max-rel",
            str(self._helper_max_relative_target_deg()),
            "--max-joint-delta-deg",
            str(self.config.max_joint_delta_deg),
            "--watchdog-timeout-sec",
            str(self.config.watchdog_timeout_sec),
            "--prepare-timeout-sec",
            str(self.config.start_pose_timeout_sec),
            "--hold-sec",
            str(self.config.start_pose_hold_sec),
            "--connect-retries",
            str(self.config.connect_retries),
            "--connect-retry-delay",
            str(self.config.connect_retry_delay_sec),
        ]
        if self.config.disable_gripper_real:
            helper_args.append("--disable-gripper-real")
        else:
            helper_args.append("--enable-gripper-real")

        if not self.config.use_ssh:
            return helper_args

        remote_inner = "cd {} && source .venv/bin/activate && {}".format(
            shlex.quote(self.config.repo),
            " ".join(shlex.quote(part) for part in helper_args),
        )
        remote_cmd = "bash -lc {}".format(shlex.quote(remote_inner))
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(self.config.connect_timeout_sec)}",
            f"{self.config.user}@{self.config.host}",
            remote_cmd,
        ]

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._process is None:
                self._process = _JsonlProcess(self._build_command(), timeout_sec=self.config.request_timeout_sec)
                ready = self._process.read(timeout_sec=max(self.config.request_timeout_sec, 15.0))
                self._latest_state_deg = ready.get("state_deg")
                self._last_request_monotonic = time.monotonic()
                return ready
            return {"ok": True, "event": "already_started"}

    def _request(self, payload: dict[str, Any], *, timeout_sec: float | None = None) -> dict[str, Any]:
        self.start()
        assert self._process is not None
        with self._lock:
            response = self._process.request(payload, timeout_sec=timeout_sec or self.config.request_timeout_sec)
            self._last_request_monotonic = time.monotonic()
            return response

    def read_state(self) -> list[float]:
        response = self._request({"op": "read_state"}, timeout_sec=self.config.request_timeout_sec)
        state = response.get("state_deg")
        if not isinstance(state, list) or len(state) != 8:
            raise MirrorSafetyError(f"Real helper returned invalid state: {state!r}")
        self._latest_state_deg = [float(value) for value in state]
        return self._latest_state_deg

    def latest_state_deg(self) -> list[float] | None:
        if self._latest_state_deg is None:
            return None
        return [float(value) for value in self._latest_state_deg]

    def prepare_start_pose(self, start_pose_deg: list[float]) -> dict[str, Any]:
        if self.config.start_pose_duration_sec is not None:
            return self.prepare_start_pose_over_duration(start_pose_deg, duration_sec=self.config.start_pose_duration_sec)

        target = clamp_command_deg(self.config.side, start_pose_deg).command_deg
        self.stop_keepalive()
        include_gripper = not self.config.disable_gripper_real
        sampled_results: list[dict[str, Any]] = []

        targets = [target]
        if self.config.start_pose_samples > 1:
            current_state = self.read_state()
            targets = self._fixed_start_pose_samples(current_state, target, samples=self.config.start_pose_samples)

        for sample_index, sample_target in enumerate(targets, start=1):
            response, sample_summary = self._prepare_start_target(
                sample_target,
                include_gripper=include_gripper,
                sample_index=sample_index,
                sample_count=len(targets),
            )
            sampled_results.append(
                {
                    "sample_index": sample_index,
                    "sample_count": len(targets),
                    "event": response.get("event", "prepared_start"),
                    "max_error_deg": sample_summary["max_error_deg"],
                    "target_deg": [round(float(value), 6) for value in sample_target],
                }
            )

        assert self._latest_state_deg is not None
        summary = arm_error_summary(target, self._latest_state_deg, include_gripper=include_gripper)
        if float(summary["max_error_deg"]) > self.config.start_pose_tolerance_deg:
            raise MirrorSafetyError(
                "Real start pose not reached after sampled preparation: "
                f"max_error={summary['max_error_deg']:.3f} deg, "
                f"tolerance={self.config.start_pose_tolerance_deg:.3f} deg."
            )
        self._last_target_deg = target
        self._first_target_checked = False
        self._prepared = True
        self.start_keepalive()
        return {
            "ok": True,
            "event": "prepared_start",
            "state_deg": self._latest_state_deg,
            "max_error_deg": summary["max_error_deg"],
            "gripper_sent_to_real": include_gripper,
            "start_pose_method": "helper_prepare_start_sampled"
            if self.config.start_pose_samples > 1
            else "helper_prepare_start",
            "start_pose_samples": self.config.start_pose_samples,
            "sampled_results": sampled_results,
        }

    def prepare_start_pose_over_duration(self, start_pose_deg: list[float], *, duration_sec: float) -> dict[str, Any]:
        target = clamp_command_deg(self.config.side, start_pose_deg).command_deg
        self.stop_keepalive()
        include_gripper = not self.config.disable_gripper_real
        current_state = self.read_state()
        start_for_commands = self._last_target_deg if self._last_target_deg is not None else current_state
        rate_hz = self._start_pose_rate_hz()
        samples = max(1, int(math.ceil(duration_sec * rate_hz)))
        max_delta = max_abs_target_delta_deg(
            target,
            start_for_commands,
            include_gripper=include_gripper,
        )
        max_sample_delta = max_delta / samples
        if max_sample_delta > self.config.max_joint_delta_deg:
            raise MirrorSafetyError(
                "Init duration is too short for the configured delta guard: "
                f"{max_sample_delta:.3f} deg/sample > {self.config.max_joint_delta_deg:.3f} deg. "
                "Increase --real-start-pose-duration-sec, increase --replay-rate-hz, "
                "or use a larger --max-joint-delta-deg."
            )

        for step_index, intermediate in enumerate(
            self._fixed_start_pose_samples(start_for_commands, target, samples=samples),
            start=1,
        ):
            response = self._send_target(
                intermediate,
                CommandContext(task_index=None, step_index=step_index, typed_task="prepare_start_pose_duration"),
                rate_hz=rate_hz,
            )
            if isinstance(response.get("state_deg"), list) and len(response["state_deg"]) == 8:
                self._latest_state_deg = [float(value) for value in response["state_deg"]]
            self._last_target_deg = intermediate

        final_state = self.read_state()
        self._latest_state_deg = [float(value) for value in final_state]
        summary = arm_error_summary(target, self._latest_state_deg, include_gripper=include_gripper)
        if float(summary["max_error_deg"]) > self.config.start_pose_tolerance_deg:
            raise MirrorSafetyError(
                "Real start pose not reached after timed preparation: "
                f"max_error={summary['max_error_deg']:.3f} deg, "
                f"tolerance={self.config.start_pose_tolerance_deg:.3f} deg."
            )
        self._last_target_deg = target
        self._first_target_checked = False
        self._prepared = True
        self.start_keepalive()
        return {
            "ok": True,
            "event": "prepared_start_timed",
            "state_deg": self._latest_state_deg,
            "max_error_deg": summary["max_error_deg"],
            "gripper_sent_to_real": include_gripper,
            "start_pose_method": "timed_target_stream",
            "start_pose_duration_sec": float(duration_sec),
            "start_pose_samples": samples,
            "rate_hz": rate_hz,
            "max_sample_delta_deg": round(float(max_sample_delta), 6),
        }

    def _prepare_start_target(
        self,
        target: list[float],
        *,
        include_gripper: bool,
        sample_index: int,
        sample_count: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self._request(
            {
                "op": "prepare_start",
                "target_deg": target,
                "tolerance_deg": self.config.start_pose_tolerance_deg,
            },
            timeout_sec=self._start_pose_request_timeout_sec(),
        )
        state = response.get("state_deg")
        if not isinstance(state, list) or len(state) != 8:
            state = self.read_state()
        self._latest_state_deg = [float(value) for value in state]
        summary = arm_error_summary(target, self._latest_state_deg, include_gripper=include_gripper)
        if float(summary["max_error_deg"]) > self.config.start_pose_tolerance_deg:
            raise MirrorSafetyError(
                f"Real start pose sample {sample_index}/{sample_count} not reached: "
                f"max_error={summary['max_error_deg']:.3f} deg, "
                f"tolerance={self.config.start_pose_tolerance_deg:.3f} deg."
            )
        return response, summary

    def stage_pose_without_audit(
        self,
        command_deg: list[float],
        *,
        duration_sec: float,
        label: str,
    ) -> dict[str, Any]:
        """Send a safe staging target for a fixed time without requiring stability.

        This is only for pass-through staging poses. The real start pose still
        must use prepare_start_pose() and pass readback audit before mirroring.
        """

        if duration_sec <= 0:
            raise MirrorSafetyError("--zero-stage-sec must be positive.")
        target = clamp_command_deg(self.config.side, command_deg).command_deg
        self.stop_keepalive()
        if self._latest_state_deg is None:
            self.read_state()
        context = CommandContext(task_index=None, step_index=None, typed_task=label)
        deadline = time.monotonic() + duration_sec
        sends = 0
        response: dict[str, Any] = {}
        while time.monotonic() < deadline:
            response = self._send_target(target, context)
            sends += 1
            if isinstance(response.get("state_deg"), list) and len(response["state_deg"]) == 8:
                self._latest_state_deg = [float(value) for value in response["state_deg"]]
            self._last_target_deg = target
        if self._latest_state_deg is None:
            self.read_state()
        assert self._latest_state_deg is not None
        include_gripper = not self.config.disable_gripper_real
        summary = arm_error_summary(target, self._latest_state_deg, include_gripper=include_gripper)
        return {
            "ok": True,
            "event": "staging_pose_sent",
            "label": label,
            "duration_sec": float(duration_sec),
            "sends": sends,
            "state_deg": self._latest_state_deg,
            "target_deg": target,
            "max_error_deg": summary["max_error_deg"],
            "errors_deg": summary["errors_deg"],
            "gripper_sent_to_real": include_gripper,
            "last_response_event": response.get("event"),
        }

    def audit_prepared_start_pose(self, target_deg: list[float]) -> dict[str, Any]:
        if self._latest_state_deg is None:
            self.read_state()
        assert self._latest_state_deg is not None
        include_gripper = not self.config.disable_gripper_real
        summary = arm_error_summary(target_deg, self._latest_state_deg, include_gripper=include_gripper)
        if float(summary["max_error_deg"]) > self.config.start_pose_tolerance_deg:
            raise MirrorSafetyError(
                "Real start-pose readback is outside tolerance: "
                f"max_error={summary['max_error_deg']:.3f} deg, "
                f"tolerance={self.config.start_pose_tolerance_deg:.3f} deg."
            )
        return summary

    def _rate_limit(self, *, rate_hz: float | None = None) -> None:
        min_interval = 1.0 / float(rate_hz or self.config.rate_hz)
        now = time.monotonic()
        elapsed = now - self._last_send_monotonic
        if self._last_send_monotonic > 0.0 and elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_send_monotonic = time.monotonic()

    def start_keepalive(self) -> None:
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return
        # The real OpenArm follower behaves best when the held target is
        # refreshed like the older real scripts' hold loops. A slower heartbeat
        # can let the arm relax while the typed-task prompt is waiting.
        interval = max(0.05, min(self.config.hold_interval_sec, self.config.watchdog_timeout_sec * 0.4))
        self._keepalive_stop.clear()

        def loop() -> None:
            while not self._keepalive_stop.wait(interval):
                if not self._prepared:
                    continue
                # Only send an explicit hold when no normal target/read request
                # has refreshed the watchdog recently.
                if time.monotonic() - self._last_request_monotonic < interval:
                    continue
                try:
                    response = self._request({"op": "hold"}, timeout_sec=self.config.request_timeout_sec)
                    if isinstance(response.get("state_deg"), list) and len(response["state_deg"]) == 8:
                        self._latest_state_deg = [float(value) for value in response["state_deg"]]
                except Exception as exc:  # noqa: BLE001 - surfaced by next foreground request/close
                    print(f"[real-mirror] hold keepalive stopped: {exc}", file=sys.stderr, flush=True)
                    self._keepalive_stop.set()

        self._keepalive_thread = threading.Thread(target=loop, name="openarm-real-hold-keepalive", daemon=True)
        self._keepalive_thread.start()

    def stop_keepalive(self) -> None:
        self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=2.0)
            self._keepalive_thread = None
        self._keepalive_stop.clear()

    def _interpolated_targets(self, start: list[float] | None, target: list[float]) -> list[list[float]]:
        if start is None:
            return [target]
        max_delta = max_abs_target_delta_deg(
            target,
            start,
            include_gripper=not self.config.disable_gripper_real,
        )
        steps = max(1, int(math.ceil(max_delta / self.config.max_joint_delta_deg)))
        out = []
        for step in range(1, steps + 1):
            alpha = step / steps
            out.append([float(start[index] + (target[index] - start[index]) * alpha) for index in range(8)])
        return out

    def _start_pose_arm_delta_deg(self) -> float:
        return float(self.config.start_pose_max_joint_delta_deg or self.config.max_joint_delta_deg)

    def _start_pose_gripper_delta_deg(self) -> float:
        return float(
            self.config.start_pose_gripper_max_delta_deg
            or self.config.start_pose_max_joint_delta_deg
            or self.config.max_joint_delta_deg
        )

    def _start_pose_rate_hz(self) -> float:
        return float(self.config.start_pose_rate_hz or self.config.rate_hz)

    def _helper_max_relative_target_deg(self) -> float:
        return float(self.config.helper_max_relative_target_deg or self.config.max_joint_delta_deg)

    def _start_pose_request_timeout_sec(self) -> float:
        return max(
            float(self.config.request_timeout_sec),
            float(self.config.start_pose_timeout_sec) + 10.0,
        )

    def _fixed_start_pose_samples(
        self,
        start: list[float],
        target: list[float],
        *,
        samples: int,
    ) -> list[list[float]]:
        if samples < 1:
            raise MirrorSafetyError("--real-start-pose-samples must be at least 1.")
        return [
            [float(start[index] + (target[index] - start[index]) * (step / samples)) for index in range(8)]
            for step in range(1, samples + 1)
        ]

    def _interpolated_start_pose_targets(self, start: list[float], target: list[float]) -> list[list[float]]:
        include_gripper = not self.config.disable_gripper_real
        limits = [self._start_pose_arm_delta_deg()] * 7 + [self._start_pose_gripper_delta_deg()]
        count = 8 if include_gripper else 7
        steps = 1
        for index in range(count):
            steps = max(steps, int(math.ceil(abs(float(target[index]) - float(start[index])) / limits[index])))
        return [
            [float(start[index] + (target[index] - start[index]) * (step / steps)) for index in range(8)]
            for step in range(1, steps + 1)
        ]

    def _send_target(
        self,
        target: list[float],
        context: CommandContext,
        *,
        rate_hz: float | None = None,
    ) -> dict[str, Any]:
        self._rate_limit(rate_hz=rate_hz)
        return self._request(
            {
                "op": "target",
                "target_deg": target,
                "task_index": context.task_index,
                "step_index": context.step_index,
                "disable_gripper_real": self.config.disable_gripper_real,
            },
            timeout_sec=self.config.request_timeout_sec,
        )

    def emit(self, command_deg: list[float], context: CommandContext) -> None:
        if not self._prepared:
            raise MirrorSafetyError("Real mirror refused: start-pose preflight has not completed.")
        target = clamp_command_deg(self.config.side, command_deg).command_deg
        if not self._first_target_checked:
            if self._latest_state_deg is None:
                self.read_state()
            assert self._latest_state_deg is not None
            first_delta = max_abs_arm_delta_deg(target, self._latest_state_deg)
            if first_delta > self.config.first_target_tolerance_deg:
                raise MirrorSafetyError(
                    "First real mirror target is too far from current robot state: "
                    f"{first_delta:.3f} deg > {self.config.first_target_tolerance_deg:.3f} deg."
                )
            self._first_target_checked = True
        for intermediate in self._interpolated_targets(self._last_target_deg, target):
            response = self._send_target(intermediate, context)
            if isinstance(response.get("state_deg"), list) and len(response["state_deg"]) == 8:
                self._latest_state_deg = [float(value) for value in response["state_deg"]]
        self._last_target_deg = target

    def close(self) -> None:
        self.stop_keepalive()
        if self._process is not None:
            self._process.close()
            self._process = None
