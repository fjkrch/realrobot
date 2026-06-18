#!/usr/bin/env python3
"""Guarded JSONL helper for SmolVLA sim-to-real OpenArm mirroring.

This script is intended to run on the Jetson.  It does not accept free-form
motion commands: the laptop sends structured JSON lines after the operator has
explicitly confirmed they are at the robot with the e-stop ready.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import subprocess
import sys
import time
from typing import Any


REQUIRED_REAL_CONFIRMATION = "I am at the robot with e-stop ready"
JOINT_NAMES = [f"joint_{index}" for index in range(1, 8)]
ALL_MOTORS = JOINT_NAMES + ["gripper"]


class HelperError(RuntimeError):
    """Raised when the helper must refuse or abort."""


class Watchdog:
    def __init__(self, timeout_sec: float) -> None:
        if timeout_sec <= 0:
            raise HelperError("--watchdog-timeout-sec must be positive.")
        self.timeout_sec = float(timeout_sec)
        self.deadline = time.monotonic() + self.timeout_sec

    def kick(self) -> None:
        self.deadline = time.monotonic() + self.timeout_sec

    def expired(self) -> bool:
        return time.monotonic() > self.deadline


def json_response(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def require_can_interface(port: str) -> None:
    result = subprocess.run(
        ["ip", "link", "show", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise HelperError(f"{port} does not exist. Start CAN first.")


def finite_target(values: Any) -> list[float]:
    if not isinstance(values, list) or len(values) != 8:
        raise HelperError("target_deg must be an 8-value list.")
    out = []
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise HelperError(f"target_deg[{index}] must be finite.")
        out.append(number)
    return out


def max_abs_target_delta_deg(a: list[float], b: list[float], *, include_gripper: bool) -> float:
    count = 8 if include_gripper else 7
    return max(abs(float(a[index]) - float(b[index])) for index in range(count))


class OpenArmMirrorSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.robot = None
        self.last_target_deg: list[float] | None = None

    def _build_robot(self):
        from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

        max_relative_target = {name: self.args.max_rel for name in JOINT_NAMES}
        if not self.args.disable_gripper_real:
            max_relative_target["gripper"] = self.args.max_rel
        return OpenArmFollower(
            OpenArmFollowerConfig(
                port=self.args.port,
                side=self.args.side,
                id=self.args.id,
                max_relative_target=max_relative_target,
            )
        )

    def connect(self) -> None:
        require_can_interface(self.args.port)
        probe = self._build_robot()
        joint_limits = getattr(probe.config, "joint_limits", {}) or {}
        missing = [name for name in JOINT_NAMES if name not in joint_limits]
        if missing:
            raise HelperError(
                f"Joints missing from live joint_limits: {missing}. "
                f"Known joints: {sorted(joint_limits.keys())}"
            )
        if not self.args.disable_gripper_real and "gripper" not in joint_limits:
            raise HelperError("Gripper enabled but live joint_limits has no gripper entry.")
        if probe.is_calibrated:
            raise HelperError(f"Calibration exists for id={self.args.id!r}; use a fresh id.")

        last_exc = None
        for attempt in range(1, self.args.connect_retries + 1):
            robot = self._build_robot()
            try:
                robot.connect(calibrate=False)
                self.robot = robot
                return
            except Exception as exc:  # noqa: BLE001 - hardware dependency exceptions vary
                last_exc = exc
                print(f"[safe-real-mirror] connect attempt {attempt} failed: {exc}", file=sys.stderr, flush=True)
                try:
                    robot.disconnect()
                except Exception:
                    pass
                if attempt < self.args.connect_retries:
                    time.sleep(self.args.connect_retry_delay)
        raise HelperError(
            f"Could not connect after {self.args.connect_retries} attempts. Last error: {last_exc}"
        )

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None

    def _observation(self) -> dict[str, Any]:
        if self.robot is None:
            raise HelperError("Robot is not connected.")
        obs = self.robot.get_observation()
        missing = [f"{name}.pos" for name in ALL_MOTORS if f"{name}.pos" not in obs]
        if missing:
            available = sorted(key for key in obs if key.endswith(".pos"))
            raise HelperError(f"Observation missing {missing}. Available position keys: {available}")
        return obs

    def read_state_deg(self) -> list[float]:
        obs = self._observation()
        return [round(float(obs[f"{name}.pos"]), 6) for name in ALL_MOTORS]

    def _validate_target(self, target_deg: list[float], *, include_gripper: bool) -> None:
        if self.robot is None:
            raise HelperError("Robot is not connected.")
        joint_limits = getattr(self.robot.config, "joint_limits", {}) or {}
        names = ALL_MOTORS if include_gripper else JOINT_NAMES
        for index, name in enumerate(names):
            low, high = joint_limits[name]
            value = target_deg[index]
            if value < low or value > high:
                raise HelperError(f"Refusing {name}={value:.3f} deg outside live limit {low:.3f}..{high:.3f}.")

    def _action_from_target(self, target_deg: list[float], *, include_gripper: bool) -> dict[str, float]:
        names = ALL_MOTORS if include_gripper else JOINT_NAMES
        return {f"{name}.pos": float(target_deg[index]) for index, name in enumerate(names)}

    def prepare_start(self, target_deg: list[float], *, tolerance_deg: float) -> dict[str, Any]:
        include_gripper = not self.args.disable_gripper_real
        self._validate_target(target_deg, include_gripper=include_gripper)
        action = self._action_from_target(target_deg, include_gripper=include_gripper)
        if self.robot is None:
            raise HelperError("Robot is not connected.")
        last_sent = self.robot.send_action(action)
        deadline = time.monotonic() + self.args.prepare_timeout_sec
        stable_since = None
        state = self.read_state_deg()
        max_err = float("inf")
        worst = "unknown"
        errors: dict[str, float] = {}
        while time.monotonic() < deadline:
            state = self.read_state_deg()
            count = 8 if include_gripper else 7
            names = ALL_MOTORS if include_gripper else JOINT_NAMES
            errors = {
                names[index]: round(abs(state[index] - target_deg[index]), 6)
                for index in range(count)
            }
            worst = max(errors, key=errors.get)
            max_err = errors[worst]
            if max_err <= tolerance_deg:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= self.args.hold_sec:
                    self.last_target_deg = target_deg
                    return {
                        "ok": True,
                        "event": "prepared_start",
                        "state_deg": state,
                        "max_error_deg": round(float(max_err), 6),
                        "gripper_sent_to_real": include_gripper,
                    }
            else:
                stable_since = None
                last_sent = self.robot.send_action(action)
            time.sleep(0.05)
        state = self.read_state_deg()
        count = 8 if include_gripper else 7
        names = ALL_MOTORS if include_gripper else JOINT_NAMES
        errors = {
            names[index]: round(abs(state[index] - target_deg[index]), 6)
            for index in range(count)
        }
        worst = max(errors, key=errors.get)
        max_err = errors[worst]
        self.disconnect()
        raise HelperError(
            f"Start pose not reached within {self.args.prepare_timeout_sec:.1f}s; "
            f"max_error={max_err:.3f} deg at {worst}; "
            f"state_deg={[round(float(value), 6) for value in state]}; "
            f"target_deg={[round(float(value), 6) for value in target_deg]}; "
            f"errors_deg={errors}; last_sent={last_sent}; "
            "torque was released by disconnect."
        )

    def send_target(self, target_deg: list[float]) -> dict[str, Any]:
        include_gripper = not self.args.disable_gripper_real
        self._validate_target(target_deg, include_gripper=include_gripper)
        if self.last_target_deg is not None:
            delta = max_abs_target_delta_deg(
                target_deg,
                self.last_target_deg,
                include_gripper=include_gripper,
            )
            if delta > self.args.max_joint_delta_deg:
                self.disconnect()
                raise HelperError(
                    f"Target delta {delta:.3f} deg exceeds --max-joint-delta-deg "
                    f"{self.args.max_joint_delta_deg:.3f}; torque was released by disconnect."
                )
        if self.robot is None:
            raise HelperError("Robot is not connected.")
        sent = self.robot.send_action(self._action_from_target(target_deg, include_gripper=include_gripper))
        self.last_target_deg = target_deg
        return {
            "ok": True,
            "event": "target_sent",
            "sent": sent,
            "state_deg": self.read_state_deg(),
            "gripper_sent_to_real": include_gripper,
        }

    def hold_current_target(self) -> dict[str, Any]:
        if self.last_target_deg is None:
            raise HelperError("No target has been prepared yet, so there is nothing to hold.")
        include_gripper = not self.args.disable_gripper_real
        if self.robot is None:
            raise HelperError("Robot is not connected.")
        self.robot.send_action(self._action_from_target(self.last_target_deg, include_gripper=include_gripper))
        return {
            "ok": True,
            "event": "hold_sent",
            "state_deg": self.read_state_deg(),
            "gripper_sent_to_real": include_gripper,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--port", default="can0")
    parser.add_argument("--id", default="vla_mirror_noncal")
    parser.add_argument("--real-confirm", default="")
    parser.add_argument("--max-rel", type=float, default=3.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=3.0)
    parser.add_argument("--watchdog-timeout-sec", type=float, default=2.0)
    parser.add_argument("--prepare-timeout-sec", type=float, default=25.0)
    parser.add_argument("--hold-sec", type=float, default=0.3)
    parser.add_argument("--connect-retries", type=int, default=3)
    parser.add_argument("--connect-retry-delay", type=float, default=1.5)
    parser.add_argument("--disable-gripper-real", dest="disable_gripper_real", action="store_true", default=True)
    parser.add_argument("--enable-gripper-real", dest="disable_gripper_real", action="store_false")
    parser.add_argument("--self-test-watchdog", action="store_true")
    return parser


def run_self_test_watchdog() -> int:
    watchdog = Watchdog(0.05)
    if watchdog.expired():
        raise SystemExit("watchdog expired too early")
    time.sleep(0.08)
    if not watchdog.expired():
        raise SystemExit("watchdog did not expire")
    watchdog.kick()
    if watchdog.expired():
        raise SystemExit("watchdog did not reset")
    json_response({"ok": True, "event": "watchdog_self_test_passed"})
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.self_test_watchdog:
        return run_self_test_watchdog()
    if args.real_confirm != REQUIRED_REAL_CONFIRMATION:
        raise SystemExit(
            "Refusing to touch CAN. Pass --real-confirm "
            f"{REQUIRED_REAL_CONFIRMATION!r} only while physically at the robot."
        )
    if args.max_rel <= 0 or args.max_joint_delta_deg <= 0:
        raise SystemExit("--max-rel and --max-joint-delta-deg must be positive.")

    watchdog = Watchdog(args.watchdog_timeout_sec)
    session = OpenArmMirrorSession(args)
    try:
        try:
            session.connect()
        except Exception as exc:  # noqa: BLE001 - report startup failures as JSONL
            json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return 2
        watchdog.kick()
        json_response(
            {
                "ok": True,
                "event": "ready",
                "side": args.side,
                "port": args.port,
                "state_deg": session.read_state_deg(),
                "gripper_sent_to_real": not args.disable_gripper_real,
            }
        )
        while True:
            if watchdog.expired():
                session.disconnect()
                json_response(
                    {
                        "ok": False,
                        "error": "watchdog timeout; torque was released by disconnect",
                    }
                )
                return 2
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            line = sys.stdin.readline()
            if not line:
                session.disconnect()
                return 0
            watchdog.kick()
            try:
                payload = json.loads(line)
                op = payload.get("op")
                if op == "read_state":
                    json_response({"ok": True, "event": "state", "state_deg": session.read_state_deg()})
                elif op == "prepare_start":
                    target = finite_target(payload.get("target_deg"))
                    tolerance = float(payload.get("tolerance_deg", 2.0))
                    json_response(session.prepare_start(target, tolerance_deg=tolerance))
                elif op == "target":
                    target = finite_target(payload.get("target_deg"))
                    json_response(session.send_target(target))
                elif op == "hold":
                    json_response(session.hold_current_target())
                elif op == "stop":
                    session.disconnect()
                    json_response({"ok": True, "event": "stopped"})
                    return 0
                else:
                    raise HelperError(f"Unknown op {op!r}.")
            except Exception as exc:  # noqa: BLE001 - return errors as JSONL
                session.disconnect()
                json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                return 2
    finally:
        session.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
