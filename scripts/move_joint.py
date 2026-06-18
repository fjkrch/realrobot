#!/usr/bin/env python3
"""Move ONE OpenArm arm joint to a target position.

This is a sibling of scripts/open_gripper_small.py. It uses the exact same
OpenArmFollower API and the same safety pattern, but commands a single arm
joint (joint_1 .. joint_7) instead of the gripper.

Safety:
- Run only while physically at the robot with e-stop ready.
- Uses connect(calibrate=False), so it does not calibrate or re-mark motor zero.
- Sends only ONE joint position target per run; all other joints are held by the
  follower exactly like the gripper script holds the arm while moving the gripper.
- The target is clamped to the joint's real joint_limits from the live config,
  and each step is capped by max_relative_target so the joint creeps to target
  instead of snapping.
- If the requested joint name or its current position cannot be read from the
  live robot, the script REFUSES and prints what it actually found, rather than
  guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig


DEFAULT_ISAACLAB_MIRROR_SERVER = "http://10.10.10.1:8765"


def require_can_interface(port: str) -> None:
    result = subprocess.run(
        ["ip", "link", "show", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{port} does not exist. Start CAN first.")


def normalize_joint_name(raw: str) -> str:
    """Accept '1'..'7' or a full name like 'joint_3' and return the motor name."""
    raw = raw.strip()
    if raw.isdigit():
        return f"joint_{int(raw)}"
    return raw


def joint_to_isaac_command(side: str, joint_name: str, target_deg: float) -> str | None:
    """Build the IsaacLab mirror command, e.g. 'joint right 1 10.000 deg'."""
    m = re.fullmatch(r"joint_(\d+)", joint_name)
    if not m:
        return None
    return f"joint {side} {int(m.group(1))} {target_deg:.3f} deg"


def connect_with_retries(build_robot, *, retries: int, delay: float):
    """Connect, retrying the CAN handshake a few times.

    The first handshake right after scripts/can_up.sh often fails because the
    ACK test leaves the bus in ERROR-PASSIVE and resets the interface; the motor
    needs a moment before it ACKs. A short retry reliably recovers it.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        robot = build_robot()
        try:
            robot.connect(calibrate=False)
            if attempt > 1:
                print(f"Connected on attempt {attempt}.")
            return robot
        except Exception as exc:  # ConnectionError and friends
            last_exc = exc
            print(f"[connect] attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            try:
                robot.disconnect()
            except Exception:
                pass
            if attempt < retries:
                time.sleep(delay)
    raise SystemExit(
        f"Could not connect after {retries} attempts. Last error: {last_exc}\n"
        "Check 24 V supply ON, e-stop released, CAN wiring, and that\n"
        "  lerobot-setup-can --mode=test --interfaces=<port>\n"
        "shows 8/8 motors on that bus."
    )


def mirror_to_isaac(command: str, *, server: str, timeout_sec: float) -> None:
    """Best-effort mirror command to the laptop Isaac Lab bridge."""
    if not server:
        print("[Isaac mirror] disabled: empty server URL")
        return
    payload = json.dumps({"command": command}).encode("utf-8")
    request = Request(server.rstrip("/") + "/command", data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    token = os.environ.get("BRIDGE_TOKEN", "")
    if token:
        request.add_header("X-Bridge-Token", token)
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"[Isaac mirror] sent {command!r}: HTTP {response.status} {body}")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[Isaac mirror] warning: HTTP {exc.code}: {body}", file=sys.stderr)
    except URLError as exc:
        print(f"[Isaac mirror] warning: could not reach {server}: {exc}", file=sys.stderr)
    except TimeoutError:
        print(f"[Isaac mirror] warning: timeout contacting {server}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="can0", help="CAN interface, e.g. can0")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument(
        "--joint",
        required=True,
        help="joint to move: 1..7 or a full name like joint_3 (gripper not allowed here)",
    )
    parser.add_argument(
        "--target-deg",
        type=float,
        default=None,
        help="absolute joint position target in degrees",
    )
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=None,
        help="relative joint move in degrees; ignored if --target-deg is provided",
    )
    parser.add_argument(
        "--max-rel",
        type=float,
        default=5.0,
        help="max degrees the joint may move per control step (per-step safety clamp)",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=15.0,
        help="refuse a --delta-deg larger than this many degrees",
    )
    parser.add_argument("--hold-sec", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=float, default=15.0, help="max time to wait for target")
    parser.add_argument("--tolerance-deg", type=float, default=0.5, help="target reached tolerance")
    parser.add_argument(
        "--disable-on-fail",
        action="store_true",
        help="disable torque even if the joint does not reach target",
    )
    parser.add_argument("--connect-retries", type=int, default=3,
                        help="retry the CAN handshake this many times (transient miss after can_up)")
    parser.add_argument("--connect-retry-delay", type=float, default=1.5,
                        help="seconds to wait between connect attempts")
    parser.add_argument("--id", default="joint_noncal", help="fresh id with no calibration file")
    parser.add_argument("--i-am-at-robot", action="store_true", help="required safety acknowledgement")
    parser.add_argument("--yes", action="store_true", help="skip confirmation prompt")
    parser.add_argument(
        "--isaac-mirror-server",
        default=os.environ.get("ISAACLAB_MIRROR_SERVER", DEFAULT_ISAACLAB_MIRROR_SERVER),
        help="laptop Isaac Lab mirror server URL",
    )
    parser.add_argument(
        "--no-isaac-mirror",
        action="store_true",
        help="do not mirror this joint command into Isaac Lab",
    )
    parser.add_argument(
        "--isaac-mirror-timeout-sec",
        type=float,
        default=2.0,
        help="max time to wait for the Isaac Lab mirror request",
    )
    args = parser.parse_args()

    if not args.i_am_at_robot:
        raise SystemExit("Refusing to run. Add --i-am-at-robot when you are physically at the robot.")
    if args.target_deg is None and args.delta_deg is None:
        raise SystemExit("Pass either --target-deg for absolute position or --delta-deg for relative movement.")
    if args.delta_deg is not None and abs(args.delta_deg) > args.max_delta:
        raise SystemExit(f"Refusing relative joint move larger than {args.max_delta:.1f} deg.")
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be positive.")
    if args.tolerance_deg <= 0:
        raise SystemExit("--tolerance-deg must be positive.")
    if args.max_rel <= 0:
        raise SystemExit("--max-rel must be positive.")

    joint_name = normalize_joint_name(args.joint)
    if joint_name == "gripper":
        raise SystemExit("Use scripts/open_gripper_small.py for the gripper.")
    pos_key = f"{joint_name}.pos"

    require_can_interface(args.port)

    def build_robot() -> OpenArmFollower:
        return OpenArmFollower(
            OpenArmFollowerConfig(
                port=args.port,
                side=args.side,
                id=args.id,
                max_relative_target={joint_name: args.max_rel},
            )
        )

    robot = build_robot()

    # Validate the joint name against the live config BEFORE connecting/moving.
    joint_limits = getattr(robot.config, "joint_limits", {}) or {}
    if joint_name not in joint_limits:
        raise SystemExit(
            f"Unknown joint {joint_name!r}. Joints with known limits are: "
            f"{sorted(joint_limits.keys())}"
        )
    joint_min, joint_max = joint_limits[joint_name]

    if robot.is_calibrated:
        raise SystemExit(
            f"Calibration exists for id={args.id!r}. Use a fresh --id to avoid any re-zero path."
        )

    if args.target_deg is not None:
        print(f"About to move ONLY {joint_name} on {args.port} ({args.side}) to absolute {args.target_deg:.2f} deg.")
    else:
        print(f"About to move ONLY {joint_name} on {args.port} ({args.side}) by {args.delta_deg:+.2f} deg.")
    print(f"Allowed range for {joint_name} side={args.side}: {joint_min:.1f} to {joint_max:.1f} deg.")
    print(f"Per-step clamp (max_relative_target): {args.max_rel:.1f} deg, so the joint creeps to target.")
    print("Keep e-stop ready. If it moves the wrong way, stop and rerun with opposite sign.")
    if not args.yes:
        if input("Type MOVE to continue: ").strip() != "MOVE":
            print("Cancelled.")
            return 1

    reached = False
    should_disconnect = False
    robot = None
    try:
        robot = connect_with_retries(
            build_robot,
            retries=args.connect_retries,
            delay=args.connect_retry_delay,
        )
        obs = robot.get_observation()
        if pos_key not in obs:
            available = sorted(k for k in obs.keys() if k.endswith(".pos"))
            should_disconnect = True
            raise SystemExit(
                f"Observation has no {pos_key!r}. Available position keys: {available}"
            )
        current = float(obs[pos_key])
        if args.target_deg is not None:
            target = args.target_deg
        else:
            target = current + args.delta_deg
        print(f"Current {pos_key}: {current:.3f} deg")
        print(f"Target  {pos_key}: {target:.3f} deg")
        if target < joint_min or target > joint_max:
            print(
                f"Refusing out-of-range target {target:.3f} deg. "
                f"Allowed range is {joint_min:.3f} to {joint_max:.3f} deg."
            )
            should_disconnect = True
            return 2
        if not args.no_isaac_mirror:
            isaac_command = joint_to_isaac_command(args.side, joint_name, target)
            if isaac_command is not None:
                mirror_to_isaac(
                    isaac_command,
                    server=args.isaac_mirror_server,
                    timeout_sec=args.isaac_mirror_timeout_sec,
                )
        sent = robot.send_action({pos_key: target})
        print(f"Sent: {sent}")

        deadline = time.monotonic() + args.timeout_sec
        stable_since = None
        last = current
        while time.monotonic() < deadline:
            obs2 = robot.get_observation()
            last = float(obs2.get(pos_key, last))
            err = abs(last - target)
            print(f"\r{pos_key}={last:.3f} deg  target={target:.3f} deg  error={err:.3f} deg", end="", flush=True)

            if err <= args.tolerance_deg:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= args.hold_sec:
                    reached = True
                    break
            else:
                stable_since = None
                robot.send_action({pos_key: target})
            time.sleep(0.05)

        print()
        if reached:
            print(
                f"Reached target within ±{args.tolerance_deg:.3f} deg "
                f"and held for {args.hold_sec:.2f}s. Disabling torque now."
            )
            should_disconnect = True
        else:
            print(
                f"NOT at target after {args.timeout_sec:.1f}s "
                f"(last={last:.3f}, target={target:.3f}, error={abs(last-target):.3f})."
            )
            if args.disable_on_fail:
                print("--disable-on-fail set, so disabling torque anyway.")
                should_disconnect = True
            else:
                print(
                    "Keeping connection/torque ON because target was not reached. "
                    "Press Ctrl-C to stop and disable torque."
                )
                try:
                    while True:
                        robot.send_action({pos_key: target})
                        time.sleep(0.2)
                except KeyboardInterrupt:
                    print("\nCtrl-C received. Disabling torque now.")
                    should_disconnect = True
    finally:
        if should_disconnect and robot is not None:
            try:
                robot.disconnect()
                print("Disconnected. Torque should be off.")
            except Exception as exc:
                print(f"Disconnect warning: {exc}", file=sys.stderr)

    return 0 if reached else 2


if __name__ == "__main__":
    raise SystemExit(main())
