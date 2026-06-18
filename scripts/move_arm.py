#!/usr/bin/env python3
"""Move ALL 7 joints of one OpenArm arm to target angles in one command.

This is the real-robot version of the Isaac mirror command
    "whole body right 0 5 -10 20 0 0 0"
It is a sibling of scripts/open_gripper_small.py and scripts/move_joint.py and
uses the exact same OpenArmFollower API and safety pattern.

Usage (matches the Isaac whole-body phrase):
    python scripts/move_arm.py "whole body right 0 5 -10 20 0 0 0" --i-am-at-robot --yes
    python scripts/move_arm.py "whole body left  0 5 -10 20 0 0 0" --i-am-at-robot --yes

The 7 values are joint_1 .. joint_7 in degrees. Port defaults from side
(right=can0, left=can1) and can be overridden with --port.

Safety:
- Run only while physically at the robot with e-stop ready.
- Uses connect(calibrate=False), so it does not calibrate or re-mark motor zero.
- Each target is clamped to that joint's real joint_limits from the live config;
  the run is refused if ANY value is out of range.
- Every joint creeps toward its target, capped per step by max_relative_target,
  so the arm does not snap.
- Refuses if any joint name or current position cannot be read from the robot.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig


DEFAULT_ISAACLAB_MIRROR_SERVER = "http://10.10.10.1:8765"
NUM_JOINTS = 7
JOINT_NAMES = [f"joint_{i}" for i in range(1, NUM_JOINTS + 1)]


def require_can_interface(port: str) -> None:
    result = subprocess.run(
        ["ip", "link", "show", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{port} does not exist. Start CAN first.")


def default_port_for_side(side: str) -> str:
    # Per OPENARM_ROBOT_HANDOFF.md: can0 = right/follower, can1 = left/leader.
    return "can0" if side == "right" else "can1"


def parse_spec(spec: str, fallback_side: str):
    """Parse 'whole body right 0 5 -10 20 0 0 0' -> ('right', [7 floats])."""
    tokens = [t for t in spec.replace(",", " ").split() if t.lower() not in {"whole", "body"}]
    side = fallback_side
    if tokens and tokens[0].lower() in {"left", "right"}:
        side = tokens.pop(0).lower()
    if len(tokens) != NUM_JOINTS:
        raise SystemExit(
            f"Expected {NUM_JOINTS} joint values for one arm, got {len(tokens)}: {tokens}. "
            f'Example: "whole body right 0 5 -10 20 0 0 0"'
        )
    try:
        values = [float(t) for t in tokens]
    except ValueError as exc:
        raise SystemExit(f"Could not parse joint values as numbers: {exc}")
    return side, values


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
    parser.add_argument(
        "spec",
        help='whole-body phrase, e.g. "whole body right 0 5 -10 20 0 0 0" (7 joint degs)',
    )
    parser.add_argument("--side", choices=["left", "right"], default="right",
                        help="used only if the phrase does not name a side")
    parser.add_argument("--port", default=None, help="CAN interface; defaults from side")
    parser.add_argument(
        "--max-rel",
        type=float,
        default=5.0,
        help="max degrees each joint may move per control step (per-step safety clamp)",
    )
    parser.add_argument("--hold-sec", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=float, default=20.0, help="max time to wait for all joints")
    parser.add_argument("--tolerance-deg", type=float, default=0.5, help="per-joint reached tolerance")
    parser.add_argument(
        "--disable-on-fail",
        action="store_true",
        help="disable torque even if the arm does not reach target",
    )
    parser.add_argument("--connect-retries", type=int, default=3,
                        help="retry the CAN handshake this many times (transient miss after can_up)")
    parser.add_argument("--connect-retry-delay", type=float, default=1.5,
                        help="seconds to wait between connect attempts")
    parser.add_argument("--id", default="arm_noncal", help="fresh id with no calibration file")
    parser.add_argument("--i-am-at-robot", action="store_true", help="required safety acknowledgement")
    parser.add_argument("--yes", action="store_true", help="skip confirmation prompt")
    parser.add_argument(
        "--isaac-mirror-server",
        default=os.environ.get("ISAACLAB_MIRROR_SERVER", DEFAULT_ISAACLAB_MIRROR_SERVER),
        help="laptop Isaac Lab mirror server URL",
    )
    parser.add_argument("--no-isaac-mirror", action="store_true",
                        help="do not mirror this command into Isaac Lab")
    parser.add_argument("--isaac-mirror-timeout-sec", type=float, default=2.0)
    args = parser.parse_args()

    if not args.i_am_at_robot:
        raise SystemExit("Refusing to run. Add --i-am-at-robot when you are physically at the robot.")
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be positive.")
    if args.tolerance_deg <= 0:
        raise SystemExit("--tolerance-deg must be positive.")
    if args.max_rel <= 0:
        raise SystemExit("--max-rel must be positive.")

    side, values = parse_spec(args.spec, args.side)
    port = args.port or default_port_for_side(side)
    targets = dict(zip(JOINT_NAMES, values))

    require_can_interface(port)

    def build_robot() -> OpenArmFollower:
        return OpenArmFollower(
            OpenArmFollowerConfig(
                port=port,
                side=side,
                id=args.id,
                max_relative_target={name: args.max_rel for name in JOINT_NAMES},
            )
        )

    robot = build_robot()

    # Validate every joint name + range against the live config BEFORE moving.
    joint_limits = getattr(robot.config, "joint_limits", {}) or {}
    missing = [n for n in JOINT_NAMES if n not in joint_limits]
    if missing:
        raise SystemExit(
            f"Joints missing from config joint_limits: {missing}. "
            f"Known joints: {sorted(joint_limits.keys())}"
        )
    out_of_range = []
    for name in JOINT_NAMES:
        lo, hi = joint_limits[name]
        if targets[name] < lo or targets[name] > hi:
            out_of_range.append(f"{name}={targets[name]:.2f} (allowed {lo:.1f}..{hi:.1f})")
    if out_of_range:
        raise SystemExit("Refusing out-of-range targets:\n  " + "\n  ".join(out_of_range))

    if robot.is_calibrated:
        raise SystemExit(
            f"Calibration exists for id={args.id!r}. Use a fresh --id to avoid any re-zero path."
        )

    print(f"About to move ALL 7 joints of the {side} arm on {port}:")
    for name in JOINT_NAMES:
        lo, hi = joint_limits[name]
        print(f"  {name}: target {targets[name]:+.2f} deg   (range {lo:.1f}..{hi:.1f})")
    print(f"Per-step clamp (max_relative_target): {args.max_rel:.1f} deg/joint, so the arm creeps to target.")
    print("Keep e-stop ready. If anything moves wrong, hit e-stop / Ctrl-C.")
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
        missing_obs = [f"{n}.pos" for n in JOINT_NAMES if f"{n}.pos" not in obs]
        if missing_obs:
            available = sorted(k for k in obs.keys() if k.endswith(".pos"))
            should_disconnect = True
            raise SystemExit(
                f"Observation missing keys {missing_obs}. Available: {available}"
            )
        action = {f"{n}.pos": targets[n] for n in JOINT_NAMES}
        print("Current vs target:")
        for name in JOINT_NAMES:
            print(f"  {name}: now {float(obs[name + '.pos']):+.2f} -> target {targets[name]:+.2f} deg")

        if not args.no_isaac_mirror:
            mirror_spec = "whole body " + side + " " + " ".join(f"{v:g}" for v in values)
            mirror_to_isaac(mirror_spec, server=args.isaac_mirror_server,
                            timeout_sec=args.isaac_mirror_timeout_sec)

        sent = robot.send_action(action)
        print(f"Sent: {sent}")

        deadline = time.monotonic() + args.timeout_sec
        stable_since = None
        max_err = None
        while time.monotonic() < deadline:
            obs2 = robot.get_observation()
            errs = {n: abs(float(obs2.get(f"{n}.pos", 0.0)) - targets[n]) for n in JOINT_NAMES}
            max_err = max(errs.values())
            worst = max(errs, key=errs.get)
            print(f"\rmax error {max_err:.3f} deg at {worst}        ", end="", flush=True)

            if max_err <= args.tolerance_deg:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= args.hold_sec:
                    reached = True
                    break
            else:
                stable_since = None
                robot.send_action(action)
            time.sleep(0.05)

        print()
        if reached:
            print(
                f"All joints within ±{args.tolerance_deg:.3f} deg, held {args.hold_sec:.2f}s. "
                "Disabling torque now."
            )
            should_disconnect = True
        else:
            print(f"NOT at target after {args.timeout_sec:.1f}s (max error {max_err:.3f} deg).")
            if args.disable_on_fail:
                print("--disable-on-fail set, so disabling torque anyway.")
                should_disconnect = True
            else:
                print(
                    "Keeping torque ON because target not reached. Press Ctrl-C to stop and disable."
                )
                try:
                    while True:
                        robot.send_action(action)
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
