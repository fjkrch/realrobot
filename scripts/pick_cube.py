#!/usr/bin/env python3
"""Scripted cube pick for OpenArm: teach poses by hand, verify in Isaac, replay real.

This is a "fixed taught location" pick. There is no vision and no IK. You teach a
short sequence of whole-arm poses once (home -> pregrasp -> descend -> grasp ->
lift -> retreat); after that the same joint angles are replayed every run.

Built on the same OpenArmFollower API and safety pattern as
scripts/open_gripper_small.py, scripts/move_joint.py and scripts/move_arm.py.

Three modes:
  --teach        Torque OFF. Move the arm by hand to each named keyframe and
                 press Enter to capture it. Writes real poses into the JSON and
                 marks it taught.
  --sim-only     No CAN. Mirror each pose into Isaac Lab only, to watch the
                 sequence before touching hardware.
  (real replay)  Default. Requires --i-am-at-robot. Plays the taught poses on the
                 real robot, each pose creeping via max_relative_target, gripper
                 included. Refused while the file is not taught unless
                 --force-untaught is given.

Safety:
- Run real replay only while physically at the robot with e-stop ready.
- Uses connect(calibrate=False); does not calibrate or re-mark motor zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig


DEFAULT_ISAACLAB_MIRROR_SERVER = "http://10.10.10.1:8765"
JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
ALL_MOTORS = JOINT_NAMES + ["gripper"]
DEFAULT_WAYPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_cube_waypoints.json")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def require_can_interface(port: str) -> None:
    result = subprocess.run(
        ["ip", "link", "show", port],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{port} does not exist. Start CAN first (sudo ./scripts/can_up.sh).")


def load_waypoints(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"Waypoints file not found: {path}")
    with open(path) as fh:
        data = json.load(fh)
    if not data.get("waypoints"):
        raise SystemExit(f"No waypoints in {path}.")
    return data


def save_waypoints(path: str, data: dict) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def build_robot(port: str, side: str, robot_id: str, max_rel: float) -> OpenArmFollower:
    return OpenArmFollower(
        OpenArmFollowerConfig(
            port=port,
            side=side,
            id=robot_id,
            max_relative_target={name: max_rel for name in ALL_MOTORS},
        )
    )


def connect_with_retries(make, *, retries: int, delay: float):
    """connect(calibrate=False) with handshake retries (transient miss after can_up)."""
    last_exc = None
    for attempt in range(1, retries + 1):
        robot = make()
        try:
            robot.connect(calibrate=False)
            if attempt > 1:
                print(f"Connected on attempt {attempt}.")
            return robot
        except Exception as exc:
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
        "Check 24 V supply ON, e-stop released, and `lerobot-setup-can "
        "--mode=test --interfaces=<port>` shows 8/8 motors."
    )


def mirror_to_isaac(command: str, *, server: str, timeout_sec: float) -> None:
    if not server:
        return
    payload = json.dumps({"command": command}).encode("utf-8")
    request = Request(server.rstrip("/") + "/command", data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    token = os.environ.get("BRIDGE_TOKEN", "")
    if token:
        request.add_header("X-Bridge-Token", token)
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            print(f"[Isaac] {command!r}: HTTP {response.status}")
    except HTTPError as exc:
        print(f"[Isaac] warning HTTP {exc.code}", file=sys.stderr)
    except (URLError, TimeoutError) as exc:
        print(f"[Isaac] warning: could not reach {server}: {exc}", file=sys.stderr)


def mirror_pose(side: str, pose: dict, *, server: str, timeout_sec: float) -> None:
    """Mirror one waypoint into Isaac as a whole-body + gripper command."""
    joints = " ".join(f"{float(pose.get(j, 0.0)):g}" for j in JOINT_NAMES)
    mirror_to_isaac(f"whole body {side} {joints}", server=server, timeout_sec=timeout_sec)
    if "gripper" in pose:
        mirror_to_isaac(f"gripper target {float(pose['gripper']):.3f} deg",
                        server=server, timeout_sec=timeout_sec)


def interpolate_poses(a: dict, b: dict, deg_per_step: float) -> list[dict]:
    """Linear sub-steps from pose a to pose b so the sim tracks the path smoothly.

    The last frame equals b exactly, so the motion ends on the target pose.
    """
    motors = [m for m in ALL_MOTORS if m in b]
    max_delta = max((abs(float(b[m]) - float(a.get(m, b[m]))) for m in motors), default=0.0)
    steps = max(1, int(math.ceil(max_delta / max(deg_per_step, 0.1))))
    frames = []
    for s in range(1, steps + 1):
        frac = s / steps
        frames.append(
            {m: float(a.get(m, b[m])) + frac * (float(b[m]) - float(a.get(m, b[m]))) for m in motors}
        )
    return frames


def validate_pose(pose: dict, joint_limits: dict, name: str) -> None:
    bad = []
    for motor, val in pose.items():
        if motor in joint_limits:
            lo, hi = joint_limits[motor]
            if val < lo or val > hi:
                bad.append(f"{motor}={val:.1f} (allowed {lo:.1f}..{hi:.1f})")
    if bad:
        raise SystemExit(f"Waypoint {name!r} out of range:\n  " + "\n  ".join(bad))


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def run_teach(args, data: dict) -> int:
    port, side = data["port"], data["side"]
    require_can_interface(port)
    robot = build_robot(port, side, args.id, data.get("max_rel", 5.0))

    print("TEACH mode: torque will be DISABLED so you can move the arm by hand.")
    print("For each keyframe, position the arm (and gripper) then press Enter to capture.")
    if not args.yes and input("Type TEACH to continue: ").strip() != "TEACH":
        print("Cancelled.")
        return 1

    # Low-level connect + torque off, like scripts/disable_torque.py, with retries.
    last_exc = None
    for attempt in range(1, args.connect_retries + 1):
        try:
            robot.bus.connect()
            break
        except Exception as exc:
            last_exc = exc
            print(f"[bus connect] attempt {attempt} failed: {exc}", file=sys.stderr)
            try:
                robot.bus.disconnect()
            except Exception:
                pass
            if attempt < args.connect_retries:
                time.sleep(args.connect_retry_delay)
    else:
        raise SystemExit(f"Bus connect failed: {last_exc}")

    joint_limits = getattr(robot.config, "joint_limits", {}) or {}
    quit_teach = False
    try:
        robot.bus.disable_torque()
        print("Torque disabled. Arm is free to move.\n")
        for wp in data["waypoints"]:
            if quit_teach:
                break
            name = wp["name"]
            while True:  # re-prompt this keyframe until it is captured within limits
                ans = input(f"  Move arm to '{name}' then Enter to capture (s=skip, q=quit): ").strip().lower()
                if ans == "q":
                    quit_teach = True
                    break
                if ans == "s":
                    print(f"    skipped {name} (kept previous pose)")
                    break
                states = robot.bus.sync_read_all_states()
                pose = {m: round(float(states.get(m, {}).get("position", 0.0)), 2) for m in ALL_MOTORS}
                over = []
                for m in ALL_MOTORS:
                    if m in joint_limits:
                        lo, hi = joint_limits[m]
                        if pose[m] < lo or pose[m] > hi:
                            over.append(f"{m}={pose[m]:+.1f}  (limit {lo:.0f}..{hi:.0f})")
                if over:
                    print(f"    !! '{name}' is OUT OF LIMIT - not saved. Move inside range and teach again:")
                    for o in over:
                        print(f"         {o}")
                    continue  # ask for this same keyframe again
                wp["pose"] = pose
                print(f"    captured {name}: " + ", ".join(f"{m}={pose[m]:+.1f}" for m in ALL_MOTORS))
                break
    finally:
        robot.bus.disconnect()
        print("\nBus disconnected.")

    if quit_teach:
        print(f"Quit before finishing. 'taught' left as {data.get('taught', False)} "
              "(not all poses captured, so real replay may stay blocked).")
    else:
        data["taught"] = True
        print("All keyframes captured within limits. Marking taught=true.")
    save_waypoints(args.waypoints, data)
    print(f"Saved to {args.waypoints}.")
    return 0


def run_sim_only(args, data: dict) -> int:
    side = data["side"]
    waypoints = data["waypoints"]
    print("SIM-ONLY: mirroring the pick sequence into Isaac Lab. No CAN, no real motion.")
    print(f"  speed: {args.sim_deg_per_step:.1f} deg/step, {args.sim_dt:.2f}s/step, "
          f"settle {args.sim_settle_sec:.1f}s at each pose.")

    # Start at the first pose, then interpolate to each following pose and hold.
    current = {m: float(v) for m, v in waypoints[0]["pose"].items()}
    print(f"  -> {waypoints[0]['name']} (start)")
    mirror_pose(side, current, server=args.isaac_mirror_server,
                timeout_sec=args.isaac_mirror_timeout_sec)
    time.sleep(args.sim_settle_sec)

    for wp in waypoints[1:]:
        print(f"  -> {wp['name']}")
        for pose in interpolate_poses(current, wp["pose"], args.sim_deg_per_step):
            mirror_pose(side, pose, server=args.isaac_mirror_server,
                        timeout_sec=args.isaac_mirror_timeout_sec)
            time.sleep(args.sim_dt)
        current = {m: float(v) for m, v in wp["pose"].items()}
        # Hold on the reached pose so the sim settles before the next waypoint.
        time.sleep(max(args.sim_settle_sec, wp.get("dwell", 0.0)))
    print("Sim sequence done.")
    return 0


def settle_to_pose(robot, pose: dict, *, tolerance, hold_sec, timeout_sec) -> bool:
    """Repeatedly command the pose; each send is creep-clamped by max_relative_target."""
    motors = [m for m in ALL_MOTORS if m in pose]
    action = {f"{m}.pos": float(pose[m]) for m in motors}
    robot.send_action(action)
    deadline = time.monotonic() + timeout_sec
    stable_since = None
    max_err = None
    while time.monotonic() < deadline:
        obs = robot.get_observation()
        errs = {m: abs(float(obs.get(f"{m}.pos", 0.0)) - float(pose[m])) for m in motors}
        max_err = max(errs.values())
        worst = max(errs, key=errs.get)
        print(f"\r    max err {max_err:5.2f} deg @ {worst}      ", end="", flush=True)
        if max_err <= tolerance:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= hold_sec:
                print()
                return True
        else:
            stable_since = None
            robot.send_action(action)
        time.sleep(0.05)
    print()
    return False


def run_real(args, data: dict) -> int:
    if not args.i_am_at_robot:
        raise SystemExit("Refusing to run on the real robot. Add --i-am-at-robot.")
    if not data.get("taught") and not args.force_untaught:
        raise SystemExit(
            "These waypoints are not taught yet (taught=false). Run with --teach first,\n"
            "or pass --force-untaught if you really intend to run the placeholder poses."
        )

    port, side = data["port"], data["side"]
    tol = data.get("tolerance_deg", 1.5)
    hold = data.get("hold_sec", 0.3)
    timeout = data.get("settle_timeout_sec", 25.0)
    require_can_interface(port)

    def make():
        return build_robot(port, side, args.id, data.get("max_rel", 5.0))

    # Validate every pose against the live joint limits before moving.
    probe = make()
    joint_limits = getattr(probe.config, "joint_limits", {}) or {}
    for wp in data["waypoints"]:
        validate_pose(wp["pose"], joint_limits, wp["name"])

    print(f"REAL pick on the {side} arm ({port}). Sequence:")
    for wp in data["waypoints"]:
        print(f"  {wp['name']}: " + ", ".join(f"{m}={wp['pose'][m]:+.0f}" for m in ALL_MOTORS if m in wp["pose"]))
    print(f"Per-step clamp {data.get('max_rel', 5.0):.1f} deg/motor. Keep e-stop ready.")
    if not args.yes and input("Type PICK to continue: ").strip() != "PICK":
        print("Cancelled.")
        return 1

    robot = None
    ok = True
    try:
        robot = connect_with_retries(make, retries=args.connect_retries, delay=args.connect_retry_delay)
        for wp in data["waypoints"]:
            print(f"-> {wp['name']}")
            if not args.no_isaac_mirror:
                mirror_pose(side, wp["pose"], server=args.isaac_mirror_server,
                            timeout_sec=args.isaac_mirror_timeout_sec)
            reached = settle_to_pose(robot, wp["pose"], tolerance=tol, hold_sec=hold, timeout_sec=timeout)
            if not reached:
                ok = False
                print(f"   '{wp['name']}' not reached within {timeout:.0f}s.")
                if not args.continue_on_miss:
                    print("   Stopping sequence (use --continue-on-miss to push through).")
                    break
            time.sleep(wp.get("dwell", 0.3))
        if ok:
            print("Pick sequence complete.")
    finally:
        if robot is not None:
            try:
                robot.disconnect()
                print("Disconnected. Torque should be off.")
            except Exception as exc:
                print(f"Disconnect warning: {exc}", file=sys.stderr)
    return 0 if ok else 2


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--waypoints", default=DEFAULT_WAYPOINTS, help="waypoints JSON file")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--teach", action="store_true", help="capture poses by hand (torque off)")
    mode.add_argument("--sim-only", action="store_true", help="mirror to Isaac only, no CAN")
    p.add_argument("--i-am-at-robot", action="store_true", help="required for real replay")
    p.add_argument("--yes", action="store_true", help="skip confirmation prompts")
    p.add_argument("--force-untaught", action="store_true", help="allow real replay of placeholder poses")
    p.add_argument("--continue-on-miss", action="store_true", help="keep going if a waypoint is not reached")
    p.add_argument("--id", default="pick_noncal", help="fresh id with no calibration file")
    p.add_argument("--connect-retries", type=int, default=3)
    p.add_argument("--connect-retry-delay", type=float, default=1.5)
    p.add_argument("--sim-deg-per-step", type=float, default=2.0,
                   help="--sim-only: degrees per interpolation sub-step (smaller = slower, smoother)")
    p.add_argument("--sim-dt", type=float, default=0.1,
                   help="--sim-only: seconds between interpolation sub-steps")
    p.add_argument("--sim-settle-sec", type=float, default=1.0,
                   help="--sim-only: seconds to hold each pose so the sim reaches it before the next")
    p.add_argument("--isaac-mirror-server",
                   default=os.environ.get("ISAACLAB_MIRROR_SERVER", DEFAULT_ISAACLAB_MIRROR_SERVER))
    p.add_argument("--no-isaac-mirror", action="store_true", help="do not mirror to Isaac during real replay")
    p.add_argument("--isaac-mirror-timeout-sec", type=float, default=2.0)
    args = p.parse_args()

    data = load_waypoints(args.waypoints)

    if args.teach:
        return run_teach(args, data)
    if args.sim_only:
        return run_sim_only(args, data)
    return run_real(args, data)


if __name__ == "__main__":
    raise SystemExit(main())
