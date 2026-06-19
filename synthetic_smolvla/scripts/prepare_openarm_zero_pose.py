#!/usr/bin/env python3
"""Move the real OpenArm to a guarded arm-zero pose only.

Default behavior is dry-run. Real motion requires --mirror-real and the exact
--real-confirm phrase. This script uses the normal OpenArm helper prepare_start
path, so it commands joint position targets and audits readback before success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirror_sinks import (  # noqa: E402
    DEFAULT_REAL_HELPER,
    DEFAULT_REAL_HOST,
    DEFAULT_REAL_REPO,
    DEFAULT_REAL_USER,
    DEFAULT_START_POSE_TOLERANCE_DEG,
    REQUIRED_REAL_CONFIRMATION,
    MirrorSafetyError,
    RealMirrorConfig,
    RealMirrorSink,
    clamp_command_deg,
    start_pose_from_config,
)
from sim_contract import JOINT_NAMES, REPO_ROOT, load_yaml_config, normalize_side  # noqa: E402


DEFAULT_CONFIG = "synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml"


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def default_real_port_for_side(side: str) -> str:
    return "can0" if side == "right" else "can1"


def active_side_from_config(config_path: str | Path) -> str:
    config = load_yaml_config(config_path)
    return normalize_side(config.get("scene", {}).get("active_arm", "right"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--side", choices=["left", "right"], default=None, help="real arm side; defaults to scene active arm")
    parser.add_argument("--mirror-real", action="store_true", help="actually move the guarded real OpenArm")
    parser.add_argument("--real-confirm", default="", help="exact real-motion acknowledgement phrase")
    parser.add_argument(
        "--gripper-deg",
        type=float,
        default=None,
        help="optional gripper target; default sends gripper to 0 during zero pose",
    )
    parser.add_argument("--max-joint-delta-deg", type=float, default=1.0)
    parser.add_argument(
        "--real-helper-max-rel-deg",
        type=float,
        default=None,
        help="OpenArmFollower max_relative_target sent to the helper",
    )
    parser.add_argument("--real-pose-tolerance-deg", type=float, default=DEFAULT_START_POSE_TOLERANCE_DEG)
    parser.add_argument("--real-pose-timeout-sec", type=float, default=120.0)
    parser.add_argument("--real-pose-hold-sec", type=float, default=0.05)
    parser.add_argument("--watchdog-timeout-sec", type=float, default=2.0)
    parser.add_argument("--hold-interval-sec", type=float, default=0.03)
    parser.add_argument(
        "--hold-final",
        dest="hold_final",
        action="store_true",
        default=True,
        help="hold the zero pose after preparation (default)",
    )
    parser.add_argument(
        "--no-hold-final",
        dest="hold_final",
        action="store_false",
        help="exit immediately after reaching zero pose; this releases the helper",
    )
    parser.add_argument(
        "--disable-gripper-real",
        dest="disable_gripper_real",
        action="store_true",
        default=True,
        help="do not send gripper commands to the real robot (default)",
    )
    parser.add_argument(
        "--enable-gripper-real",
        dest="disable_gripper_real",
        action="store_false",
        help="allow real gripper commands",
    )
    parser.add_argument("--real-port", default=None, help="real CAN port; defaults from side")
    parser.add_argument("--real-host", default=DEFAULT_REAL_HOST)
    parser.add_argument("--real-user", default=DEFAULT_REAL_USER)
    parser.add_argument("--real-repo", default=DEFAULT_REAL_REPO)
    parser.add_argument("--real-helper", default=DEFAULT_REAL_HELPER)
    parser.add_argument("--real-request-timeout-sec", type=float, default=8.0)
    parser.add_argument("--real-connect-timeout-sec", type=float, default=5.0)
    parser.add_argument("--real-connect-retries", type=int, default=3)
    parser.add_argument("--real-connect-retry-delay-sec", type=float, default=1.5)
    return parser


def _build_real_config(args: argparse.Namespace, *, side: str) -> RealMirrorConfig:
    port = args.real_port or default_real_port_for_side(side)
    return RealMirrorConfig(
        side=side,
        port=port,
        confirm=args.real_confirm,
        rate_hz=1.0,
        max_joint_delta_deg=args.max_joint_delta_deg,
        watchdog_timeout_sec=args.watchdog_timeout_sec,
        disable_gripper_real=args.disable_gripper_real,
        helper_max_relative_target_deg=args.real_helper_max_rel_deg,
        host=args.real_host,
        user=args.real_user,
        repo=args.real_repo,
        helper=args.real_helper,
        connect_timeout_sec=args.real_connect_timeout_sec,
        request_timeout_sec=args.real_request_timeout_sec,
        start_pose_tolerance_deg=args.real_pose_tolerance_deg,
        start_pose_timeout_sec=args.real_pose_timeout_sec,
        start_pose_hold_sec=args.real_pose_hold_sec,
        connect_retries=args.real_connect_retries,
        connect_retry_delay_sec=args.real_connect_retry_delay_sec,
        hold_interval_sec=args.hold_interval_sec,
    )


def _target_record(label: str, target: list[float]) -> dict[str, object]:
    return {
        "label": label,
        "command_deg": [round(float(value), 6) for value in target],
        "arm_command_deg": {
            joint: round(float(target[index]), 6) for index, joint in enumerate(JOINT_NAMES)
        },
        "gripper_command_deg": round(float(target[7]), 6),
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    config_path = _abs(args.config)
    side = normalize_side(args.side or active_side_from_config(config_path))
    gripper = 0.0 if args.gripper_deg is None else float(args.gripper_deg)
    zero_pose = clamp_command_deg(side, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]).command_deg

    plan = {
        "ok": True,
        "side": side,
        "real_motion_requested": bool(args.mirror_real),
        "real_gripper_disabled": bool(args.disable_gripper_real),
        "target": _target_record("arm_zero_pose", zero_pose),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.mirror_real:
        print("[zero-pose] dry-run only; no real robot commands sent.", file=sys.stderr, flush=True)
        return 0
    if args.real_confirm != REQUIRED_REAL_CONFIRMATION:
        raise MirrorSafetyError(
            "Real motion refused: pass --real-confirm "
            f"{REQUIRED_REAL_CONFIRMATION!r} only while physically at the robot."
        )

    real = RealMirrorSink(_build_real_config(args, side=side))
    try:
        ready = real.start()
        print(
            json.dumps(
                {
                    "event": "real_helper_ready",
                    "real_side": side,
                    "real_state_deg": ready.get("state_deg"),
                    "real_gripper_disabled": bool(args.disable_gripper_real),
                },
                indent=2,
                sort_keys=True,
            )
        )

        print("[zero-pose] moving to guarded arm-zero pose with default helper method.", file=sys.stderr, flush=True)
        result = real.prepare_start_pose(zero_pose)
        audit = real.audit_prepared_start_pose(zero_pose)
        print(
            json.dumps(
                {
                    "event": "zero_pose_reached",
                    "prepare_result": result,
                    "audit": audit,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not args.hold_final:
            print("[zero-pose] done; exiting and releasing helper.", file=sys.stderr, flush=True)
            return 0
        print("[zero-pose] holding zero pose. Use Ctrl-C when ready to release.", file=sys.stderr, flush=True)
        try:
            import time

            while True:
                real.start_keepalive()
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[zero-pose] releasing helper.", file=sys.stderr, flush=True)
        return 0
    except MirrorSafetyError as exc:
        print(f"[zero-pose] safety abort: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        real.close()


if __name__ == "__main__":
    raise SystemExit(main())
