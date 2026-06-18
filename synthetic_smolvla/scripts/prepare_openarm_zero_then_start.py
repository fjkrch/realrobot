#!/usr/bin/env python3
"""Prepare the real OpenArm by going arm-zero first, then the sim start pose.

Default behavior is dry-run: it only prints the two targets. Real motion
requires --mirror-real and the exact --real-confirm phrase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
        "--zero-gripper-deg",
        type=float,
        default=None,
        help="optional gripper target for the zero stage; default sends gripper to 0",
    )
    parser.add_argument("--max-joint-delta-deg", type=float, default=1.0)
    parser.add_argument(
        "--zero-stage-sec",
        type=float,
        default=8.0,
        help="seconds to keep sending the arm-zero staging target before moving to configured start pose",
    )
    parser.add_argument("--watchdog-timeout-sec", type=float, default=2.0)
    parser.add_argument("--hold-interval-sec", type=float, default=0.03)
    parser.add_argument(
        "--hold-final",
        dest="hold_final",
        action="store_true",
        default=True,
        help="keep the helper alive and hold the configured start pose after preparation (default)",
    )
    parser.add_argument(
        "--no-hold-final",
        dest="hold_final",
        action="store_false",
        help="exit immediately after reaching the configured start pose; this releases the helper",
    )
    parser.add_argument("--real-port", default=None, help="real CAN port; defaults from side")
    parser.add_argument("--real-host", default=DEFAULT_REAL_HOST)
    parser.add_argument("--real-user", default=DEFAULT_REAL_USER)
    parser.add_argument("--real-repo", default=DEFAULT_REAL_REPO)
    parser.add_argument("--real-helper", default=DEFAULT_REAL_HELPER)
    parser.add_argument("--real-request-timeout-sec", type=float, default=8.0)
    parser.add_argument("--real-connect-timeout-sec", type=float, default=5.0)
    parser.add_argument("--real-start-pose-tolerance-deg", type=float, default=DEFAULT_START_POSE_TOLERANCE_DEG)
    parser.add_argument(
        "--real-start-pose-timeout-sec",
        type=float,
        default=120.0,
        help="seconds allowed for each normal helper prepare_start stage",
    )
    parser.add_argument(
        "--real-start-pose-hold-sec",
        type=float,
        default=0.3,
        help="seconds the configured start pose must stay within tolerance before success",
    )
    parser.add_argument("--real-connect-retries", type=int, default=3)
    parser.add_argument("--real-connect-retry-delay-sec", type=float, default=1.5)
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
        help="allow real gripper commands during zero/start preparation",
    )
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
        host=args.real_host,
        user=args.real_user,
        repo=args.real_repo,
        helper=args.real_helper,
        connect_timeout_sec=args.real_connect_timeout_sec,
        request_timeout_sec=args.real_request_timeout_sec,
        start_pose_tolerance_deg=args.real_start_pose_tolerance_deg,
        start_pose_timeout_sec=args.real_start_pose_timeout_sec,
        start_pose_hold_sec=args.real_start_pose_hold_sec,
        connect_retries=args.real_connect_retries,
        connect_retry_delay_sec=args.real_connect_retry_delay_sec,
        hold_interval_sec=args.hold_interval_sec,
    )


def _target_record(label: str, target: list[float]) -> dict[str, Any]:
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

    start_pose = start_pose_from_config(config_path, side)
    zero_gripper = 0.0 if args.zero_gripper_deg is None else float(args.zero_gripper_deg)
    zero_pose = clamp_command_deg(side, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero_gripper]).command_deg

    plan = {
        "ok": True,
        "side": side,
        "real_motion_requested": bool(args.mirror_real),
        "real_gripper_disabled": bool(args.disable_gripper_real),
        "targets": [
            _target_record("zero_arm_pose", zero_pose),
            _target_record("configured_start_pose", start_pose),
        ],
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.mirror_real:
        print("[zero-then-start] dry-run only; no real robot commands sent.", file=sys.stderr, flush=True)
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

        print(
            "[zero-then-start] sending arm-zero staging pose; not waiting for stable zero.",
            file=sys.stderr,
            flush=True,
        )
        zero_result = real.stage_pose_without_audit(
            zero_pose,
            duration_sec=args.zero_stage_sec,
            label="zero_arm_staging_pose",
        )
        print(json.dumps({"event": "zero_arm_stage_complete", **zero_result}, indent=2, sort_keys=True))

        print("[zero-then-start] moving to configured sim start pose.", file=sys.stderr, flush=True)
        start_result = real.prepare_start_pose(start_pose)
        start_audit = real.audit_prepared_start_pose(start_pose)
        print(
            json.dumps(
                {
                    "event": "configured_start_pose_reached",
                    "prepare_result": start_result,
                    "audit": start_audit,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not args.hold_final:
            print("[zero-then-start] done; exiting and releasing helper.", file=sys.stderr, flush=True)
            return 0
        print("[zero-then-start] done; holding configured start pose. Use Ctrl-C when ready to release.", file=sys.stderr)
        try:
            while True:
                real.start_keepalive()
                import time

                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[zero-then-start] releasing helper.", file=sys.stderr, flush=True)
        return 0
    except MirrorSafetyError as exc:
        print(f"[zero-then-start] safety abort: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        real.close()


if __name__ == "__main__":
    raise SystemExit(main())
