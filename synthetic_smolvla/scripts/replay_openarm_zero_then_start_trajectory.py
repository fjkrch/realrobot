#!/usr/bin/env python3
"""Zero-stage, prepare sim start pose, then replay a saved SmolVLA trajectory.

Default behavior is audit-only. Real motion requires --mirror-real and the exact
--real-confirm phrase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirror_sinks import (  # noqa: E402
    DEFAULT_FIRST_TARGET_TOLERANCE_DEG,
    DEFAULT_REAL_HELPER,
    DEFAULT_REAL_HOST,
    DEFAULT_REAL_REPO,
    DEFAULT_REAL_USER,
    DEFAULT_START_POSE_TOLERANCE_DEG,
    REQUIRED_REAL_CONFIRMATION,
    CommandContext,
    MirrorSafetyError,
    RealMirrorConfig,
    RealMirrorSink,
    clamp_command_deg,
    start_pose_from_config,
)
from replay_openarm_trajectory import (  # noqa: E402
    _audit_trajectory,
    _load_trajectory,
    _open_replay_log,
    _write_replay_event,
)
from sim_contract import JOINT_NAMES, REPO_ROOT, load_yaml_config, normalize_side, validate_scene_config  # noqa: E402


DEFAULT_CONFIG = "synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml"


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def default_real_port_for_side(side: str) -> str:
    return "can0" if side == "right" else "can1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="saved command JSONL to replay after start-pose prep")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--side", choices=["left", "right"], default=None, help="real arm side; defaults to scene active arm")
    parser.add_argument("--expected-steps", type=int, default=None)
    parser.add_argument("--audit-output-json", default=None)
    parser.add_argument("--replay-log", default=None)
    parser.add_argument("--mirror-real", action="store_true", help="actually move and replay on the guarded real OpenArm")
    parser.add_argument("--real-confirm", default="", help="exact real-motion acknowledgement phrase")
    parser.add_argument("--read-real-state", action="store_true", help="print current real state before zero staging")
    parser.add_argument("--real-preflight-only", action="store_true", help="zero-stage and prepare/audit start pose, then exit")
    parser.add_argument(
        "--mirror-rate-hz",
        "--replay-rate-hz",
        dest="mirror_rate_hz",
        type=float,
        default=2.0,
        help="send rate for zero staging and saved trajectory replay",
    )
    parser.add_argument("--max-joint-delta-deg", type=float, default=1.0)
    parser.add_argument(
        "--trajectory-max-joint-delta-deg",
        type=float,
        default=None,
        help=(
            "audit tolerance for the saved trajectory file; default uses --max-joint-delta-deg. "
            "Real helper safety still uses --max-joint-delta-deg."
        ),
    )
    parser.add_argument(
        "--zero-stage-sec",
        type=float,
        default=8.0,
        help="seconds to send arm-zero staging target before configured init pose",
    )
    parser.add_argument(
        "--zero-gripper-deg",
        type=float,
        default=None,
        help="optional gripper target for zero stage; default sends gripper to 0",
    )
    parser.add_argument("--watchdog-timeout-sec", type=float, default=2.0)
    parser.add_argument("--hold-interval-sec", type=float, default=0.03)
    parser.add_argument(
        "--hold-final",
        dest="hold_final",
        action="store_true",
        default=True,
        help="hold final pose after replay (default)",
    )
    parser.add_argument("--no-hold-final", dest="hold_final", action="store_false")
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
    parser.add_argument("--real-start-pose-tolerance-deg", type=float, default=DEFAULT_START_POSE_TOLERANCE_DEG)
    parser.add_argument("--real-start-pose-timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--real-start-pose-hold-sec",
        type=float,
        default=0.3,
        help="seconds the configured init pose must stay within tolerance before VLA replay begins",
    )
    parser.add_argument("--first-real-target-tolerance-deg", type=float, default=DEFAULT_FIRST_TARGET_TOLERANCE_DEG)
    parser.add_argument("--real-connect-retries", type=int, default=3)
    parser.add_argument("--real-connect-retry-delay-sec", type=float, default=1.5)
    return parser


def _build_real_config(args: argparse.Namespace, *, side: str) -> RealMirrorConfig:
    port = args.real_port or default_real_port_for_side(side)
    return RealMirrorConfig(
        side=side,
        port=port,
        confirm=args.real_confirm,
        rate_hz=args.mirror_rate_hz,
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
        first_target_tolerance_deg=args.first_real_target_tolerance_deg,
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


def _validate_real_flags(args: argparse.Namespace) -> None:
    if args.real_preflight_only and not args.mirror_real:
        raise SystemExit("--real-preflight-only requires --mirror-real.")
    if args.read_real_state and not args.mirror_real:
        raise SystemExit("--read-real-state requires --mirror-real.")
    if args.mirror_real and args.real_confirm != REQUIRED_REAL_CONFIRMATION:
        raise SystemExit(
            "Refusing real replay. Pass --real-confirm "
            f"{REQUIRED_REAL_CONFIRMATION!r} only while physically at the robot with e-stop ready."
        )


def main() -> int:
    args = build_arg_parser().parse_args()
    _validate_real_flags(args)

    config = load_yaml_config(args.config)
    validate_scene_config(config)
    side = normalize_side(args.side or str(config["scene"].get("active_arm", "right")))
    if not str(args.trajectory).strip():
        raise SystemExit("--trajectory is empty. Pass a saved *_commands.jsonl path.")
    trajectory_path = _abs(args.trajectory)
    if not trajectory_path.is_file():
        raise SystemExit(f"--trajectory must be a JSONL file, got: {trajectory_path}")

    records = _load_trajectory(trajectory_path)
    start_pose = start_pose_from_config(args.config, side)
    zero_gripper = 0.0 if args.zero_gripper_deg is None else float(args.zero_gripper_deg)
    zero_pose = clamp_command_deg(side, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero_gripper]).command_deg

    audit = _audit_trajectory(
        records,
        side=side,
        expected_steps=args.expected_steps,
        max_joint_delta_deg=args.trajectory_max_joint_delta_deg or args.max_joint_delta_deg,
        start_pose_deg=start_pose,
        first_target_tolerance_deg=args.first_real_target_tolerance_deg,
        disable_gripper_real=args.disable_gripper_real,
    )
    audit["trajectory"] = str(trajectory_path)
    audit["zero_then_start"] = {
        "zero_stage_sec": float(args.zero_stage_sec),
        "zero_pose": _target_record("zero_arm_staging_pose", zero_pose),
        "configured_start_pose": _target_record("configured_start_pose", start_pose),
    }
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if args.audit_output_json:
        out = _abs(args.audit_output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.mirror_real:
        print("[zero-start-replay] audit-only complete; no real robot commands sent.", file=sys.stderr, flush=True)
        return 0 if audit["ok"] else 2
    if not audit["ok"]:
        raise SystemExit("[zero-start-replay] refusing real replay because trajectory audit failed.")

    real = RealMirrorSink(_build_real_config(args, side=side))
    replay_log = None
    try:
        replay_log = _open_replay_log(args, trajectory_path)
        ready = real.start()
        print("[zero-start-replay] real helper ready.", file=sys.stderr, flush=True)
        if args.read_real_state:
            print(
                json.dumps(
                    {
                        "real_state_deg": [round(float(v), 5) for v in ready.get("state_deg", [])],
                        "real_side": side,
                        "real_gripper_disabled": args.disable_gripper_real,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )

        print("[zero-start-replay] sending zero staging pose; not waiting for stable zero.", file=sys.stderr, flush=True)
        zero_result = real.stage_pose_without_audit(
            zero_pose,
            duration_sec=args.zero_stage_sec,
            label="zero_arm_staging_pose",
        )
        _write_replay_event(replay_log, {"type": "zero_stage", **zero_result})

        print("[zero-start-replay] moving to configured sim init pose.", file=sys.stderr, flush=True)
        real.prepare_start_pose(start_pose)
        start_audit = real.audit_prepared_start_pose(start_pose)
        print(
            "[zero-start-replay] init audit passed: " + json.dumps(start_audit, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        _write_replay_event(replay_log, {"type": "start_pose_audit", **start_audit})
        if args.real_preflight_only:
            print("[zero-start-replay] preflight-only requested; exiting before VLA replay.", file=sys.stderr, flush=True)
            return 0

        print(f"[zero-start-replay] replaying {len(records)} VLA commands.", file=sys.stderr, flush=True)
        for sequence, record in enumerate(records):
            command = [float(value) for value in record["command_deg"]]
            context = CommandContext(
                task_index=record.get("task_index"),
                step_index=record.get("step_index", sequence),
                typed_task=record.get("typed_task"),
                policy_instruction=record.get("policy_instruction"),
                target_object=record.get("target_object"),
            )
            real.emit(command, context)
            real_state = real.latest_state_deg()
            _write_replay_event(
                replay_log,
                {
                    "type": "command_replayed",
                    "sequence": sequence,
                    "step_index": context.step_index,
                    "command_deg": [round(float(value), 6) for value in command],
                    "real_state_deg": None
                    if real_state is None
                    else [round(float(value), 6) for value in real_state],
                },
            )

        final_state = real.read_state()
        print("[zero-start-replay] VLA replay finished; holding final pose.", file=sys.stderr, flush=True)
        _write_replay_event(
            replay_log,
            {"type": "replay_finished", "final_state_deg": [round(float(v), 6) for v in final_state]},
        )
        if args.hold_final:
            while True:
                try:
                    typed = input("Replay finished and holding. Type reset, hold, or q: ").strip().lower()
                except EOFError:
                    break
                if typed in {"q", "quit", "exit"}:
                    break
                if typed in {"r", "reset", "home", "start"}:
                    print("[zero-start-replay] reset requested; moving real arm to init pose.", file=sys.stderr, flush=True)
                    real.prepare_start_pose(start_pose)
                    _write_replay_event(replay_log, {"type": "reset_to_start"})
                    continue
                if typed in {"h", "hold", "wait", "stay", ""}:
                    print("[zero-start-replay] holding current pose.", file=sys.stderr, flush=True)
                    continue
                print("[zero-start-replay] unknown command; use reset, hold, or q.", file=sys.stderr, flush=True)
        return 0
    except MirrorSafetyError as exc:
        print(f"[zero-start-replay] safety abort: {exc}", file=sys.stderr, flush=True)
        _write_replay_event(replay_log, {"type": "safety_abort", "error": str(exc)})
        return 2
    finally:
        if replay_log is not None:
            replay_log.close()
        real.close()


if __name__ == "__main__":
    raise SystemExit(main())
