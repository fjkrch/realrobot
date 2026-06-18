#!/usr/bin/env python3
"""Replay a saved SmolVLA joint-target trajectory on the guarded real OpenArm path.

Default behavior is audit-only. Real motion requires --mirror-real,
--prepare-real-start-pose, and the exact --real-confirm phrase.
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
    max_abs_arm_delta_deg,
    max_abs_target_delta_deg,
    start_pose_from_config,
)
from sim_contract import JOINT_NAMES, REPO_ROOT, load_yaml_config, validate_scene_config  # noqa: E402


DEFAULT_CONFIG = "synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml"


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def default_real_port_for_side(side: str) -> str:
    return "can0" if side == "right" else "can1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="saved command JSONL from interactive_vla_isaac.py")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--side", choices=["left", "right"], default=None, help="real arm side; defaults to scene active arm")
    parser.add_argument("--expected-steps", type=int, default=None)
    parser.add_argument("--audit-output-json", default=None, help="optional path for audit summary JSON")
    parser.add_argument("--replay-log", default=None, help="optional JSONL path for replay events")
    parser.add_argument("--mirror-real", action="store_true", help="actually replay on the guarded real OpenArm path")
    parser.add_argument("--real-confirm", default="", help="exact real-motion acknowledgement phrase")
    parser.add_argument(
        "--prepare-real-start-pose",
        action="store_true",
        help="move the real arm to the config reset pose before replay",
    )
    parser.add_argument("--read-real-state", action="store_true", help="read current real arm+gripper state before replay")
    parser.add_argument(
        "--real-preflight-only",
        action="store_true",
        help="prepare/audit the real start pose, then exit before replaying trajectory",
    )
    parser.add_argument(
        "--mirror-rate-hz",
        "--replay-rate-hz",
        dest="mirror_rate_hz",
        type=float,
        default=2.0,
        help="continuous saved-trajectory replay send rate",
    )
    parser.add_argument("--max-joint-delta-deg", type=float, default=3.0)
    parser.add_argument(
        "--start-pose-max-joint-delta-deg",
        type=float,
        default=None,
        help="compatibility option; normal start-pose preparation uses the guarded helper prepare_start path",
    )
    parser.add_argument(
        "--start-pose-gripper-max-delta-deg",
        type=float,
        default=None,
        help="compatibility option; normal start-pose preparation uses the guarded helper prepare_start path",
    )
    parser.add_argument(
        "--start-pose-rate-hz",
        type=float,
        default=None,
        help="compatibility option; normal start-pose preparation uses the guarded helper prepare_start path",
    )
    parser.add_argument("--watchdog-timeout-sec", type=float, default=2.0)
    parser.add_argument("--hold-interval-sec", type=float, default=0.2)
    parser.add_argument(
        "--hold-final",
        dest="hold_final",
        action="store_true",
        default=True,
        help="keep the real helper alive after replay so the final pose holds (default)",
    )
    parser.add_argument(
        "--no-hold-final",
        dest="hold_final",
        action="store_false",
        help="exit immediately after replay; this releases the helper",
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
        help="allow real gripper commands; not recommended for first replay tests",
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
        default=25.0,
        help="seconds to let the normal helper prepare_start path reach the configured start pose",
    )
    parser.add_argument(
        "--real-start-pose-hold-sec",
        type=float,
        default=0.3,
        help="seconds the real start pose must stay within tolerance before replay begins",
    )
    parser.add_argument("--first-real-target-tolerance-deg", type=float, default=DEFAULT_FIRST_TARGET_TOLERANCE_DEG)
    parser.add_argument("--real-connect-retries", type=int, default=3)
    parser.add_argument("--real-connect-retry-delay-sec", type=float, default=1.5)
    return parser


def _load_trajectory(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "command_deg" not in record:
                continue
            command = record["command_deg"]
            if not isinstance(command, list) or len(command) != 8:
                raise MirrorSafetyError(f"{path}:{line_no}: command_deg must be an 8-value list.")
            record["_line_no"] = line_no
            record["_sequence"] = len(records)
            records.append(record)
    return records


def _audit_trajectory(
    records: list[dict[str, Any]],
    *,
    side: str,
    expected_steps: int | None,
    max_joint_delta_deg: float,
    start_pose_deg: list[float],
    first_target_tolerance_deg: float,
    disable_gripper_real: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    include_gripper = not disable_gripper_real
    delta_label = "target" if include_gripper else "arm"
    if not records:
        errors.append("trajectory has no command records")
    if expected_steps is not None and len(records) != expected_steps:
        errors.append(f"expected {expected_steps} command records, found {len(records)}")

    previous: list[float] | None = None
    max_observed_delta = 0.0
    clamp_events = 0
    first_target_delta = None
    for index, record in enumerate(records):
        try:
            clamped = clamp_command_deg(side, record["command_deg"])
        except Exception as exc:  # noqa: BLE001 - reported as audit error
            errors.append(f"line {record.get('_line_no', '?')}: invalid command: {exc}")
            continue
        command = [float(value) for value in record["command_deg"]]
        changed = [
            joint_index
            for joint_index, value in enumerate(command)
            if abs(float(value) - float(clamped.command_deg[joint_index])) > 1e-5
        ]
        if changed:
            names = list(JOINT_NAMES) + ["gripper"]
            clamp_events += len(changed)
            errors.append(
                f"line {record.get('_line_no', '?')}: command outside safe limits for "
                + ", ".join(names[joint_index] for joint_index in changed)
            )
        if previous is not None:
            if include_gripper:
                delta = max_abs_target_delta_deg(command, previous, include_gripper=True)
            else:
                delta = max_abs_arm_delta_deg(command, previous)
            max_observed_delta = max(max_observed_delta, delta)
            if delta > max_joint_delta_deg:
                errors.append(
                    f"line {record.get('_line_no', '?')}: max {delta_label} delta {delta:.6f} deg exceeds "
                    f"{max_joint_delta_deg:.6f} deg"
                )
        previous = command
        if index == 0:
            if include_gripper:
                first_target_delta = max_abs_target_delta_deg(command, start_pose_deg, include_gripper=True)
            else:
                first_target_delta = max_abs_arm_delta_deg(command, start_pose_deg)
            if first_target_delta > first_target_tolerance_deg:
                errors.append(
                    f"first target {delta_label} delta from configured start pose "
                    f"{first_target_delta:.6f} deg exceeds {first_target_tolerance_deg:.6f} deg"
                )

    return {
        "ok": not errors,
        "errors": errors,
        "records": len(records),
        "side": side,
        "expected_steps": expected_steps,
        "max_allowed_arm_delta_deg": float(max_joint_delta_deg),
        "max_observed_arm_delta_deg": round(float(max_observed_delta), 6),
        "max_allowed_checked_delta_deg": float(max_joint_delta_deg),
        "max_observed_checked_delta_deg": round(float(max_observed_delta), 6),
        "delta_check_includes_gripper": include_gripper,
        "first_target_delta_from_start_deg": None
        if first_target_delta is None
        else round(float(first_target_delta), 6),
        "first_target_tolerance_deg": float(first_target_tolerance_deg),
        "clamp_events": int(clamp_events),
        "real_gripper_disabled": bool(disable_gripper_real),
        "gripper_sent_to_real": not disable_gripper_real,
    }


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
        start_pose_max_joint_delta_deg=args.start_pose_max_joint_delta_deg,
        start_pose_gripper_max_delta_deg=args.start_pose_gripper_max_delta_deg,
        start_pose_rate_hz=args.start_pose_rate_hz,
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


def _validate_real_flags(args: argparse.Namespace) -> None:
    if args.real_preflight_only and not args.mirror_real:
        raise SystemExit("--real-preflight-only requires --mirror-real.")
    if args.prepare_real_start_pose and not args.mirror_real:
        raise SystemExit("--prepare-real-start-pose can move the real robot and requires --mirror-real.")
    if args.read_real_state and not args.mirror_real:
        raise SystemExit("--read-real-state uses the real helper here and requires --mirror-real.")
    if args.mirror_real and not args.prepare_real_start_pose:
        raise SystemExit("--mirror-real requires --prepare-real-start-pose before replay.")
    if args.mirror_real and args.real_confirm != REQUIRED_REAL_CONFIRMATION:
        raise SystemExit(
            "Refusing real replay. Pass --real-confirm "
            f"{REQUIRED_REAL_CONFIRMATION!r} only while physically at the robot with e-stop ready."
        )


def _open_replay_log(args: argparse.Namespace, trajectory_path: Path):
    if not args.replay_log:
        return None
    path = _abs(args.replay_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    handle.write(
        json.dumps(
            {
                "type": "replay_start",
                "trajectory": str(trajectory_path),
                "mirror_rate_hz": args.mirror_rate_hz,
                "max_joint_delta_deg": args.max_joint_delta_deg,
                "disable_gripper_real": args.disable_gripper_real,
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def _write_replay_event(handle, payload: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


def main() -> int:
    args = build_arg_parser().parse_args()
    _validate_real_flags(args)

    config = load_yaml_config(args.config)
    validate_scene_config(config)
    side = args.side or str(config["scene"].get("active_arm", "right"))
    if not str(args.trajectory).strip():
        raise SystemExit(
            "--trajectory is empty. Set SPLIT_TRAJ to a saved *_commands.jsonl file, "
            "or pass the trajectory path directly."
        )
    trajectory_path = _abs(args.trajectory)
    if not trajectory_path.is_file():
        raise SystemExit(f"--trajectory must be a JSONL file, got: {trajectory_path}")
    records = _load_trajectory(trajectory_path)
    start_pose_deg = start_pose_from_config(args.config, side)
    audit = _audit_trajectory(
        records,
        side=side,
        expected_steps=args.expected_steps,
        max_joint_delta_deg=args.max_joint_delta_deg,
        start_pose_deg=start_pose_deg,
        first_target_tolerance_deg=args.first_real_target_tolerance_deg,
        disable_gripper_real=args.disable_gripper_real,
    )
    audit["trajectory"] = str(trajectory_path)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if args.audit_output_json:
        out = _abs(args.audit_output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.mirror_real:
        print("[replay-openarm] audit-only complete; no real robot commands sent.", file=sys.stderr, flush=True)
        return 0 if audit["ok"] else 2
    if not audit["ok"]:
        raise SystemExit("[replay-openarm] refusing real replay because trajectory audit failed.")

    real = RealMirrorSink(_build_real_config(args, side=side))
    replay_log = None
    try:
        replay_log = _open_replay_log(args, trajectory_path)
        ready = real.start()
        print("[replay-openarm] real helper ready.", file=sys.stderr, flush=True)
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
        print("[replay-openarm] moving real arm to configured start pose.", file=sys.stderr, flush=True)
        real.prepare_start_pose(start_pose_deg)
        start_audit = real.audit_prepared_start_pose(start_pose_deg)
        print(
            "[replay-openarm] real start-pose audit passed: "
            + json.dumps(start_audit, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        _write_replay_event(replay_log, {"type": "start_pose_audit", **start_audit})
        if args.real_preflight_only:
            print("[replay-openarm] preflight-only requested; exiting before replay.", file=sys.stderr, flush=True)
            return 0

        print(f"[replay-openarm] replaying {len(records)} saved commands.", file=sys.stderr, flush=True)
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
        print(
            "[replay-openarm] replay finished; holding final pose. "
            "Type q when ready to release/exit, reset to return to start pose, or hold to keep holding.",
            file=sys.stderr,
            flush=True,
        )
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
                    print("[replay-openarm] reset requested; moving real arm to start pose.", file=sys.stderr, flush=True)
                    real.prepare_start_pose(start_pose_deg)
                    _write_replay_event(replay_log, {"type": "reset_to_start"})
                    continue
                if typed in {"h", "hold", "wait", "stay", ""}:
                    print("[replay-openarm] holding current pose.", file=sys.stderr, flush=True)
                    continue
                print("[replay-openarm] unknown command; use reset, hold, or q.", file=sys.stderr, flush=True)
        return 0
    except MirrorSafetyError as exc:
        print(f"[replay-openarm] safety abort: {exc}", file=sys.stderr, flush=True)
        _write_replay_event(replay_log, {"type": "safety_abort", "error": str(exc)})
        return 2
    finally:
        if replay_log is not None:
            replay_log.close()
        real.close()


if __name__ == "__main__":
    raise SystemExit(main())
