#!/usr/bin/env python3
"""Split a saved SmolVLA trajectory into small joint-delta replay steps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirror_sinks import (  # noqa: E402
    clamp_command_deg,
    max_abs_arm_delta_deg,
    max_abs_target_delta_deg,
    start_pose_from_config,
)
from sim_contract import REPO_ROOT, load_yaml_config, validate_scene_config  # noqa: E402


DEFAULT_CONFIG = "synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml"


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="input saved command JSONL")
    parser.add_argument("--output", required=True, help="output split command JSONL")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--side", choices=["left", "right"], default=None, help="defaults to scene active arm")
    parser.add_argument("--max-joint-delta-deg", type=float, default=3.0)
    parser.add_argument(
        "--samples-per-command",
        type=int,
        default=None,
        help=(
            "emit exactly this many interpolation samples for each input VLA command; "
            "overrides max-delta-based splitting"
        ),
    )
    parser.add_argument(
        "--include-gripper-delta",
        action="store_true",
        help="also split on gripper changes, for trajectories replayed with --enable-gripper-real",
    )
    parser.add_argument(
        "--include-start-interpolation",
        action="store_true",
        default=True,
        help="insert steps from configured start pose to the first command (default)",
    )
    parser.add_argument(
        "--no-start-interpolation",
        dest="include_start_interpolation",
        action="store_false",
        help="do not insert start-to-first-command steps",
    )
    parser.add_argument("--summary-json", default=None, help="optional split summary JSON")
    return parser


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            command = record.get("command_deg")
            if not isinstance(command, list) or len(command) != 8:
                continue
            record["_line_no"] = line_no
            records.append(record)
    if not records:
        raise SystemExit(f"No command records found in {path}")
    return records


def _checked_delta(start: list[float], target: list[float], *, include_gripper: bool) -> float:
    if include_gripper:
        return max_abs_target_delta_deg(target, start, include_gripper=True)
    return max_abs_arm_delta_deg(target, start)


def _interpolate(
    start: list[float],
    target: list[float],
    *,
    max_delta: float,
    include_gripper: bool,
) -> list[list[float]]:
    delta = _checked_delta(start, target, include_gripper=include_gripper)
    steps = max(1, int(math.ceil(delta / max_delta)))
    return [
        [float(start[index] + (target[index] - start[index]) * (step / steps)) for index in range(8)]
        for step in range(1, steps + 1)
    ]


def _interpolate_fixed_samples(start: list[float], target: list[float], *, samples: int) -> list[list[float]]:
    return [
        [float(start[index] + (target[index] - start[index]) * (step / samples)) for index in range(8)]
        for step in range(1, samples + 1)
    ]


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.max_joint_delta_deg <= 0:
        raise SystemExit("--max-joint-delta-deg must be positive.")
    if args.samples_per_command is not None and args.samples_per_command <= 0:
        raise SystemExit("--samples-per-command must be positive.")

    config = load_yaml_config(args.config)
    validate_scene_config(config)
    side = args.side or str(config["scene"].get("active_arm", "right"))
    input_path = _abs(args.trajectory)
    output_path = _abs(args.output)
    records = _load_records(input_path)
    start_pose = start_pose_from_config(args.config, side)

    output_records: list[dict[str, Any]] = []
    previous = start_pose if args.include_start_interpolation else None
    inserted_steps = 0
    max_input_delta = 0.0
    max_output_delta = 0.0

    for input_index, record in enumerate(records):
        target = clamp_command_deg(side, record["command_deg"]).command_deg
        if input_index > 0:
            raw_previous = [float(value) for value in records[input_index - 1]["command_deg"]]
            max_input_delta = max(
                max_input_delta,
                _checked_delta(target, raw_previous, include_gripper=args.include_gripper_delta),
            )
        if previous is None:
            if args.samples_per_command is None:
                split_commands = [target]
            else:
                split_commands = [target for _ in range(args.samples_per_command)]
        else:
            if args.samples_per_command is None:
                split_commands = _interpolate(
                    previous,
                    target,
                    max_delta=args.max_joint_delta_deg,
                    include_gripper=args.include_gripper_delta,
                )
            else:
                split_commands = _interpolate_fixed_samples(
                    previous,
                    target,
                    samples=args.samples_per_command,
                )
            inserted_steps += max(0, len(split_commands) - 1)
        for command in split_commands:
            if output_records:
                max_output_delta = max(
                    max_output_delta,
                    _checked_delta(
                        command,
                        output_records[-1]["command_deg"],
                        include_gripper=args.include_gripper_delta,
                    ),
                )
            out = {
                key: value
                for key, value in record.items()
                if key not in {"_line_no", "_sequence", "command_deg", "sequence", "step_index"}
            }
            out.update(
                {
                    "type": "command",
                    "sequence": len(output_records),
                    "source_sequence": record.get("sequence", input_index),
                    "source_step_index": record.get("step_index", input_index),
                    "step_index": len(output_records),
                    "command_deg": [round(float(value), 6) for value in command],
                    "split_from_saved_trajectory": True,
                    "max_joint_delta_deg": float(args.max_joint_delta_deg),
                    "samples_per_source_command": args.samples_per_command,
                    "split_includes_gripper_delta": bool(args.include_gripper_delta),
                }
            )
            output_records.append(out)
        previous = target

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "side": side,
        "input_records": len(records),
        "output_records": len(output_records),
        "inserted_steps": inserted_steps,
        "max_joint_delta_deg": float(args.max_joint_delta_deg),
        "samples_per_command": args.samples_per_command,
        "split_mode": "fixed_samples" if args.samples_per_command is not None else "max_delta",
        "max_input_arm_delta_deg": round(float(max_input_delta), 6),
        "max_output_arm_delta_deg": round(float(max_output_delta), 6),
        "include_start_interpolation": bool(args.include_start_interpolation),
        "include_gripper_delta": bool(args.include_gripper_delta),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.summary_json:
        summary_path = _abs(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
