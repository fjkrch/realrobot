#!/usr/bin/env python3
"""Audit a dry-run SmolVLA mirror JSONL trace before any real mirror test."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_contract import JOINT_NAMES, SAFE_ARM_LIMITS_DEG, SAFE_GRIPPER_LIMIT_DEG, REPO_ROOT  # noqa: E402
from mirror_sinks import start_pose_from_config, max_abs_arm_delta_deg  # noqa: E402


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="dry-run command JSONL trace")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--max-joint-delta-deg", type=float, required=True)
    parser.add_argument("--expected-steps", type=int, default=None)
    parser.add_argument("--start-pose-config", default=None)
    parser.add_argument("--first-target-tolerance-deg", type=float, default=None)
    parser.add_argument("--output-md", default=None)
    return parser


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_line_no"] = line_no
            records.append(record)
    return records


def _check_record(record: dict[str, Any], *, side: str) -> list[str]:
    errors = []
    command = record.get("command_deg")
    if not isinstance(command, list) or len(command) != 8:
        return [f"line {record['_line_no']}: command_deg must be an 8-value list"]
    for index, value in enumerate(command):
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"line {record['_line_no']}: command_deg[{index}] is not numeric")
            continue
        if not math.isfinite(number):
            errors.append(f"line {record['_line_no']}: command_deg[{index}] is not finite")
            continue
        if index < 7:
            joint = JOINT_NAMES[index]
            low, high = SAFE_ARM_LIMITS_DEG[side][joint]
        else:
            joint = "gripper"
            low, high = SAFE_GRIPPER_LIMIT_DEG
        if number < low or number > high:
            errors.append(
                f"line {record['_line_no']}: {joint}={number:.6f} outside {low:.3f}..{high:.3f} deg"
            )
    if record.get("gripper_sent_to_real") is not False:
        errors.append(f"line {record['_line_no']}: gripper_sent_to_real is not false")
    return errors


def audit(
    records: list[dict[str, Any]],
    *,
    side: str,
    max_delta: float,
    expected_steps: int | None,
    start_pose_deg: list[float] | None = None,
    first_target_tolerance_deg: float | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_steps is not None and len(records) != expected_steps:
        errors.append(f"expected {expected_steps} command records, found {len(records)}")
    if not records:
        errors.append("trace has no command records")

    for record in records:
        errors.extend(_check_record(record, side=side))

    max_observed_delta = 0.0
    previous: list[float] | None = None
    for record in records:
        command = record.get("command_deg")
        if not isinstance(command, list) or len(command) != 8:
            continue
        if previous is not None:
            delta = max(abs(float(command[index]) - float(previous[index])) for index in range(7))
            max_observed_delta = max(max_observed_delta, delta)
            if delta > max_delta:
                errors.append(
                    f"line {record['_line_no']}: max arm delta {delta:.6f} deg exceeds {max_delta:.6f} deg"
                )
        previous = [float(v) for v in command]

    by_task = Counter(record.get("target_object") or "unknown" for record in records)
    first_target_delta = None
    if start_pose_deg is not None and records:
        first_command = records[0].get("command_deg")
        if isinstance(first_command, list) and len(first_command) == 8:
            first_target_delta = max_abs_arm_delta_deg(first_command, start_pose_deg)
            if first_target_tolerance_deg is not None and first_target_delta > first_target_tolerance_deg:
                errors.append(
                    "first target arm delta from configured start pose "
                    f"{first_target_delta:.6f} deg exceeds {first_target_tolerance_deg:.6f} deg"
                )
    return {
        "ok": not errors,
        "errors": errors,
        "records": len(records),
        "side": side,
        "expected_steps": expected_steps,
        "max_allowed_delta_deg": max_delta,
        "max_observed_arm_delta_deg": round(max_observed_delta, 6),
        "first_target_delta_from_start_deg": None
        if first_target_delta is None
        else round(float(first_target_delta), 6),
        "first_target_tolerance_deg": first_target_tolerance_deg,
        "tasks": dict(sorted(by_task.items())),
        "gripper_sent_to_real": any(record.get("gripper_sent_to_real") is True for record in records),
    }


def write_report(summary: dict[str, Any], path: Path, *, trace_path: Path) -> None:
    lines = [
        "# SmolVLA Dry-Run Mirror Trace Audit",
        "",
        f"- Trace: `{trace_path}`",
        f"- Status: `{'PASS' if summary['ok'] else 'FAIL'}`",
        f"- Records: `{summary['records']}`",
        f"- Expected steps: `{summary['expected_steps']}`",
        f"- Side: `{summary['side']}`",
        f"- Max observed arm delta: `{summary['max_observed_arm_delta_deg']:.6f} deg`",
        f"- Max allowed arm delta: `{summary['max_allowed_delta_deg']:.6f} deg`",
        f"- First target delta from start: `{summary['first_target_delta_from_start_deg']}`",
        f"- First target tolerance: `{summary['first_target_tolerance_deg']}`",
        f"- Gripper sent to real: `{summary['gripper_sent_to_real']}`",
        "",
        "## Task Counts",
        "",
        "| Target | Command Records |",
        "|---|---:|",
    ]
    for target, count in summary["tasks"].items():
        lines.append(f"| {target} | {count} |")
    lines.extend(["", "## Errors", ""])
    if summary["errors"]:
        lines.extend(f"- {error}" for error in summary["errors"])
    else:
        lines.append("- None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    trace_path = _abs(args.trace)
    records = _load_records(trace_path)
    start_pose = start_pose_from_config(args.start_pose_config, args.side) if args.start_pose_config else None
    summary = audit(
        records,
        side=args.side,
        max_delta=args.max_joint_delta_deg,
        expected_steps=args.expected_steps,
        start_pose_deg=start_pose,
        first_target_tolerance_deg=args.first_target_tolerance_deg,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_md:
        write_report(summary, _abs(args.output_md), trace_path=trace_path)
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
