#!/usr/bin/env python3
"""Plot replayed VLA targets against real OpenArm readback states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_contract import JOINT_NAMES, REPO_ROOT  # noqa: E402


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-log", required=True, help="JSONL replay log with command_replayed events")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--title", default="OpenArm Real vs VLA Replay")
    return parser


def _load_replay_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "command_replayed":
                continue
            command = record.get("command_deg")
            state = record.get("real_state_deg")
            if not isinstance(command, list) or len(command) != 8:
                continue
            if not isinstance(state, list) or len(state) != 8:
                continue
            record["_line_no"] = line_no
            record["_command"] = [float(value) for value in command]
            record["_state"] = [float(value) for value in state]
            records.append(record)
    if not records:
        raise SystemExit(
            f"No command_replayed records with real_state_deg found in {path}. "
            "Rerun replay with the updated logger, then plot that new log."
        )
    return records


def _max_abs_error(records: list[dict[str, Any]], *, include_gripper: bool) -> float:
    count = 8 if include_gripper else 7
    return max(
        max(abs(record["_command"][index] - record["_state"][index]) for index in range(count))
        for record in records
    )


def _mean_abs_error(records: list[dict[str, Any]], *, include_gripper: bool) -> float:
    count = 8 if include_gripper else 7
    total = 0.0
    n = 0
    for record in records:
        for index in range(count):
            total += abs(record["_command"][index] - record["_state"][index])
            n += 1
    return total / max(1, n)


def _plot(
    *,
    records: list[dict[str, Any]],
    names: list[str],
    indices: list[int],
    output_png: Path,
    output_pdf: Path,
    title: str,
) -> None:
    x = list(range(len(records)))
    rows = len(names)
    fig, axes = plt.subplots(rows, 1, figsize=(13.5, 2.05 * rows), sharex=True)
    if rows == 1:
        axes = [axes]
    for axis, name, index in zip(axes, names, indices):
        command = [record["_command"][index] for record in records]
        state = [record["_state"][index] for record in records]
        axis.plot(x, command, color="#2563eb", linewidth=1.35, label="VLA target")
        axis.plot(x, state, color="#16a34a", linewidth=1.15, alpha=0.9, label="real readback")
        axis.fill_between(x, command, state, color="#f59e0b", alpha=0.16, linewidth=0)
        axis.grid(True, which="major", linewidth=0.5, alpha=0.35)
        axis.set_ylabel(f"{name}\n(deg)", rotation=0, ha="right", va="center")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes[-1].set_xlabel("Replay command sequence")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> int:
    args = build_arg_parser().parse_args()
    log_path = _abs(args.replay_log)
    output_dir = _abs(args.output_dir)
    prefix = args.prefix or log_path.stem
    records = _load_replay_records(log_path)

    arm_png = output_dir / f"{prefix}_real_vs_vla_right_arm_joints.png"
    arm_pdf = output_dir / f"{prefix}_real_vs_vla_right_arm_joints.pdf"
    all_png = output_dir / f"{prefix}_real_vs_vla_all_8d.png"
    all_pdf = output_dir / f"{prefix}_real_vs_vla_all_8d.pdf"

    title_suffix = (
        f"records={len(records)}, max arm err={_max_abs_error(records, include_gripper=False):.2f} deg, "
        f"mean arm err={_mean_abs_error(records, include_gripper=False):.2f} deg"
    )
    _plot(
        records=records,
        names=list(JOINT_NAMES),
        indices=list(range(7)),
        output_png=arm_png,
        output_pdf=arm_pdf,
        title=f"{args.title} - {title_suffix}",
    )
    _plot(
        records=records,
        names=list(JOINT_NAMES) + ["gripper"],
        indices=list(range(8)),
        output_png=all_png,
        output_pdf=all_pdf,
        title=f"{args.title} - {title_suffix}",
    )

    summary = {
        "replay_log": str(log_path),
        "records": len(records),
        "max_arm_abs_error_deg": round(_max_abs_error(records, include_gripper=False), 6),
        "mean_arm_abs_error_deg": round(_mean_abs_error(records, include_gripper=False), 6),
        "max_8d_abs_error_deg": round(_max_abs_error(records, include_gripper=True), 6),
        "mean_8d_abs_error_deg": round(_mean_abs_error(records, include_gripper=True), 6),
        "plots": {
            "right_arm_joints_png": str(arm_png),
            "right_arm_joints_pdf": str(arm_pdf),
            "right_all_8d_png": str(all_png),
            "right_all_8d_pdf": str(all_pdf),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{prefix}_real_vs_vla_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
