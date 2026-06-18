#!/usr/bin/env python3
"""Plot raw VLA joint targets against a resampled OpenArm trajectory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mirror_sinks import start_pose_from_config  # noqa: E402
from sim_contract import JOINT_NAMES, REPO_ROOT, load_yaml_config, normalize_side, validate_scene_config  # noqa: E402


DEFAULT_CONFIG = "synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml"


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="raw VLA *_commands.jsonl")
    parser.add_argument("--resampled", required=True, help="resampled/split trajectory JSONL")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--side", choices=["left", "right"], default=None, help="defaults to scene active arm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--title", default="OpenArm VLA Resampling")
    return parser


def _load_commands(path: Path) -> list[dict[str, Any]]:
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
            record["_command"] = [float(value) for value in command]
            records.append(record)
    if not records:
        raise SystemExit(f"No command_deg records found in {path}")
    return records


def _resampled_x(records: list[dict[str, Any]], raw_count: int) -> list[float]:
    if all("source_sequence" in record for record in records):
        group_counts = Counter(int(record["source_sequence"]) for record in records)
        seen: dict[int, int] = defaultdict(int)
        out = []
        for record in records:
            source = int(record["source_sequence"])
            seen[source] += 1
            out.append(source - 1.0 + seen[source] / group_counts[source])
        return out
    if len(records) == 1:
        return [0.0]
    scale = max(1, raw_count - 1) / max(1, len(records) - 1)
    return [index * scale for index in range(len(records))]


def _max_delta(records: list[dict[str, Any]], *, include_gripper: bool) -> float:
    count = 8 if include_gripper else 7
    max_delta = 0.0
    for previous, current in zip(records, records[1:]):
        max_delta = max(
            max_delta,
            max(abs(current["_command"][index] - previous["_command"][index]) for index in range(count)),
        )
    return max_delta


def _plot(
    *,
    raw_records: list[dict[str, Any]],
    resampled_records: list[dict[str, Any]],
    start_pose: list[float],
    resampled_x: list[float],
    joints: list[str],
    joint_indices: list[int],
    title: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    raw_x = list(range(len(raw_records)))
    rows = len(joints)
    fig, axes = plt.subplots(rows, 1, figsize=(13.5, 2.05 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    for axis, joint, index in zip(axes, joints, joint_indices):
        raw_y = [record["_command"][index] for record in raw_records]
        resampled_y = [record["_command"][index] for record in resampled_records]
        axis.plot(resampled_x, resampled_y, color="#2563eb", linewidth=1.35, label="fixed/resampled")
        axis.scatter(raw_x, raw_y, color="#dc2626", s=16, zorder=3, label="raw VLA")
        axis.scatter([-1.0], [start_pose[index]], color="#111827", marker="D", s=24, zorder=4, label="sim start")
        axis.axvline(-1.0, color="#9ca3af", linewidth=0.8, linestyle=":")
        axis.grid(True, which="major", linewidth=0.5, alpha=0.35)
        axis.set_ylabel(f"{joint}\n(deg)", rotation=0, ha="right", va="center")

    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes[-1].set_xlabel("VLA command index (sim start at -1)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    validate_scene_config(config)
    side = normalize_side(args.side or str(config["scene"].get("active_arm", "right")))

    raw_path = _abs(args.raw)
    resampled_path = _abs(args.resampled)
    output_dir = _abs(args.output_dir)
    prefix = args.prefix or resampled_path.stem

    raw_records = _load_commands(raw_path)
    resampled_records = _load_commands(resampled_path)
    start_pose = start_pose_from_config(args.config, side)
    x_resampled = _resampled_x(resampled_records, len(raw_records))

    raw_count = len(raw_records)
    resampled_count = len(resampled_records)
    title_suffix = (
        f"{side} arm | raw={raw_count}, resampled={resampled_count}, "
        f"avg={resampled_count / raw_count:.2f} samples/VLA cmd"
    )

    arm_names = list(JOINT_NAMES)
    arm_indices = list(range(7))
    all_names = list(JOINT_NAMES) + ["gripper"]
    all_indices = list(range(8))

    arm_png = output_dir / f"{prefix}_right_arm_joints.png"
    arm_pdf = output_dir / f"{prefix}_right_arm_joints.pdf"
    all_png = output_dir / f"{prefix}_right_all_8d.png"
    all_pdf = output_dir / f"{prefix}_right_all_8d.pdf"

    _plot(
        raw_records=raw_records,
        resampled_records=resampled_records,
        start_pose=start_pose,
        resampled_x=x_resampled,
        joints=arm_names,
        joint_indices=arm_indices,
        title=f"{args.title} - {title_suffix}",
        output_png=arm_png,
        output_pdf=arm_pdf,
    )
    _plot(
        raw_records=raw_records,
        resampled_records=resampled_records,
        start_pose=start_pose,
        resampled_x=x_resampled,
        joints=all_names,
        joint_indices=all_indices,
        title=f"{args.title} - {title_suffix}",
        output_png=all_png,
        output_pdf=all_pdf,
    )

    summary = {
        "raw": str(raw_path),
        "resampled": str(resampled_path),
        "side": side,
        "raw_records": raw_count,
        "resampled_records": resampled_count,
        "average_samples_per_vla_command": round(resampled_count / raw_count, 6),
        "max_raw_arm_delta_deg": round(_max_delta(raw_records, include_gripper=False), 6),
        "max_resampled_arm_delta_deg": round(_max_delta(resampled_records, include_gripper=False), 6),
        "max_raw_8d_delta_deg": round(_max_delta(raw_records, include_gripper=True), 6),
        "max_resampled_8d_delta_deg": round(_max_delta(resampled_records, include_gripper=True), 6),
        "plots": {
            "right_arm_joints_png": str(arm_png),
            "right_arm_joints_pdf": str(arm_pdf),
            "right_all_8d_png": str(all_png),
            "right_all_8d_pdf": str(all_pdf),
        },
    }
    summary_path = output_dir / f"{prefix}_resampling_plot_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
