#!/usr/bin/env python3
"""Compare a SmolVLA checkpoint against clean teacher actions offline.

This is a simulation-only diagnostic. It does not launch Isaac and never touches
real robot hardware. The goal is to check whether a trained policy can reproduce
the dense teacher commands on the exact dataset frames used for training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_contract import JOINT_NAMES, REPO_ROOT, SAFE_ARM_LIMITS_DEG  # noqa: E402


CAMERA_KEY = "observation.images.camera1"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_NAMES = [*JOINT_NAMES, "gripper"]


DEFAULT_DATASET = "synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right"
DEFAULT_CKPT = (
    "synthetic_smolvla/checkpoints/"
    "smolvla_openarm_real_table_zero_lift5cm_routed_v2_from020_lr3e5/"
    "checkpoints/010000/pretrained_model_typed"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default=None, help="defaults to local/<dataset-root-name>")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument(
        "--task-filter",
        default="pick up the orange ball,pick up the red cube",
        help="comma-separated task strings to include; empty means all tasks",
    )
    parser.add_argument("--episodes-per-task", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9321)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--max-gripper-close-deg",
        type=float,
        default=-3.0,
        help="gripper upper bound from the user constraint; predictions above this are too closed",
    )
    parser.add_argument("--output-json", default="synthetic_smolvla/reports/policy_teacher_action_diagnostic.json")
    parser.add_argument("--output-md", default="synthetic_smolvla/reports/policy_teacher_action_diagnostic.md")
    return parser


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _repo_id_for(root: Path, explicit: str | None) -> str:
    return explicit or f"local/{root.name}"


def _image_hwc_uint8(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return array.astype(np.uint8)


def _float32(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return array.astype(np.float32)


def _task_target(task: str) -> str:
    text = task.lower()
    if "orange" in text:
        return "orange_ball"
    if "red" in text:
        return "red_cube"
    if "green" in text:
        return "green_cube"
    if "blue" in text:
        return "blue_cube"
    return text.replace(" ", "_")


def _phase(frame_index: int, episode_len: int) -> str:
    # Collection phases are proportional to 14/12/8/12/4 over a 50-step base.
    cuts = np.cumsum(np.asarray([14, 12, 8, 12, 4], dtype=np.float32) / 50.0 * float(episode_len))
    labels = ("approach", "descend", "close", "lift", "hold")
    for label, cut in zip(labels, cuts):
        if frame_index < int(round(float(cut))):
            return label
    return "hold"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), pct))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _row_to_obs(row: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, str]:
    obs = {
        CAMERA_KEY: _image_hwc_uint8(row[CAMERA_KEY]),
        STATE_KEY: _float32(row[STATE_KEY]),
    }
    teacher = _float32(row[ACTION_KEY])
    task = str(row["task"])
    return obs, obs[STATE_KEY], teacher, task


def _predict(
    row: dict[str, Any],
    *,
    policy: Any,
    pre: Any,
    post: Any,
    predict_action: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    obs, state, teacher, task = _row_to_obs(row)
    action = predict_action(obs, policy, device, pre, post, use_amp=False, task=task)
    pred = action.detach().cpu().numpy().astype(np.float32).reshape(-1, len(ACTION_NAMES))[0]
    return pred, teacher, state, task


def _limit_events(pred: np.ndarray, *, side: str, max_gripper_close_deg: float) -> dict[str, int]:
    arm_limits = SAFE_ARM_LIMITS_DEG[side]
    arm_violations = 0
    for index, joint in enumerate(JOINT_NAMES):
        low, high = arm_limits[joint]
        if float(pred[index]) < low or float(pred[index]) > high:
            arm_violations += 1
    # In this robot convention, 0 deg is closed and -65 deg is open, so a value
    # greater than -3 deg is too closed for the user's current constraint.
    gripper_too_closed = int(float(pred[-1]) > float(max_gripper_close_deg))
    return {
        "arm_limit_violations": int(arm_violations),
        "gripper_too_closed": int(gripper_too_closed),
    }


def _make_record(
    *,
    mode: str,
    row: dict[str, Any],
    episode_len: int,
    pred: np.ndarray,
    teacher: np.ndarray,
    state: np.ndarray,
    side: str,
    max_gripper_close_deg: float,
) -> dict[str, Any]:
    abs_err = np.abs(pred - teacher)
    target = _task_target(str(row["task"]))
    frame_index = int(row["frame_index"])
    record = {
        "mode": mode,
        "target": target,
        "task": str(row["task"]),
        "episode_index": int(row["episode_index"]),
        "frame_index": frame_index,
        "phase": _phase(frame_index, episode_len),
        "mae_all_deg": round(float(abs_err.mean()), 6),
        "mae_arm_deg": round(float(abs_err[:7].mean()), 6),
        "gripper_abs_deg": round(float(abs_err[7]), 6),
        "max_abs_deg": round(float(abs_err.max()), 6),
        "per_joint_abs_deg": {name: round(float(value), 6) for name, value in zip(ACTION_NAMES, abs_err)},
        "teacher_action_deg": [round(float(v), 6) for v in teacher.tolist()],
        "pred_action_deg": [round(float(v), 6) for v in pred.tolist()],
        "state_deg": [round(float(v), 6) for v in state.tolist()],
    }
    record.update(_limit_events(pred, side=side, max_gripper_close_deg=max_gripper_close_deg))
    return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped["all"].append(record)
        grouped[f"mode={record['mode']}"].append(record)
        grouped[f"target={record['target']}"].append(record)
        grouped[f"mode={record['mode']} target={record['target']}"].append(record)
        grouped[f"mode={record['mode']} phase={record['phase']}"].append(record)
        grouped[f"mode={record['mode']} target={record['target']} phase={record['phase']}"].append(record)

    summary: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(grouped.items()):
        mae = [float(row["mae_all_deg"]) for row in rows]
        arm = [float(row["mae_arm_deg"]) for row in rows]
        grip = [float(row["gripper_abs_deg"]) for row in rows]
        max_abs = [float(row["max_abs_deg"]) for row in rows]
        summary[key] = {
            "count": len(rows),
            "mean_mae_all_deg": round(_mean(mae), 4),
            "p90_mae_all_deg": round(_percentile(mae, 90), 4),
            "mean_mae_arm_deg": round(_mean(arm), 4),
            "mean_gripper_abs_deg": round(_mean(grip), 4),
            "p90_max_abs_deg": round(_percentile(max_abs, 90), 4),
            "arm_limit_violation_records": sum(1 for row in rows if row["arm_limit_violations"]),
            "gripper_too_closed_records": sum(1 for row in rows if row["gripper_too_closed"]),
        }
    return summary


def _write_report(
    *,
    path: Path,
    dataset_root: Path,
    checkpoint: Path,
    selected_episodes: dict[str, list[int]],
    records: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    top_errors = sorted(records, key=lambda item: float(item["mae_all_deg"]), reverse=True)[:12]
    lines = [
        "# Policy vs Teacher Action Diagnostic",
        "",
        "Simulation only. This report uses stored clean dataset frames and does not run Isaac or real robot hardware.",
        "",
        f"- Dataset: `{dataset_root}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Selected episodes: `{json.dumps(selected_episodes, sort_keys=True)}`",
        f"- Policy `n_action_steps`: `{config.get('n_action_steps')}`",
        f"- Policy input features: `{json.dumps(config.get('input_features', {}), sort_keys=True)}`",
        f"- Policy output features: `{json.dumps(config.get('output_features', {}), sort_keys=True)}`",
        "",
        "## Summary",
        "",
        "| Group | Count | Mean MAE | P90 MAE | Mean Arm MAE | Mean Grip Abs | P90 Max Abs | Arm Limit Rows | Grip Too Closed Rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary.items():
        if "phase=" in key or key.startswith("target="):
            continue
        lines.append(
            f"| {key} | {row['count']} | {row['mean_mae_all_deg']:.4f} | "
            f"{row['p90_mae_all_deg']:.4f} | {row['mean_mae_arm_deg']:.4f} | "
            f"{row['mean_gripper_abs_deg']:.4f} | {row['p90_max_abs_deg']:.4f} | "
            f"{row['arm_limit_violation_records']} | {row['gripper_too_closed_records']} |"
        )
    lines.extend([
        "",
        "## Phase Breakdown",
        "",
        "| Group | Count | Mean MAE | P90 MAE | Mean Arm MAE | Mean Grip Abs | P90 Max Abs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, row in summary.items():
        if "phase=" not in key:
            continue
        lines.append(
            f"| {key} | {row['count']} | {row['mean_mae_all_deg']:.4f} | "
            f"{row['p90_mae_all_deg']:.4f} | {row['mean_mae_arm_deg']:.4f} | "
            f"{row['mean_gripper_abs_deg']:.4f} | {row['p90_max_abs_deg']:.4f} |"
        )
    lines.extend([
        "",
        "## Largest Errors",
        "",
        "| Mode | Target | Episode | Frame | Phase | MAE | Max Abs | Joint With Max Error | Teacher | Predicted |",
        "|---|---|---:|---:|---|---:|---:|---|---|---|",
    ])
    for row in top_errors:
        per_joint = row["per_joint_abs_deg"]
        joint = max(per_joint, key=per_joint.get)
        lines.append(
            f"| {row['mode']} | {row['target']} | {row['episode_index']} | "
            f"{row['frame_index']} | {row['phase']} | {row['mae_all_deg']:.4f} | "
            f"{row['max_abs_deg']:.4f} | {joint} | "
            f"`{row['teacher_action_deg']}` | `{row['pred_action_deg']}` |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    dataset_root = _resolve(args.dataset_root)
    checkpoint = _resolve(args.checkpoint)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)

    import torch  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415
    from lerobot.utils.control_utils import predict_action  # noqa: PLC0415

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = LeRobotDataset(_repo_id_for(dataset_root, args.repo_id), root=dataset_root)

    tasks = [task.strip() for task in args.task_filter.split(",") if task.strip()]
    task_set = set(tasks)
    episodes_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in ds.meta.episodes:
        episode_tasks = [str(task) for task in episode["tasks"]]
        task = episode_tasks[0] if episode_tasks else ""
        if task_set and task not in task_set:
            continue
        episodes_by_task[task].append(episode)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    selected_episodes: dict[str, list[int]] = {}
    for task, episodes in sorted(episodes_by_task.items()):
        picks = list(episodes)
        rng.shuffle(picks)
        picks = sorted(picks[: args.episodes_per_task], key=lambda ep: int(ep["episode_index"]))
        selected.extend(picks)
        selected_episodes[task] = [int(ep["episode_index"]) for ep in picks]

    if not selected:
        raise SystemExit("No episodes selected; check --task-filter.")

    policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
    policy.to(device)
    policy.eval()
    policy.config.pretrained_path = str(checkpoint)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    records: list[dict[str, Any]] = []
    total_episode_count = len(selected)
    for episode_pos, episode in enumerate(selected, start=1):
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        episode_len = int(episode["length"])
        frame_indexes = list(range(start, stop, max(1, args.frame_stride)))
        if stop - 1 not in frame_indexes:
            frame_indexes.append(stop - 1)

        print(
            f"[diagnose] episode {episode_pos}/{total_episode_count} "
            f"ep={int(episode['episode_index'])} task={episode['tasks'][0]} "
            f"frames={len(frame_indexes)}",
            file=sys.stderr,
            flush=True,
        )

        for index in frame_indexes:
            row = ds[index]
            policy.reset()
            pre.reset()
            post.reset()
            pred, teacher, state, _task = _predict(
                row,
                policy=policy,
                pre=pre,
                post=post,
                predict_action=predict_action,
                device=device,
            )
            records.append(
                _make_record(
                    mode="independent",
                    row=row,
                    episode_len=episode_len,
                    pred=pred,
                    teacher=teacher,
                    state=state,
                    side=args.side,
                    max_gripper_close_deg=args.max_gripper_close_deg,
                )
            )

        policy.reset()
        pre.reset()
        post.reset()
        for index in range(start, stop):
            row = ds[index]
            pred, teacher, state, _task = _predict(
                row,
                policy=policy,
                pre=pre,
                post=post,
                predict_action=predict_action,
                device=device,
            )
            records.append(
                _make_record(
                    mode="sequential_teacher_forced",
                    row=row,
                    episode_len=episode_len,
                    pred=pred,
                    teacher=teacher,
                    state=state,
                    side=args.side,
                    max_gripper_close_deg=args.max_gripper_close_deg,
                )
            )

    summary = _summarize(records)
    payload = {
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "selected_episodes": selected_episodes,
        "summary": summary,
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    _write_report(
        path=output_md,
        dataset_root=dataset_root,
        checkpoint=checkpoint,
        selected_episodes=selected_episodes,
        records=records,
        summary=summary,
        config=cfg,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "records": len(records),
                "summary": summary.get("all", {}),
                "json": str(output_json),
                "report": str(output_md),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
