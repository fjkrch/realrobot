#!/usr/bin/env python3
"""Create a filtered prefix-only LeRobot dataset from an existing local dataset.

This is useful when a policy fails near reset/approach: it reuses already
validated clean episodes and copies only the first N frames for selected tasks.
No simulation or real robot hardware is touched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_dense_isaac_dataset import ACTION_KEY, CAMERA_KEY, STATE_KEY, STATE_NAMES  # noqa: E402
from sim_contract import REPO_ROOT  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-repo-id", default=None, help="defaults to local/<input-root-name>")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--task-filter",
        default="pick up the orange ball,pick up the red cube",
        help="comma-separated task strings to include; empty means all tasks",
    )
    parser.add_argument("--prefix-frames", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1, help="repeat each copied prefix as a separate episode")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default="synthetic_smolvla/reports/prefix_lerobot_dataset.json")
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


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.prefix_frames <= 0:
        raise SystemExit("--prefix-frames must be positive")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    input_root = _resolve(args.input_root)
    output_root = _resolve(args.output_root)
    report_path = _resolve(args.report)
    task_filter = {task.strip() for task in args.task_filter.split(",") if task.strip()}

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing output root: {output_root}")
        shutil.rmtree(output_root)

    features = {
        CAMERA_KEY: {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        STATE_KEY: {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
    }
    source = LeRobotDataset(_repo_id_for(input_root, args.input_repo_id), root=input_root)
    output = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=args.fps,
        robot_type="openarm_synthetic_isaac_dense_prefix",
        features=features,
        use_videos=False,
        image_writer_threads=0,
        image_writer_processes=0,
    )

    copied_source_episodes = 0
    output_episodes = 0
    output_frames = 0
    task_counts: Counter[str] = Counter()
    selected_episode_indices: list[int] = []

    for episode in source.meta.episodes:
        tasks = [str(task) for task in episode["tasks"]]
        task = tasks[0] if tasks else ""
        if task_filter and task not in task_filter:
            continue
        copied_source_episodes += 1
        selected_episode_indices.append(int(episode["episode_index"]))
        start = int(episode["dataset_from_index"])
        stop = min(int(episode["dataset_to_index"]), start + int(args.prefix_frames))
        for _repeat_index in range(args.repeat):
            for index in range(start, stop):
                item = source[index]
                output.add_frame(
                    {
                        CAMERA_KEY: _image_hwc_uint8(item[CAMERA_KEY]),
                        STATE_KEY: _float32(item[STATE_KEY]),
                        ACTION_KEY: _float32(item[ACTION_KEY]),
                        "task": str(item["task"]),
                    }
                )
                task_counts[str(item["task"])] += 1
                output_frames += 1
            output.save_episode()
            output_episodes += 1

    output.finalize()

    report = {
        "input_root": str(input_root),
        "input_repo_id": _repo_id_for(input_root, args.input_repo_id),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "task_filter": sorted(task_filter),
        "prefix_frames": args.prefix_frames,
        "repeat": args.repeat,
        "copied_source_episodes": copied_source_episodes,
        "output_episodes": output_episodes,
        "output_frames": output_frames,
        "task_frame_counts": dict(sorted(task_counts.items())),
        "selected_episode_indices_head": selected_episode_indices[:50],
        "selected_episode_indices_tail": selected_episode_indices[-50:],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(report_path), **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
