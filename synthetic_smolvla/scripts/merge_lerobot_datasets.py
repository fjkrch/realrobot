#!/usr/bin/env python3
"""Merge local LeRobot OpenArm datasets into one training root.

This keeps the existing dense dataset schema and rewrites episodes through the
LeRobotDataset writer instead of editing parquet files directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_dense_isaac_dataset import ACTION_KEY, CAMERA_KEY, STATE_KEY, STATE_NAMES  # noqa: E402
from sim_contract import REPO_ROOT  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="ROOT",
        help="input dataset root; repeat for multiple datasets",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-writer-threads", type=int, default=0)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument(
        "--max-total-episodes",
        type=int,
        default=None,
        help="optional cap on complete episodes written across all inputs",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default="synthetic_smolvla/reports/merged_lerobot_dataset.json")
    return parser


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _repo_id_for(root: Path) -> str:
    return f"local/{root.name}"


def _image_hwc_uint8(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return array.astype(np.uint8)


def _float32(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return array.astype(np.float32)


def main() -> int:
    args = build_arg_parser().parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    output_root = _resolve(args.output_root)
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing output root: {output_root}")
        shutil.rmtree(output_root)

    features = {
        CAMERA_KEY: {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        STATE_KEY: {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
    }
    merged = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=args.fps,
        robot_type="openarm_synthetic_isaac_dense",
        features=features,
        use_videos=False,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    counts: list[dict[str, object]] = []
    total_episodes = 0
    total_frames = 0
    task_counts: dict[str, int] = {}
    stop_requested = False

    for input_root_text in args.input:
        if stop_requested:
            break
        input_root = _resolve(input_root_text)
        source = LeRobotDataset(_repo_id_for(input_root), root=input_root)
        current_episode = None
        source_episodes = 0
        source_frames = 0
        for index in range(len(source)):
            item = source[index]
            episode_index = int(item["episode_index"])
            if current_episode is None:
                current_episode = episode_index
            elif episode_index != current_episode:
                merged.save_episode()
                source_episodes += 1
                total_episodes += 1
                if args.max_total_episodes is not None and total_episodes >= args.max_total_episodes:
                    current_episode = None
                    stop_requested = True
                    break
                current_episode = episode_index
            task = str(item["task"])
            task_counts[task] = task_counts.get(task, 0) + 1
            merged.add_frame(
                {
                    CAMERA_KEY: _image_hwc_uint8(item[CAMERA_KEY]),
                    STATE_KEY: _float32(item[STATE_KEY]),
                    ACTION_KEY: _float32(item[ACTION_KEY]),
                    "task": task,
                }
            )
            source_frames += 1
            total_frames += 1
        if current_episode is not None and not stop_requested:
            merged.save_episode()
            source_episodes += 1
            total_episodes += 1
            if args.max_total_episodes is not None and total_episodes >= args.max_total_episodes:
                stop_requested = True
        counts.append(
            {
                "root": str(input_root),
                "frames": source_frames,
                "episodes": source_episodes,
                "repo_id": _repo_id_for(input_root),
            }
        )

    merged.finalize()

    report = {
        "ok": True,
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "inputs": counts,
        "max_total_episodes": args.max_total_episodes,
        "stopped_at_episode_cap": stop_requested,
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "task_frame_counts": task_counts,
    }
    report_path = _resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
