#!/usr/bin/env python3
"""Generate JSONL oracle demonstration manifests for the synthetic pick task."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from dataset_export import export_lerobot_dataset
from language import instruction_for_object
from oracle_policy import generate_episode
from sim_contract import CONFIG_DIR, load_yaml_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        default=str(CONFIG_DIR / "dataset_v1.yaml"),
        help="dataset YAML with episode counts and output path",
    )
    parser.add_argument("--episodes", type=int, default=None, help="override number of episodes to generate")
    parser.add_argument("--output", default=None, help="override output JSONL manifest path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomized", action="store_true", help="force randomized object poses")
    parser.add_argument("--all-objects-visible-ratio", type=float, default=None)
    parser.add_argument("--export-lerobot", action="store_true", help="also write a local LeRobot dataset")
    parser.add_argument("--lerobot-root", default=None, help="override local LeRobot dataset root")
    parser.add_argument("--repo-id", default=None, help="LeRobot dataset repo_id metadata")
    parser.add_argument("--image-size", type=int, default=96, help="synthetic RGB image size for LeRobot export")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing LeRobot export directory")
    return parser


def object_schedule(config: dict, total: int) -> list[str]:
    counts = config["dataset"].get("episodes_per_object", {})
    if not counts:
        names = ("orange_ball", "red_cube", "green_cube", "blue_cube")
        return [names[index % len(names)] for index in range(total)]

    names = list(counts)
    schedule: list[str] = []
    index = 0
    while len(schedule) < total:
        name = names[index % len(names)]
        schedule.append(name)
        index += 1
    return schedule[:total]


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.dataset_config)
    dataset = config["dataset"]
    total = int(args.episodes or dataset["total_episodes"])
    if total <= 0:
        raise SystemExit("--episodes must be positive.")

    output = Path(args.output or dataset["manifest"])
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else dataset.get("seed", 0))
    rng = random.Random(seed)
    randomized = bool(args.randomized or dataset.get("randomized", False))
    visible_ratio = float(
        args.all_objects_visible_ratio
        if args.all_objects_visible_ratio is not None
        else dataset.get("all_objects_visible_ratio", 1.0)
    )
    visible_ratio = max(0.0, min(1.0, visible_ratio))

    schedule = object_schedule(config, total)
    counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for episode_index, object_name in enumerate(schedule):
            instruction = instruction_for_object(object_name)
            all_objects_visible = rng.random() <= visible_ratio
            episode = generate_episode(
                episode_index,
                instruction,
                seed=seed,
                randomized=randomized,
                all_objects_visible=all_objects_visible,
            )
            handle.write(json.dumps(episode, sort_keys=True) + "\n")
            counts[object_name] += 1

    summary = {
        "ok": True,
        "manifest": str(output),
        "episodes": total,
        "counts": dict(sorted(counts.items())),
        "randomized": randomized,
        "all_objects_visible_ratio": visible_ratio,
    }
    if args.export_lerobot:
        output_dir = Path(args.lerobot_root or dataset["output_dir"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        repo_id = args.repo_id or f"local/{dataset['name']}"
        metadata = export_lerobot_dataset(
            manifest_path=output,
            output_dir=output_dir,
            repo_id=repo_id,
            image_size=args.image_size,
            overwrite=args.overwrite,
        )
        summary["lerobot_dataset"] = metadata
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
