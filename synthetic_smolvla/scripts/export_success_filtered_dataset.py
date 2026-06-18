#!/usr/bin/env python3
"""Filter measured oracle successes and export a local LeRobot dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from dataset_export import export_lerobot_dataset, load_manifest
from sim_contract import CONFIG_DIR, REPO_ROOT, load_yaml_config
from train_smolvla import main as train_preflight_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        default=str(CONFIG_DIR / "dataset_success_filtered.yaml"),
        help="success-filtered dataset YAML",
    )
    parser.add_argument("--source-manifest", default=None, help="all-episodes JSONL manifest to filter")
    parser.add_argument("--success-manifest", default=None, help="success-only JSONL path to write/read")
    parser.add_argument("--dataset-root", default=None, help="LeRobot dataset root to write")
    parser.add_argument("--repo-id", default=None, help="local LeRobot repo id metadata")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--drop-limit-exceeded", action="store_true")
    parser.add_argument("--min-successes", type=int, default=1)
    parser.add_argument("--report", default="synthetic_smolvla/reports/success_filtered_dataset.md")
    parser.add_argument(
        "--prepare-train",
        action="store_true",
        help="also regenerate the success-filtered SmolVLA training preflight/script",
    )
    return parser


def resolve(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def keep_success(record: dict[str, Any], *, drop_limit_exceeded: bool) -> bool:
    if not bool(record.get("success_label")):
        return False
    if bool(record.get("wrong_object_lifted")):
        return False
    if drop_limit_exceeded and bool(record.get("limit_exceeded")):
        return False
    if not record.get("steps"):
        return False
    return True


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_report(
    *,
    path: Path,
    source_manifest: Path | None,
    success_manifest: Path,
    dataset_metadata: dict[str, Any],
    all_records: list[dict[str, Any]] | None,
    kept_records: list[dict[str, Any]],
    drop_limit_exceeded: bool,
) -> None:
    total = len(all_records) if all_records is not None else len(kept_records)
    kept = len(kept_records)
    by_target = Counter(str(record.get("target_object", "unknown")) for record in kept_records)
    all_by_target = Counter(str(record.get("target_object", "unknown")) for record in all_records or kept_records)
    wrong = sum(1 for record in (all_records or kept_records) if record.get("wrong_object_lifted"))
    limit = sum(1 for record in (all_records or kept_records) if record.get("limit_exceeded"))
    rate = 0.0 if total == 0 else kept / total

    lines = [
        "# Success-Filtered SmolVLA Dataset",
        "",
        "This dataset keeps only measured successful target-object lifts from the parallel IK oracle.",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Source episodes | {total} | 1.000 |",
        f"| Kept successes | {kept} | {rate:.3f} |",
        f"| Source wrong-object lifts | {wrong} | {0.0 if total == 0 else wrong / total:.3f} |",
        f"| Source limit-clamp flags | {limit} | {0.0 if total == 0 else limit / total:.3f} |",
        "",
        "## Kept Successes By Target",
        "",
        "| Target | Kept | Source Episodes |",
        "|---|---:|---:|",
    ]
    for target in sorted(all_by_target):
        lines.append(f"| {target} | {by_target[target]} | {all_by_target[target]} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Source manifest: `{source_manifest}`" if source_manifest else "- Source manifest: not provided",
            f"- Success manifest: `{success_manifest}`",
            f"- LeRobot root: `{dataset_metadata['root']}`",
            f"- LeRobot repo id: `{dataset_metadata['repo_id']}`",
            "",
            "## Notes",
            "",
            f"- Drop limit-exceeded records: `{drop_limit_exceeded}`.",
            "- RGB frames currently use the deterministic top-down renderer in `dataset_export.py`.",
            "- The parallel oracle measures physics success; this export does not keep failed attempts.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_train_preflight() -> None:
    import sys

    previous = sys.argv[:]
    sys.argv = [
        "train_smolvla.py",
        "--train-config",
        str(CONFIG_DIR / "train_success_filtered.yaml"),
        "--output",
        "synthetic_smolvla/reports/train_success_filtered_preflight.json",
        "--command-output",
        "synthetic_smolvla/reports/train_success_filtered.sh",
        "--overwrite-output-dir",
    ]
    try:
        train_preflight_main()
    finally:
        sys.argv = previous


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.dataset_config)
    dataset = config["dataset"]
    filter_cfg = dataset.get("filter", {})

    source_manifest = args.source_manifest or dataset.get("source_manifest")
    success_manifest = resolve(args.success_manifest or dataset["manifest"])
    dataset_root = resolve(args.dataset_root or dataset["output_dir"])
    repo_id = args.repo_id or dataset.get("repo_id") or f"local/{dataset['name']}"
    image_size = int(args.image_size or dataset.get("image_size", 96))
    fps = int(args.fps or dataset.get("fps", 10))
    drop_limit = bool(args.drop_limit_exceeded or filter_cfg.get("reject_limit_exceeded", False))

    all_records = None
    if source_manifest:
        source_path = resolve(source_manifest)
        all_records = load_manifest(source_path)
        kept_records = [record for record in all_records if keep_success(record, drop_limit_exceeded=drop_limit)]
        write_jsonl(kept_records, success_manifest)
    else:
        source_path = None
        kept_records = load_manifest(success_manifest)
        kept_records = [record for record in kept_records if keep_success(record, drop_limit_exceeded=drop_limit)]
        write_jsonl(kept_records, success_manifest)

    if len(kept_records) < args.min_successes:
        raise SystemExit(
            f"Only {len(kept_records)} successful episodes after filtering; "
            f"need at least {args.min_successes}."
        )

    metadata = export_lerobot_dataset(
        manifest_path=success_manifest,
        output_dir=dataset_root,
        repo_id=repo_id,
        fps=fps,
        image_size=image_size,
        overwrite=args.overwrite,
    )
    report = resolve(args.report)
    write_report(
        path=report,
        source_manifest=source_path,
        success_manifest=success_manifest,
        dataset_metadata=metadata,
        all_records=all_records,
        kept_records=kept_records,
        drop_limit_exceeded=drop_limit,
    )
    if args.prepare_train:
        run_train_preflight()

    print(
        json.dumps(
            {
                "ok": True,
                "source_episodes": len(all_records) if all_records is not None else len(kept_records),
                "kept_successes": len(kept_records),
                "success_manifest": str(success_manifest),
                "dataset": metadata,
                "report": str(report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
