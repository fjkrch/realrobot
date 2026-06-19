#!/usr/bin/env python3
"""Fast local LeRobot merge for byte-embedded image datasets.

This is intentionally narrower than merge_lerobot_datasets.py: it keeps all
episodes from the first input, appends a capped number of episodes from the
second input, and rewrites only parquet metadata/indices. It assumes images are
already embedded as bytes in parquet rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_contract import REPO_ROOT  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--append-root", required=True)
    parser.add_argument("--append-episodes", type=int, required=True)
    parser.add_argument(
        "--append-balance-by-task",
        action="store_true",
        help="select append episodes evenly across task labels instead of taking the first N",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default="synthetic_smolvla/reports/merged_lerobot_dataset_parquet.json")
    return parser


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _stats_from_episode_row(row: dict, features: dict) -> dict:
    stats: dict[str, dict] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        stat_key = key.removeprefix("stats/")
        feature_name, stat_name = stat_key.rsplit("/", 1)
        stats.setdefault(feature_name, {})
        if feature_name in features:
            dtype = features[feature_name]["dtype"]
            if dtype == "image" and stat_name != "count":
                array = np.asarray(value, dtype=object)
                flat_values = []
                for item in array:
                    while isinstance(item, np.ndarray):
                        item = item.flatten()[0]
                    flat_values.append(item)
                value = np.asarray(flat_values, dtype=np.float64).reshape(3, 1, 1)
            else:
                value = np.asarray(value)
        stats[feature_name][stat_name] = value
    return stats


def _add_to_stat_cell(value, offset: int):
    if isinstance(value, np.ndarray):
        return (value + offset).tolist()
    if isinstance(value, list):
        return (np.asarray(value) + offset).tolist()
    return value + offset


def _adjust_append_episode_rows(df: pd.DataFrame, *, episode_offset: int, frame_offset: int) -> pd.DataFrame:
    df = df.copy()
    original_episode_index = df["episode_index"].astype(int)
    new_episode_index = original_episode_index + episode_offset
    length = df["length"].astype(int)
    old_from = df["dataset_from_index"].astype(int)
    df["episode_index"] = new_episode_index
    df["dataset_from_index"] = old_from + frame_offset
    df["dataset_to_index"] = df["dataset_from_index"] + length
    df["data/chunk_index"] = 0
    df["meta/episodes/chunk_index"] = 0
    df["meta/episodes/file_index"] = 0

    # One appended data parquet file; set by caller after base file count is known.
    for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        ep_col = f"stats/episode_index/{stat}"
        if ep_col in df:
            df[ep_col] = [_add_to_stat_cell(v, episode_offset) for v in df[ep_col]]
        idx_col = f"stats/index/{stat}"
        if idx_col in df:
            df[idx_col] = [_add_to_stat_cell(v, frame_offset) for v in df[idx_col]]
    return df


def _write_table_with_metadata(table: pa.Table, path: Path, metadata: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata:
        table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="snappy", use_dictionary=True)


def _append_data_table(
    append_root: Path,
    selected_episode_ids: set[int],
    *,
    episode_mapping: dict[int, int],
    frame_offset: int,
    episode_length: int,
) -> pa.Table:
    tables = []
    for path in sorted((append_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        mask = pc.is_in(table["episode_index"], value_set=pa.array(sorted(selected_episode_ids), type=pa.int64()))
        filtered = table.filter(mask)
        if filtered.num_rows:
            tables.append(filtered)
    if not tables:
        raise ValueError("no append rows selected")
    table = pa.concat_tables(tables, promote_options="default")
    first_new_episode = min(episode_mapping.values())
    old_episode_values = [int(v.as_py()) for v in table["episode_index"]]
    frame_index_values = [int(v.as_py()) for v in table["frame_index"]]
    new_episode_values = [episode_mapping[old_ep] for old_ep in old_episode_values]
    new_episode = pa.array(new_episode_values, type=pa.int64())
    new_index = pa.array(
        [
            frame_offset + (new_ep - first_new_episode) * episode_length + frame_idx
            for new_ep, frame_idx in zip(new_episode_values, frame_index_values, strict=False)
        ],
        type=pa.int64(),
    )
    table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", new_episode)
    table = table.set_column(table.schema.get_field_index("index"), "index", new_index)
    return table


def main() -> int:
    args = build_arg_parser().parse_args()
    base_root = _resolve(args.base_root)
    append_root = _resolve(args.append_root)
    output_root = _resolve(args.output_root)

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing output root: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True)
    (output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    base_info = json.loads((base_root / "meta" / "info.json").read_text())
    base_episodes = pd.read_parquet(base_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    append_episodes_all = pd.read_parquet(append_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    if args.append_balance_by_task:
        selected_parts = []
        task_groups = list(append_episodes_all.groupby(append_episodes_all["tasks"].map(lambda x: tuple(x))))
        base_count = args.append_episodes // len(task_groups)
        remainder = args.append_episodes % len(task_groups)
        for group_index, (_, group) in enumerate(task_groups):
            take = base_count + (1 if group_index < remainder else 0)
            selected_parts.append(group.sort_values("episode_index").head(take))
        append_episodes = pd.concat(selected_parts).sort_values("episode_index").reset_index(drop=True)
    else:
        append_episodes = append_episodes_all[append_episodes_all["episode_index"] < args.append_episodes].reset_index(drop=True)
    if len(append_episodes) != args.append_episodes:
        raise ValueError(f"selected {len(append_episodes)} append episodes, expected {args.append_episodes}")

    total_base_episodes = int(base_info["total_episodes"])
    total_base_frames = int(base_info["total_frames"])
    total_append_frames = int(append_episodes["length"].sum())
    total_episodes = total_base_episodes + int(len(append_episodes))
    total_frames = total_base_frames + total_append_frames

    base_data_files = sorted((base_root / "data").glob("chunk-*/*.parquet"))
    for src in base_data_files:
        dst = output_root / src.relative_to(base_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    append_file_index = max(int(p.stem.split("-")[1]) for p in base_data_files) + 1
    selected_episode_ids = set(int(v) for v in append_episodes["episode_index"].tolist())
    append_episode_mapping = {
        int(old_idx): total_base_episodes + new_offset
        for new_offset, old_idx in enumerate(append_episodes["episode_index"].tolist())
    }

    append_data = _append_data_table(
        append_root,
        selected_episode_ids,
        episode_mapping=append_episode_mapping,
        frame_offset=total_base_frames,
        episode_length=int(append_episodes["length"].iloc[0]),
    )
    append_data_path = output_root / "data" / "chunk-000" / f"file-{append_file_index:03d}.parquet"
    _write_table_with_metadata(append_data, append_data_path, append_data.schema.metadata)

    adjusted_append = _adjust_append_episode_rows(
        append_episodes,
        episode_offset=0,
        frame_offset=0,
    )
    adjusted_append["episode_index"] = [
        append_episode_mapping[int(old_idx)] for old_idx in append_episodes["episode_index"].tolist()
    ]
    for row_offset, (row_index, row) in enumerate(append_episodes.iterrows()):
        length = int(row["length"])
        old_from = int(row["dataset_from_index"])
        new_from = total_base_frames + row_offset * length
        new_to = new_from + length
        adjusted_append.loc[row_index, "dataset_from_index"] = new_from
        adjusted_append.loc[row_index, "dataset_to_index"] = new_to
        index_delta = new_from - old_from
        for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
            idx_col = f"stats/index/{stat}"
            if idx_col in adjusted_append:
                adjusted_append.at[row_index, idx_col] = _add_to_stat_cell(row[idx_col], index_delta)
    for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        ep_col = f"stats/episode_index/{stat}"
        if ep_col in adjusted_append:
            adjusted_append[ep_col] = [
                [append_episode_mapping[int(old_idx)] + float(np.asarray(value).reshape(-1)[0] - int(old_idx))]
                for old_idx, value in zip(append_episodes["episode_index"].tolist(), append_episodes[ep_col], strict=False)
            ]
    adjusted_append["data/file_index"] = append_file_index
    base_episodes = base_episodes.copy()
    base_episodes["meta/episodes/chunk_index"] = 0
    base_episodes["meta/episodes/file_index"] = 0
    combined_episodes = pd.concat([base_episodes, adjusted_append], ignore_index=True)
    combined_episodes.to_parquet(output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    shutil.copy2(base_root / "meta" / "tasks.parquet", output_root / "meta" / "tasks.parquet")
    info = dict(base_info)
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": base_info["total_tasks"],
            "splits": {"train": f"0:{total_episodes}"},
        }
    )
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")

    from lerobot.datasets.compute_stats import aggregate_stats  # noqa: PLC0415
    from lerobot.datasets.utils import write_stats  # noqa: PLC0415

    episode_stats = [
        _stats_from_episode_row(row, base_info["features"])
        for row in combined_episodes.to_dict(orient="records")
    ]
    write_stats(aggregate_stats(episode_stats), output_root)

    report = {
        "ok": True,
        "repo_id": args.repo_id,
        "output_root": str(output_root),
        "base_root": str(base_root),
        "append_root": str(append_root),
        "append_episodes": int(len(append_episodes)),
        "append_balance_by_task": args.append_balance_by_task,
        "append_episode_ids": sorted(selected_episode_ids),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "base_data_files_copied": len(base_data_files),
        "append_data_file_index": append_file_index,
    }
    report_path = _resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
