#!/usr/bin/env python3
"""Export oracle manifests to the installed LeRobot dataset format."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from sim_contract import JOINT_NAMES


STATE_NAMES = [*JOINT_NAMES, "gripper"]
CAMERA_KEY = "observation.images.camera1"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def lerobot_features(image_size: int = 96) -> dict[str, dict[str, Any]]:
    return {
        CAMERA_KEY: {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
    }


def _object_color(name: str) -> tuple[int, int, int]:
    return {
        "orange_ball": (245, 112, 12),
        "red_cube": (220, 24, 24),
        "green_cube": (28, 174, 62),
        "blue_cube": (28, 64, 224),
    }.get(name, (180, 180, 180))


def _project_xy(pose_m: list[float], image_size: int) -> tuple[int, int]:
    x, y, _ = pose_m
    px = int(np.interp(x, [0.28, 0.62], [14, image_size - 15]))
    py = int(np.interp(y, [-0.20, 0.20], [image_size - 15, 14]))
    return px, py


def render_synthetic_camera(record: dict[str, Any], *, image_size: int = 96) -> np.ndarray:
    """Render a simple top-down RGB view of visible objects.

    This is a deterministic placeholder until Isaac camera capture is wired.
    It preserves target color, visible-distractor layout, and target emphasis so
    the dataset schema can be exercised by LeRobot tooling now.
    """
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image[:, :] = (38, 44, 48)
    table_margin = 8
    image[table_margin:-table_margin, table_margin:-table_margin] = (118, 112, 100)

    target = record["target_object"]
    poses = record["object_poses_m"]
    visible = record.get("visible_objects") or [target]
    for name in visible:
        px, py = _project_xy(poses[name], image_size)
        color = _object_color(name)
        radius = 4 if name == "orange_ball" else 5
        yy, xx = np.ogrid[:image_size, :image_size]
        if name == "orange_ball":
            mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius**2
        else:
            mask = (np.abs(xx - px) <= radius) & (np.abs(yy - py) <= radius)
        image[mask] = color
        if name == target:
            ring = (xx - px) ** 2 + (yy - py) ** 2 <= (radius + 3) ** 2
            image[ring & ~mask] = (250, 250, 250)
    return image


def _state_from_step(step: dict[str, Any]) -> np.ndarray:
    action = step["action"]
    joints = action["joint_targets_deg"]
    values = [float(joints[name]) for name in JOINT_NAMES]
    values.append(float(action["gripper_target_deg"]))
    return np.asarray(values, dtype=np.float32)


def _episode_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    visible_all = 0
    for record in records:
        counts[record["target_object"]] = counts.get(record["target_object"], 0) + 1
        visible_all += int(bool(record.get("all_objects_visible")))
    return {
        "schema_version": "openarm_synth_lerobot_export_v1",
        "episodes": len(records),
        "target_counts": counts,
        "all_objects_visible_episodes": visible_all,
        "source_manifest_note": "RGB frames are deterministic synthetic top-down placeholders, not Isaac camera renders yet.",
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def export_lerobot_dataset(
    *,
    manifest_path: Path,
    output_dir: Path,
    repo_id: str,
    fps: int = 10,
    image_size: int = 96,
    overwrite: bool = False,
) -> dict[str, Any]:
    records = load_manifest(manifest_path)
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty dataset directory: {output_dir}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type="openarm_synthetic_isaac",
        features=lerobot_features(image_size),
        use_videos=False,
        image_writer_threads=0,
        image_writer_processes=0,
    )

    for record in records:
        previous_state = None
        for step in record["steps"]:
            action = _state_from_step(step)
            state = action if previous_state is None else previous_state
            dataset.add_frame(
                {
                    CAMERA_KEY: render_synthetic_camera(record, image_size=image_size),
                    STATE_KEY: state,
                    ACTION_KEY: action,
                    "task": record["instruction"],
                }
            )
            previous_state = action
        dataset.save_episode()
    dataset.finalize()

    metadata = _episode_metadata(records)
    metadata.update(
        {
            "repo_id": repo_id,
            "root": str(output_dir),
            "fps": fps,
            "image_size": image_size,
            "frames_per_episode": len(records[0]["steps"]) if records else 0,
        }
    )
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
        metadata["restored_manifest"] = str(manifest_path)
    (output_dir / "openarm_synth_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
