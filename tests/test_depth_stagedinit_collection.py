from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "synthetic_smolvla" / "scripts"))

from collect_dense_isaac_dataset import (  # noqa: E402
    ACTION_KEY,
    CAMERA_KEY,
    dense_phase_plan,
    DEPTH_KEY,
    interpolate_staged_init_commands,
    load_staged_init_csv,
    LocalNpzEpisodeDataset,
    STATE_KEY,
)
from collect_depth_stagedinit_sweep import (  # noqa: E402
    _append_resume_output,
    _build_config,
    _combo_progress,
    _manifest_kept,
    COMMANDS,
    DEFAULT_BASE_CONFIG,
)
from collect_height_sweep_successes import load_scene_config  # noqa: E402


def _write_stage_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "timestamp_utc,port,side,joint_1.pos,joint_2.pos,joint_3.pos,joint_4.pos,joint_5.pos,joint_6.pos,joint_7.pos,gripper.pos",
                "2026-06-23T00:00:00Z,can1,left,0,-1,1,2,0,0,0,-10",
                "2026-06-23T00:00:01Z,can1,left,3,-1,1,5,0,0,0,-10",
                "2026-06-23T00:00:02Z,can1,left,6,-1,1,8,0,0,0,-13",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_staged_init_csv_loads_three_left_arm_degree_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "joint_positions.csv"
    _write_stage_csv(csv_path)

    stages = load_staged_init_csv(csv_path, expected_side="left")

    assert len(stages) == 3
    assert stages[0] == [0.0, -1.0, 1.0, 2.0, 0.0, 0.0, 0.0, -10.0]
    assert stages[-1][-1] == -13.0


def test_staged_init_interpolation_hits_all_stages_and_obeys_cap(tmp_path: Path) -> None:
    csv_path = tmp_path / "joint_positions.csv"
    _write_stage_csv(csv_path)
    stages = load_staged_init_csv(csv_path, expected_side="left")

    commands = interpolate_staged_init_commands(stages, max_step_deg=1.5)
    max_step = max(
        max(abs(b - a) for a, b in zip(commands[index], commands[index + 1], strict=True))
        for index in range(len(commands) - 1)
    )

    assert commands[0] == stages[0]
    assert stages[1] in commands
    assert commands[-1] == stages[2]
    assert max_step <= 1.5


def test_dense_phase_plan_includes_staged_init_before_grasp_phases() -> None:
    args = Namespace(
        record_zero_to_init=False,
        _staged_init_commands=[[0.0] * 8, [1.0] * 8, [2.0] * 8],
        approach_steps=40,
        descend_steps=40,
        close_steps=32,
        lift_steps=48,
        hold_steps=237,
    )

    assert dense_phase_plan(args) == [
        ("staged_init", 3),
        ("approach", 40),
        ("descend", 40),
        ("close", 32),
        ("lift", 48),
        ("hold", 237),
    ]


def test_depth_stagedinit_config_requests_depth_and_stage1_reset(tmp_path: Path) -> None:
    csv_path = tmp_path / "joint_positions.csv"
    _write_stage_csv(csv_path)
    base_config = load_scene_config(DEFAULT_BASE_CONFIG)

    cfg_path, adjusted = _build_config(
        base_config,
        height_cm=120.0,
        init_name="initA",
        init_csv=csv_path,
        task="orange_ball",
        config_root=tmp_path / "configs",
    )

    camera = adjusted["scene"]["camera"]
    assert camera["resolution"] == [256, 256]
    assert camera["data_types"] == ["rgb", "distance_to_image_plane"]
    assert adjusted["robot"]["reset_pose_deg"]["left"]["joint_1"] == 0.0
    assert adjusted["robot"]["reset_pose_deg"]["left"]["gripper"] == -10.0
    assert adjusted["scene"]["depth_stagedinit"]["task"] == "orange_ball"
    assert DEPTH_KEY == "observation.images.depth"
    assert cfg_path.exists()


def test_local_npz_writer_persists_depth_key(tmp_path: Path) -> None:
    import numpy as np

    dataset = LocalNpzEpisodeDataset.create(
        repo_id="local/test_depth",
        root=tmp_path / "dataset",
        fps=20,
        robot_type="openarm_synthetic_isaac_dense",
        features={
            CAMERA_KEY: {},
            STATE_KEY: {},
            ACTION_KEY: {},
            DEPTH_KEY: {},
        },
    )
    for index in range(2):
        dataset.add_frame(
            {
                CAMERA_KEY: np.full((4, 4, 3), index, dtype=np.uint8),
                STATE_KEY: np.full((8,), index, dtype=np.float32),
                ACTION_KEY: np.full((8,), index, dtype=np.float32),
                DEPTH_KEY: np.full((4, 4, 1), index + 0.25, dtype=np.float32),
                "task": "pick up the orange ball",
            }
        )

    dataset.save_episode()
    saved = np.load(tmp_path / "dataset" / "episodes" / "episode_000000.npz", allow_pickle=True)

    assert DEPTH_KEY in saved.files
    assert saved[DEPTH_KEY].shape == (2, 4, 4, 1)
    assert saved[DEPTH_KEY].dtype == np.float32


def test_resume_append_renumbers_npz_and_updates_manifest_and_meta(tmp_path: Path) -> None:
    import json
    import numpy as np

    dst = tmp_path / "dst"
    src = tmp_path / "src"
    (dst / "episodes").mkdir(parents=True)
    (src / "episodes").mkdir(parents=True)
    for root, indices in ((dst, [0]), (src, [0, 1])):
        for index in indices:
            np.savez_compressed(
                root / "episodes" / f"episode_{index:06d}.npz",
                action=np.zeros((COMMANDS, 8), dtype=np.float32) + index,
                camera=np.zeros((COMMANDS, 2, 2, 3), dtype=np.uint8),
                **{DEPTH_KEY: np.ones((COMMANDS, 2, 2, 1), dtype=np.float32)},
            )
    (dst / "meta.json").write_text(json.dumps({"num_episodes": 1, "num_frames": COMMANDS}) + "\n")

    dst_manifest = tmp_path / "dst_manifest.jsonl"
    src_manifest = tmp_path / "src_manifest.jsonl"
    dst_manifest.write_text(json.dumps({"episode_index": 0, "kept": True}) + "\n", encoding="utf-8")
    src_manifest.write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, "kept": True}),
                json.dumps({"episode_index": 1, "kept": False}),
                json.dumps({"episode_index": 2, "kept": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    append_info = _append_resume_output(
        source_dataset_root=src,
        source_manifest_path=src_manifest,
        destination_dataset_root=dst,
        destination_manifest_path=dst_manifest,
    )

    assert append_info == {"start_index": 1, "appended": 2}
    assert sorted(path.name for path in (dst / "episodes").glob("episode_*.npz")) == [
        "episode_000000.npz",
        "episode_000001.npz",
        "episode_000002.npz",
    ]
    assert len(_manifest_kept(dst_manifest)) == 3
    assert json.loads((dst / "meta.json").read_text())["num_episodes"] == 3
    assert _combo_progress(dataset_root=dst, manifest_path=dst_manifest, target_count=3)["status"] == "complete"
