"""Unit checks for the SmolVLA mirror safety helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "synthetic_smolvla" / "scripts"))

from audit_mirror_trace import audit  # noqa: E402
from mirror_sinks import (  # noqa: E402
    CommandContext,
    REQUIRED_REAL_CONFIRMATION,
    DryRunMirrorSink,
    MirrorSafetyError,
    RealMirrorConfig,
    RealMirrorSink,
    clamp_command_deg,
    start_pose_from_config,
)


def test_command_clamp_uses_sim_contract_limits() -> None:
    result = clamp_command_deg("right", [999, -999, 999, -999, 999, -999, 999, 100])

    assert result.command_deg == [73.0, -7.0, 83.0, 2.0, 83.0, -38.0, 78.0, 0.0]
    assert result.clamp_events == 8


def test_real_mirror_config_requires_exact_confirmation() -> None:
    with pytest.raises(MirrorSafetyError):
        RealMirrorConfig(
            side="right",
            port="can0",
            confirm="yes",
            rate_hz=2.0,
            max_joint_delta_deg=3.0,
            watchdog_timeout_sec=2.0,
        )

    cfg = RealMirrorConfig(
        side="right",
        port="can0",
        confirm=REQUIRED_REAL_CONFIRMATION,
        rate_hz=2.0,
        max_joint_delta_deg=3.0,
        watchdog_timeout_sec=2.0,
    )
    assert cfg.disable_gripper_real is True


def test_real_mirror_interpolates_large_deltas_to_max_step() -> None:
    cfg = RealMirrorConfig(
        side="right",
        port="can0",
        confirm=REQUIRED_REAL_CONFIRMATION,
        rate_hz=10.0,
        max_joint_delta_deg=3.0,
        watchdog_timeout_sec=2.0,
    )
    sink = RealMirrorSink(cfg)
    start = [0.0, 20.0, 0.0, 55.0, 0.0, 15.0, 0.0, -65.0]
    target = [0.0, 20.0, 0.0, 67.1, 0.0, 15.0, 0.0, -65.0]

    path = sink._interpolated_targets(start, target)  # noqa: SLF001 - direct safety helper check

    assert len(path) == 5
    prev = start
    for item in path:
        assert max(abs(item[index] - prev[index]) for index in range(7)) <= 3.0
        prev = item
    assert path[-1] == target


def test_start_pose_reads_scene_reset_pose() -> None:
    pose = start_pose_from_config(
        "synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml",
        "right",
    )

    assert pose == [0.0, 20.0, 0.0, 55.0, 0.0, 15.0, 0.0, -65.0]


def test_dry_run_trace_and_audit_pass(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    sink = DryRunMirrorSink(trace, side="right", disable_gripper_real=True)
    try:
        sink.emit(
            [0.0, 20.0, 0.0, 55.0, 0.0, 15.0, 0.0, -65.0],
            context=CommandContext(task_index=1, step_index=0, target_object="red_cube"),
        )
        sink.emit(
            [1.0, 20.5, 0.0, 55.0, 0.0, 15.0, 0.0, -64.0],
            context=CommandContext(task_index=1, step_index=1, target_object="red_cube"),
        )
    finally:
        sink.close()

    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    summary = audit(records, side="right", max_delta=2.0, expected_steps=2)

    assert summary["ok"] is True
    assert summary["records"] == 2
    assert summary["gripper_sent_to_real"] is False


def test_audit_fails_when_step_delta_exceeds_threshold() -> None:
    records = [
        {"_line_no": 1, "command_deg": [0, 20, 0, 55, 0, 15, 0, -65], "gripper_sent_to_real": False},
        {"_line_no": 2, "command_deg": [10, 20, 0, 55, 0, 15, 0, -65], "gripper_sent_to_real": False},
    ]

    summary = audit(records, side="right", max_delta=3.0, expected_steps=2)

    assert summary["ok"] is False
    assert any("exceeds" in error for error in summary["errors"])
