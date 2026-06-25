from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import replay_openarm_saved_episode_real as replay  # noqa: E402


def test_height_map_uses_only_upsampled_episode_paths() -> None:
    for height in ("125", "122.5", "120", "117.5", "115"):
        path = replay.episode_path_for_height(height)

        assert path.name == "episode_000000.npz"
        assert path.parent.name == "episodes"
        assert path.parent.parent.name.endswith("_upsampled")
        assert path.is_file()


def test_requested_upsampled_height_episodes_satisfy_replay_contract() -> None:
    for height in ("125", "122.5", "120", "117.5", "115"):
        episode = replay.episode_path_for_height(height)
        rows = replay.load_action_rows(episode)

        audit = replay.validate_action_contract(
            rows,
            episode=episode,
            height=height,
            profile=replay.REPLAY_PROFILES["10hz"],
            side="left",
            rate_hz=10.0,
            first_zero_tolerance_deg=0.25,
        )

        assert audit.commands > 0
        assert audit.max_step_deg <= 2.0
        assert audit.max_speed_deg_s <= 20.0
        assert audit.expected_duration_10hz_sec == pytest.approx(audit.commands / 10.0)
        assert max(abs(value) for value in rows[0]) <= 0.25


def test_20hz400_height_episodes_satisfy_replay_contract() -> None:
    profile = replay.REPLAY_PROFILES["20hz400"]
    for height in ("112.5", "110", "107.5"):
        episode = replay.episode_path_for_height(height, dataset_family="20hz400")
        rows = replay.load_action_rows(episode)

        audit = replay.validate_action_contract(
            rows,
            episode=episode,
            height=height,
            profile=profile,
            side="left",
            rate_hz=20.0,
            first_zero_tolerance_deg=0.25,
        )

        assert audit.dataset_family == "20hz400"
        assert audit.commands == 400
        assert audit.max_step_deg <= 1.5
        assert audit.max_speed_deg_s <= 30.0
        assert audit.expected_duration_sec == pytest.approx(20.0)
        assert max(abs(value) for value in rows[0]) <= 0.25


def test_auto_profile_selects_lower_height_20hz400() -> None:
    profile = replay.select_replay_profile("112.5", dataset_family="auto")

    assert profile.name == "20hz400"


def test_non_upsampled_episode_is_refused() -> None:
    path = (
        REPO_ROOT
        / "synthetic_smolvla"
        / "datasets"
        / "openarm_photo_clean_v1_one_per_height"
        / "h125cm"
        / "episodes"
        / "episode_000000.npz"
    )

    with pytest.raises(replay.ReplaySafetyError, match="non-upsampled"):
        replay.load_action_rows(path)


def test_contract_refuses_slew_above_twenty_degrees_per_second() -> None:
    rows = [[0.0] * 8, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    with pytest.raises(replay.ReplaySafetyError, match="20.0 deg/s"):
        replay.validate_action_contract(
            rows,
            episode=Path("episode_000000.npz"),
            height="125",
            side="left",
            rate_hz=10.01,
            first_zero_tolerance_deg=0.25,
        )


def test_real_mode_requires_exact_ten_hz() -> None:
    args = replay.build_arg_parser().parse_args(
        [
            "--height",
            "125",
            "--no-dry-run",
            "--confirm-real-hardware",
            "--confirm-height",
            "--rate-hz",
            "9.0",
        ]
    )
    profile = replay.select_replay_profile(args.height, dataset_family=args.dataset_family, rate_hz=args.rate_hz)
    replay.apply_profile_defaults(args, profile)

    with pytest.raises(replay.ReplaySafetyError, match="fixed at 10.0 Hz"):
        replay.validate_real_flags(args, profile=profile)


def test_real_mode_requires_exact_twenty_hz_for_20hz400() -> None:
    args = replay.build_arg_parser().parse_args(
        [
            "--height",
            "112.5",
            "--no-dry-run",
            "--confirm-real-hardware",
            "--confirm-height",
            "--rate-hz",
            "10.0",
        ]
    )
    profile = replay.select_replay_profile(args.height, dataset_family=args.dataset_family, rate_hz=args.rate_hz)
    replay.apply_profile_defaults(args, profile)

    with pytest.raises(replay.ReplaySafetyError, match="fixed at 20.0 Hz"):
        replay.validate_real_flags(args, profile=profile)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, list[float] | float]] = []

    def send_arm(self, row_index: int, arm_deg: list[float]) -> None:
        self.events.append(("arm", row_index, list(arm_deg)))

    def send_gripper(self, row_index: int, gripper_deg: float) -> None:
        self.events.append(("gripper", row_index, float(gripper_deg)))


def test_replay_rows_sends_arm_then_gripper_for_each_saved_row() -> None:
    rows = [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -10.0],
        [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, -11.0],
    ]
    sink = RecordingSink()

    replay.replay_rows(rows, sink=sink, rate_hz=10.0, sleep_enabled=False)

    assert sink.events == [
        ("arm", 0, rows[0][:7]),
        ("gripper", 0, rows[0][7]),
        ("arm", 1, rows[1][:7]),
        ("gripper", 1, rows[1][7]),
    ]
