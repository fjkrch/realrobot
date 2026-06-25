from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "synthetic_smolvla" / "scripts"))

from collect_height_sweep_successes import (  # noqa: E402
    _collector_command,
    _merge_command,
    apply_height_to_config,
    build_arg_parser,
    height_tag,
    parse_heights,
    parse_quota_counts,
)


def _config():
    return {
        "scene": {
            "name": "base_scene",
            "layout_info_cm": {"robot_plus_table_height": 125.0},
            "camera": {"eye_m": [0.56, 0.0, 1.08], "target_m": [0.76, 0.0, 0.745]},
        },
        "robot": {"base_pose_m": [0.38, 0.0, 0.60]},
    }


def test_height_tag_is_filesystem_friendly():
    assert height_tag(122.5) == "h122p5cm"
    assert height_tag(120.0) == "h120cm"


def test_parse_heights():
    assert parse_heights("125,122.5,120") == [125.0, 122.5, 120.0]


def test_parse_default_quota_counts():
    assert parse_quota_counts("3,3,2,2", ["orange_ball", "red_cube", "green_cube", "blue_cube"]) == {
        "orange_ball": 3,
        "red_cube": 3,
        "green_cube": 2,
        "blue_cube": 2,
    }


def test_apply_baseline_height_keeps_root_z():
    adjusted, delta = apply_height_to_config(_config(), height_cm=125.0)
    assert delta == 0.0
    assert adjusted["robot"]["base_pose_m"] == [0.38, 0.0, 0.6]
    assert adjusted["scene"]["layout_info_cm"]["robot_plus_table_height"] == 125.0


def test_apply_lower_height_moves_robot_and_camera_down():
    adjusted, delta = apply_height_to_config(_config(), height_cm=120.0)
    assert abs(delta - (-0.05)) < 1e-9
    assert adjusted["robot"]["base_pose_m"] == [0.38, 0.0, 0.55]
    assert adjusted["scene"]["camera"]["eye_m"] == [0.56, 0.0, 1.03]
    assert adjusted["scene"]["height_sweep"]["applied_robot_root_z_delta_m"] == -0.05


def test_apply_lower_height_can_leave_camera_fixed():
    adjusted, _ = apply_height_to_config(_config(), height_cm=120.0, shift_camera_with_robot=False)
    assert adjusted["robot"]["base_pose_m"] == [0.38, 0.0, 0.55]
    assert adjusted["scene"]["camera"]["eye_m"] == [0.56, 0.0, 1.08]


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_launcher_defaults_are_50_command_contract():
    args = build_arg_parser().parse_args([])
    assert args.target_quotas == "3,3,2,2"
    assert args.episode_commands == 50
    assert args.fps == 10
    assert args.substeps == 20
    assert args.gripper_close_deg == -10.0
    assert sum([args.approach_steps, args.descend_steps, args.close_steps, args.lift_steps, args.hold_steps]) == 50


def test_collector_command_uses_exact_quotas_and_8d_upsampling():
    args = build_arg_parser().parse_args([])
    command = _collector_command(args, height_cm=122.5, height_config=Path("/tmp/h122p5.yaml"), index=1)
    assert "--successes-per-target" not in command
    assert "--max-arm-action-step-deg" not in command
    assert _flag_value(command, "--target-quotas") == "3,3,2,2"
    assert _flag_value(command, "--fps") == "10"
    assert _flag_value(command, "--substeps") == "20"
    assert _flag_value(command, "--approach-steps") == "14"
    assert _flag_value(command, "--descend-steps") == "12"
    assert _flag_value(command, "--close-steps") == "8"
    assert _flag_value(command, "--lift-steps") == "12"
    assert _flag_value(command, "--hold-steps") == "4"
    assert _flag_value(command, "--grasp-close-deg") == "-10.0"
    assert _flag_value(command, "--max-gripper-close-deg") == "-10.0"
    assert _flag_value(command, "--max-action-step-deg") == "1.000000"
    assert "--early-stop-on-lift" in command
    assert "openarm_height_sweep_lift5cm_10hz_50step_h122p5cm" in " ".join(command)


def test_collector_command_passes_renderer_settings_when_requested():
    args = build_arg_parser().parse_args([
        "--experience",
        "isaaclab.python.headless.rendering.kit",
        "--rendering-mode",
        "performance",
        "--kit-args",
        "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1",
    ])
    command = _collector_command(args, height_cm=125.0, height_config=Path("/tmp/h125.yaml"), index=0)
    assert _flag_value(command, "--experience") == "isaaclab.python.headless.rendering.kit"
    assert _flag_value(command, "--rendering-mode") == "performance"
    assert _flag_value(command, "--kit-args") == "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1"


def test_merge_command_caps_combined_dataset_at_50_episodes():
    args = build_arg_parser().parse_args([])
    heights = [125.0, 122.5, 120.0, 117.5, 115.0]
    command = _merge_command(args, heights, quota_total=10)
    assert _flag_value(command, "--max-total-episodes") == "50"
    assert _flag_value(command, "--fps") == "10"
    assert _flag_value(command, "--repo-id") == "local/openarm_height_sweep_lift5cm_10hz_50eps_50step"
    assert command.count("--input") == 5


def test_one_per_height_launcher_uses_zero_start_122p5_and_one_any_target_success():
    text = (REPO_ROOT / "synthetic_smolvla" / "scripts" / "_run_one_per_height.sh").read_text()
    assert "122.5 122p5cm" in text
    assert "--heights-cm 125,122.5,120,117.5,115" in text
    assert "--target-weights 1,1,1,1 --max-keep 1" in text
    assert "--target-quotas 0,0,1,0" not in text
    assert "--record-zero-to-init --zero-init-steps 20 --zero-start-gripper-deg 0 --init-gripper-deg -50" in text
    assert "--approach-steps 20 --descend-steps 20 --close-steps 16 --lift-steps 24 --hold-steps 0" in text
    assert "--gripper-close-range-deg -17 -13" in text
    assert "--action-clip-tol-deg 180.0" in text
    assert "--max-step-deg 2.0 --fps 10" in text


def test_photo_clean_replay_configs_enable_black_grey_robot_palette():
    config_paths = [
        REPO_ROOT / "synthetic_smolvla" / "configs" / "scene_openarm_real_photo_left_centered_clean_v1.yaml",
        *(
            REPO_ROOT
            / "synthetic_smolvla"
            / "configs"
            / "generated_height_sweep_photo_clean_v1"
            / f"scene_openarm_real_photo_left_centered_clean_v1_{tag}.yaml"
            for tag in ["h125cm", "h122p5cm", "h120cm", "h117p5cm", "h115cm"]
        ),
    ]
    for path in config_paths:
        text = path.read_text()
        assert "palette: black_grey_real_arm" in text
        assert "hide_inactive_arm: true" in text
        assert "camera_model: Intel RealSense D435i RGB" in text
        assert "horizontal_fov_deg: 69.4" in text
        assert "vertical_fov_deg: 42.5" in text
        assert "horizontal_aperture_mm: 20.955" in text
        assert "focal_length_mm: 15.131432" in text
