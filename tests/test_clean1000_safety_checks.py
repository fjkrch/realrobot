"""Unit checks for the clean-1000 collector safety helpers.

These exercise the pure-python logic of the new safety checks added to
``collect_dense_isaac_dataset.py`` (finger/table penetration, object-pushed-down,
and the refined action-clip flag) WITHOUT importing Isaac or touching a GPU. The
in-loop collector mirrors this logic with torch tensors.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "synthetic_smolvla" / "scripts"))

from collect_dense_isaac_dataset import (  # noqa: E402
    build_arg_parser,
    cap_gripper_close_deg,
    dense_phase_plan,
    finger_table_penetration,
    hold_pad_to_length,
    limit_action_step_deg,
    normalize_gripper_close_range_deg,
    object_pushed_down,
    parse_target_quotas,
    refined_action_clip,
    target_quotas_satisfied,
    zero_to_init_command_deg,
)
from make_scene import (  # noqa: E402
    REAL_ARM_BLACK_MATERIAL_PATH,
    REAL_ARM_DARK_MATERIAL_PATH,
    REAL_ARM_SILVER_MATERIAL_PATH,
    _camera_focal_length_mm,
    _real_arm_material_for_path,
    _robot_material_output_color,
    _should_hide_robot_visual_path,
)

TABLE_TOP_Z = 0.73
MARGIN = 0.003


# --- finger_table_penetration -------------------------------------------------

def test_penetration_when_finger_below_top_and_over_footprint():
    # Finger 5 mm below the table top while over the footprint -> penetration.
    min_clear, penetrated = finger_table_penetration(
        finger_z=[TABLE_TOP_Z - 0.005, TABLE_TOP_Z + 0.02],
        over_table=[True, True],
        table_top_z=TABLE_TOP_Z,
        margin_m=MARGIN,
    )
    assert penetrated is True
    assert min_clear < 0.0


def test_no_penetration_when_below_top_but_off_footprint():
    # The exact bug in the old TCP check: a finger dips below the table plane but is
    # beside the table (not over the footprint) -> must NOT count as penetration.
    min_clear, penetrated = finger_table_penetration(
        finger_z=[TABLE_TOP_Z - 0.08, TABLE_TOP_Z - 0.05],
        over_table=[False, False],
        table_top_z=TABLE_TOP_Z,
        margin_m=MARGIN,
    )
    assert penetrated is False
    assert min_clear == float("inf")


def test_no_penetration_when_finger_above_table():
    min_clear, penetrated = finger_table_penetration(
        finger_z=[TABLE_TOP_Z + 0.01, TABLE_TOP_Z + 0.03],
        over_table=[True, True],
        table_top_z=TABLE_TOP_Z,
        margin_m=MARGIN,
    )
    assert penetrated is False
    assert min_clear > 0.0


def test_penetration_uses_lowest_over_footprint_finger():
    # Only the lower finger is over the footprint; clearance comes from it.
    min_clear, penetrated = finger_table_penetration(
        finger_z=[TABLE_TOP_Z - 0.01, TABLE_TOP_Z + 0.05],
        over_table=[True, False],
        table_top_z=TABLE_TOP_Z,
        margin_m=MARGIN,
    )
    assert penetrated is True
    assert abs(min_clear - (-0.01)) < 1e-9


def test_clearance_within_margin_is_not_penetration():
    # 2 mm below top is within the 3 mm margin -> contact-but-not-rejected.
    min_clear, penetrated = finger_table_penetration(
        finger_z=[TABLE_TOP_Z - 0.002],
        over_table=[True],
        table_top_z=TABLE_TOP_Z,
        margin_m=MARGIN,
    )
    assert penetrated is False
    assert min_clear < 0.0


# --- object_pushed_down -------------------------------------------------------

def test_object_pushed_down_true_when_drop_exceeds_margin():
    assert object_pushed_down(obj_z=[0.74, 0.700], rest_z=[0.745, 0.765], margin_m=0.005) is True


def test_object_pushed_down_false_when_within_margin():
    # Each object dropped exactly 4 mm (< 5 mm margin) -> not pushed down.
    assert object_pushed_down(obj_z=[0.741, 0.761], rest_z=[0.745, 0.765], margin_m=0.005) is False


def test_object_pushed_down_false_when_objects_rise():
    assert object_pushed_down(obj_z=[0.80, 0.79], rest_z=[0.745, 0.765], margin_m=0.005) is False


# --- refined_action_clip ------------------------------------------------------

JOINT4_IDX = 3
TOL = 1.0
J4_STARTUP_TOL = 3.0


def _no_clip():
    return [0.0] * 7


def test_joint4_zero_start_clamp_is_ignored():
    # joint_4 desired below its 2 deg floor, clamped up by 2 deg -> ignored (the
    # confound that made the old limit_exceeded flag fire on ~100% of episodes).
    jdes = _no_clip()
    jclamped = _no_clip()
    jdes[JOINT4_IDX] = 0.0
    jclamped[JOINT4_IDX] = 2.0  # clip magnitude 2 deg <= startup tol 3 deg
    clipped, max_clip = refined_action_clip(jdes, jclamped, JOINT4_IDX, TOL, J4_STARTUP_TOL)
    assert clipped is False
    assert max_clip == 0.0


def test_large_joint4_clip_is_flagged():
    jdes = _no_clip()
    jclamped = _no_clip()
    jdes[JOINT4_IDX] = -8.0
    jclamped[JOINT4_IDX] = 2.0  # clip magnitude 10 deg > startup tol -> genuine clip
    clipped, max_clip = refined_action_clip(jdes, jclamped, JOINT4_IDX, TOL, J4_STARTUP_TOL)
    assert clipped is True
    assert abs(max_clip - 10.0) < 1e-9


def test_other_joint_clip_is_flagged():
    jdes = _no_clip()
    jclamped = _no_clip()
    jdes[1] = 90.0
    jclamped[1] = 85.0  # joint_2 clipped 5 deg -> genuine clip
    clipped, max_clip = refined_action_clip(jdes, jclamped, JOINT4_IDX, TOL, J4_STARTUP_TOL)
    assert clipped is True
    assert abs(max_clip - 5.0) < 1e-9


def test_small_clips_under_tolerance_are_not_flagged():
    jdes = _no_clip()
    jclamped = _no_clip()
    jdes[0] = 0.5
    jclamped[0] = 0.0  # 0.5 deg < 1 deg tol
    clipped, max_clip = refined_action_clip(jdes, jclamped, JOINT4_IDX, TOL, J4_STARTUP_TOL)
    assert clipped is False
    assert abs(max_clip - 0.5) < 1e-9


# --- per-target quotas --------------------------------------------------------

OBJECTS = ["orange_ball", "red_cube", "green_cube", "blue_cube"]


def test_successes_per_target_builds_uniform_quotas():
    assert parse_target_quotas(None, OBJECTS, successes_per_target=1) == {
        "orange_ball": 1,
        "red_cube": 1,
        "green_cube": 1,
        "blue_cube": 1,
    }


def test_parse_aligned_target_quota_counts():
    assert parse_target_quotas("1,0,2,3", OBJECTS) == {
        "orange_ball": 1,
        "red_cube": 0,
        "green_cube": 2,
        "blue_cube": 3,
    }


def test_parse_named_target_quota_counts():
    assert parse_target_quotas("red_cube=2,blue_cube=1", OBJECTS) == {
        "orange_ball": 0,
        "red_cube": 2,
        "green_cube": 0,
        "blue_cube": 1,
    }


def test_target_quotas_satisfied():
    quotas = {"orange_ball": 1, "red_cube": 1}
    assert target_quotas_satisfied({"orange_ball": 1, "red_cube": 0}, quotas) is False
    assert target_quotas_satisfied({"orange_ball": 1, "red_cube": 1}, quotas) is True


# --- command upsampling / gripper cap ----------------------------------------

def test_gripper_close_cap_never_exceeds_minus_10():
    assert cap_gripper_close_deg(-5.0, -10.0) == -10.0
    assert cap_gripper_close_deg(-10.0, -10.0) == -10.0
    assert cap_gripper_close_deg(-25.0, -10.0) == -25.0


def test_gripper_close_cap_prevents_post_init_commands_above_minus_13():
    assert cap_gripper_close_deg(-12.0, -13.0) == -13.0
    assert cap_gripper_close_deg(-13.0, -13.0) == -13.0
    assert cap_gripper_close_deg(-17.0, -13.0) == -17.0


def test_gripper_close_range_accepts_minus_17_to_minus_13():
    assert normalize_gripper_close_range_deg([-17.0, -13.0]) == (-17.0, -13.0)


def test_black_grey_real_arm_palette_outputs_black_and_silver_parts():
    dark = _robot_material_output_color((0.1, 0.2, 0.3), "black_grey_real_arm")
    bright = _robot_material_output_color((0.98, 0.96, 0.92), "black_grey_real_arm")
    assert dark[0] == dark[1] == dark[2]
    assert bright[0] == bright[1] == bright[2]
    assert dark[0] < bright[0]
    assert dark[0] <= 0.025
    assert bright[0] >= 0.80


def test_d435i_rgb_camera_fov_computes_expected_focal_length():
    camera = {
        "horizontal_aperture_mm": 20.955,
        "horizontal_fov_deg": 69.4,
        "focal_length_mm": 24.0,
    }
    assert abs(_camera_focal_length_mm(camera) - 15.131432) < 1e-5


def test_black_silver_real_arm_path_mapping_uses_solid_materials():
    assert (
        _real_arm_material_for_path(
            "/World/envs/env_0/Robot/openarm_left_link1/visuals/openarm_left_link1_visual/mesh",
            "left",
        )
        == REAL_ARM_BLACK_MATERIAL_PATH
    )
    assert (
        _real_arm_material_for_path(
            "/World/envs/env_0/Robot/openarm_left_link5/visuals/openarm_left_link5_visual/mesh",
            "left",
        )
        == REAL_ARM_BLACK_MATERIAL_PATH
    )
    assert (
        _real_arm_material_for_path(
            "/World/envs/env_0/Robot/openarm_left_link7/visuals/openarm_left_link7_visual/mesh",
            "left",
        )
        == REAL_ARM_BLACK_MATERIAL_PATH
    )
    assert (
        _real_arm_material_for_path(
            "/World/envs/env_0/Robot/openarm_left_left_finger/visuals/openarm_left_left_finger_visual/mesh",
            "left",
        )
        == REAL_ARM_BLACK_MATERIAL_PATH
    )
    assert (
        _real_arm_material_for_path(
            "/World/envs/env_0/Robot/openarm_body_link/visuals/openarm_left_link0_visual/mesh",
            "left",
        )
        == REAL_ARM_DARK_MATERIAL_PATH
    )


def test_black_silver_real_arm_hides_inactive_and_shared_body_visuals():
    assert _should_hide_robot_visual_path(
        "/World/envs/env_0/Robot/openarm_right_link3/visuals/openarm_right_link3_visual/mesh",
        hidden_arm="right",
        hide_inactive=True,
    )
    assert _should_hide_robot_visual_path(
        "/World/envs/env_0/Robot/openarm_body_link/visuals/openarm_body_link0_visual/mesh",
        hidden_arm="right",
        hide_inactive=True,
    )
    assert not _should_hide_robot_visual_path(
        "/World/envs/env_0/Robot/openarm_left_link3/visuals/openarm_left_link3_visual/mesh",
        hidden_arm="right",
        hide_inactive=True,
    )
    assert not _should_hide_robot_visual_path(
        "/World/envs/env_0/Robot/openarm_body_link/visuals/openarm_body_link0_visual/mesh",
        hidden_arm="right",
        hide_inactive=False,
    )


def test_gripper_close_range_rejects_reversed_bounds():
    try:
        normalize_gripper_close_range_deg([-13.0, -17.0])
    except ValueError as exc:
        assert "MIN must be <= MAX" in str(exc)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("expected reversed gripper range to fail")


def test_default_phase_plan_keeps_old_50_command_contract():
    args = build_arg_parser().parse_args([])
    assert args.record_zero_to_init is False
    assert dense_phase_plan(args) == [
        ("approach", 14),
        ("descend", 12),
        ("close", 8),
        ("lift", 12),
        ("hold", 4),
    ]
    assert sum(n for _, n in dense_phase_plan(args)) == 50


def test_zero_start_phase_plan_is_100_commands():
    args = build_arg_parser().parse_args([
        "--record-zero-to-init",
        "--zero-init-steps",
        "20",
        "--approach-steps",
        "20",
        "--descend-steps",
        "20",
        "--close-steps",
        "16",
        "--lift-steps",
        "24",
        "--hold-steps",
        "0",
    ])
    assert dense_phase_plan(args)[0] == ("zero_to_init", 20)
    assert sum(n for _, n in dense_phase_plan(args)) == 100


def test_zero_to_init_first_command_is_all_zeros():
    command = zero_to_init_command_deg(
        [21.13, 4.85, -1.32, 47.8, 11.84, 37.0, -45.88],
        step_index=0,
        steps=20,
        zero_gripper_deg=0.0,
        init_gripper_deg=-50.0,
    )
    assert command == [0.0] * 8


def test_zero_to_init_last_command_reaches_init_and_open_gripper():
    init_arm = [21.13, 4.85, -1.32, 47.8, 11.84, 37.0, -45.88]
    command = zero_to_init_command_deg(
        init_arm,
        step_index=19,
        steps=20,
        zero_gripper_deg=0.0,
        init_gripper_deg=-50.0,
    )
    assert command[:7] == init_arm
    assert command[-1] == -50.0


def test_limit_action_step_caps_all_8_dimensions_to_one_degree():
    previous = [0.0, 5.0, -5.0, 10.0, 20.0, -20.0, 1.0, -65.0]
    target = [10.0, -5.0, 15.0, -10.0, 5.0, 10.0, -20.0, -10.0]
    command, raw_max, applied_max, limited = limit_action_step_deg(
        previous,
        target,
        max_step_deg=1.0,
        gripper_close_cap_deg=-10.0,
    )
    assert limited is True
    assert raw_max == 55.0
    assert applied_max <= 1.0
    assert all(abs(c - p) <= 1.0 for p, c in zip(previous, command, strict=True))
    assert command[-1] <= -10.0


def test_limit_action_step_caps_gripper_even_without_slew_limit():
    command, _, _, _ = limit_action_step_deg(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -65.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0],
        max_step_deg=0.0,
        gripper_close_cap_deg=-10.0,
    )
    assert command[-1] == -10.0


def test_early_lift_stop_hold_pads_to_50_commands():
    held = hold_pad_to_length([[1.0], [2.0], [3.0]], 50)
    assert len(held) == 50
    assert held[:3] == [[1.0], [2.0], [3.0]]
    assert held[-1] == [3.0]
