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
    finger_table_penetration,
    object_pushed_down,
    refined_action_clip,
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
