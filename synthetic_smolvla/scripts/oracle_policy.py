#!/usr/bin/env python3
"""Deterministic oracle scaffold for synthetic OpenArm pick demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

from language import instruction_for_object, parse_target_object
from sim_contract import clamp, gripper_deg_to_sim_finger_m, validate_joint_targets


BASE_RIGHT_POSE = {
    "joint_1": 0.0,
    "joint_2": 20.0,
    "joint_3": 0.0,
    "joint_4": 55.0,
    "joint_5": 0.0,
    "joint_6": 15.0,
    "joint_7": 0.0,
}

OBJECT_APPROACH_POSES = {
    "orange_ball": {"joint_1": -18.0, "joint_2": 28.0, "joint_3": 8.0, "joint_4": 66.0},
    "red_cube": {"joint_1": -8.0, "joint_2": 31.0, "joint_3": 1.0, "joint_4": 70.0},
    "green_cube": {"joint_1": 9.0, "joint_2": 32.0, "joint_3": -3.0, "joint_4": 70.0},
    "blue_cube": {"joint_1": 20.0, "joint_2": 29.0, "joint_3": -9.0, "joint_4": 67.0},
}

OBJECT_LOWER_DELTAS = {
    "orange_ball": {"joint_2": 4.0, "joint_4": 12.0},
    "red_cube": {"joint_2": 3.0, "joint_4": 11.0},
    "green_cube": {"joint_2": 3.0, "joint_4": 11.0},
    "blue_cube": {"joint_2": 4.0, "joint_4": 12.0},
}

DEFAULT_OBJECT_POSES_M = {
    "orange_ball": [0.42, -0.12, 0.43],
    "red_cube": [0.50, -0.04, 0.43],
    "green_cube": [0.50, 0.08, 0.43],
    "blue_cube": [0.38, 0.12, 0.43],
}


@dataclass(frozen=True)
class OracleStep:
    name: str
    target_object: str
    arm_side: str
    joint_targets_deg: dict[str, float]
    gripper_target_deg: float
    sim_finger_target_m: float
    cartesian_hint_m: list[float]

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["action"] = {
            "joint_targets_deg": self.joint_targets_deg,
            "gripper_target_deg": self.gripper_target_deg,
            "sim_finger_target_m": self.sim_finger_target_m,
        }
        return record


def merged_pose(*parts: dict[str, float]) -> dict[str, float]:
    pose = dict(BASE_RIGHT_POSE)
    for part in parts:
        pose.update(part)
    return pose


def pose_with_delta(pose: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    updated = dict(pose)
    for joint, amount in delta.items():
        updated[joint] = float(updated[joint]) + float(amount)
    return updated


def jitter_object_poses(seed: int, *, enabled: bool) -> dict[str, list[float]]:
    rng = random.Random(seed)
    poses = {name: list(pose) for name, pose in DEFAULT_OBJECT_POSES_M.items()}
    if not enabled:
        return poses
    for pose in poses.values():
        pose[0] = clamp(pose[0] + rng.uniform(-0.035, 0.035), 0.28, 0.62)
        pose[1] = clamp(pose[1] + rng.uniform(-0.035, 0.035), -0.20, 0.20)
    return poses


def build_oracle_steps(instruction: str, *, arm_side: str = "right") -> list[OracleStep]:
    target_object = parse_target_object(instruction)
    approach = merged_pose(OBJECT_APPROACH_POSES[target_object])
    lower = pose_with_delta(approach, OBJECT_LOWER_DELTAS[target_object])
    lift = pose_with_delta(approach, {"joint_2": -6.0, "joint_4": -8.0})
    object_pose = DEFAULT_OBJECT_POSES_M[target_object]

    raw_steps = [
        ("open_gripper", approach, -65.0, [object_pose[0], object_pose[1], object_pose[2] + 0.10]),
        ("move_above_target", approach, -65.0, [object_pose[0], object_pose[1], object_pose[2] + 0.08]),
        ("lower_to_target", lower, -65.0, [object_pose[0], object_pose[1], object_pose[2] + 0.02]),
        ("close_gripper", lower, 0.0, [object_pose[0], object_pose[1], object_pose[2] + 0.02]),
        ("lift_target", lift, 0.0, [object_pose[0], object_pose[1], object_pose[2] + 0.11]),
    ]

    steps: list[OracleStep] = []
    for name, joints, gripper_deg, cartesian_hint in raw_steps:
        validate_joint_targets(arm_side, {**joints, "gripper": gripper_deg})
        steps.append(
            OracleStep(
                name=name,
                target_object=target_object,
                arm_side=arm_side,
                joint_targets_deg=dict(joints),
                gripper_target_deg=gripper_deg,
                sim_finger_target_m=gripper_deg_to_sim_finger_m(gripper_deg),
                cartesian_hint_m=[float(value) for value in cartesian_hint],
            )
        )
    return steps


def generate_episode(
    episode_index: int,
    instruction: str,
    *,
    seed: int,
    randomized: bool,
    all_objects_visible: bool,
    arm_side: str = "right",
) -> dict[str, Any]:
    target_object = parse_target_object(instruction)
    steps = build_oracle_steps(instruction, arm_side=arm_side)
    object_poses = jitter_object_poses(seed + episode_index, enabled=randomized)
    visible_objects = sorted(object_poses) if all_objects_visible else [target_object]
    return {
        "schema_version": "openarm_synth_oracle_manifest_v1",
        "source": "synthetic_smolvla.oracle_policy",
        "episode_index": int(episode_index),
        "instruction": instruction,
        "target_object": target_object,
        "arm_side": arm_side,
        "randomized": bool(randomized),
        "all_objects_visible": bool(all_objects_visible),
        "visible_objects": visible_objects,
        "object_poses_m": object_poses,
        "steps": [step.as_record() for step in steps],
        "success_label": True,
        "wrong_object_lifted": False,
        "limit_exceeded": False,
        "rgb_frames": [],
        "notes": "Oracle scaffold only; Isaac RGB/physics capture is the next implementation step.",
    }


def balanced_instructions() -> tuple[str, ...]:
    return tuple(instruction_for_object(name) for name in sorted(OBJECT_APPROACH_POSES))

