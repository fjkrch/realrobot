#!/usr/bin/env python3
"""Shared simulation safety contract for synthetic OpenArm SmolVLA work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "synthetic_smolvla"
CONFIG_DIR = PROJECT_ROOT / "configs"

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
SIDES = ("left", "right")

SAFE_ARM_LIMITS_DEG: dict[str, dict[str, tuple[float, float]]] = {
    "right": {
        "joint_1": (-73.0, 73.0),
        "joint_2": (-7.0, 88.0),
        "joint_3": (-83.0, 83.0),
        "joint_4": (2.0, 133.0),
        "joint_5": (-83.0, 83.0),
        "joint_6": (-38.0, 38.0),
        "joint_7": (-78.0, 78.0),
    },
    "left": {
        "joint_1": (-73.0, 73.0),
        "joint_2": (-88.0, 7.0),
        "joint_3": (-83.0, 83.0),
        "joint_4": (2.0, 133.0),
        "joint_5": (-83.0, 83.0),
        "joint_6": (-38.0, 38.0),
        "joint_7": (-78.0, 78.0),
    },
}

SAFE_GRIPPER_LIMIT_DEG = (-65.0, 0.0)
MAX_FINGER_OPEN_M = 0.044


class ContractError(ValueError):
    """Raised when a scene, reset, or action violates the simulation contract."""


@dataclass(frozen=True)
class LimitViolation:
    side: str
    joint: str
    value_deg: float
    low_deg: float
    high_deg: float

    def message(self) -> str:
        return (
            f"{self.side}.{self.joint}={self.value_deg:.3f} deg is outside "
            f"{self.low_deg:.3f}..{self.high_deg:.3f} deg"
        )


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def normalize_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in SIDES:
        raise ContractError(f"Unknown side {side!r}; expected one of {SIDES}.")
    return normalized


def normalize_joint_name(joint: str) -> str:
    text = str(joint).strip().lower().replace("-", "_")
    if text.isdigit():
        text = f"joint_{int(text)}"
    if text.startswith("joint") and "_" not in text:
        text = text.replace("joint", "joint_", 1)
    if text not in JOINT_NAMES and text != "gripper":
        raise ContractError(f"Unknown joint {joint!r}; expected joint_1..joint_7 or gripper.")
    return text


def check_joint_target(side: str, joint: str, value_deg: float) -> LimitViolation | None:
    side = normalize_side(side)
    joint = normalize_joint_name(joint)
    if joint == "gripper":
        low, high = SAFE_GRIPPER_LIMIT_DEG
    else:
        low, high = SAFE_ARM_LIMITS_DEG[side][joint]
    value = float(value_deg)
    if value < low or value > high:
        return LimitViolation(side, joint, value, low, high)
    return None


def validate_joint_targets(side: str, targets_deg: dict[str, float]) -> None:
    violations = [
        violation
        for joint, value in targets_deg.items()
        if (violation := check_joint_target(side, joint, float(value))) is not None
    ]
    if violations:
        details = "\n  ".join(violation.message() for violation in violations)
        raise ContractError(f"Simulation limit contract violation:\n  {details}")


def clamp_joint_targets(side: str, targets_deg: dict[str, float]) -> dict[str, float]:
    side = normalize_side(side)
    clamped: dict[str, float] = {}
    for joint, value in targets_deg.items():
        joint_name = normalize_joint_name(joint)
        if joint_name == "gripper":
            low, high = SAFE_GRIPPER_LIMIT_DEG
        else:
            low, high = SAFE_ARM_LIMITS_DEG[side][joint_name]
        clamped[joint_name] = clamp(float(value), low, high)
    return clamped


def gripper_deg_to_sim_finger_m(target_deg: float) -> float:
    clipped = clamp(float(target_deg), *SAFE_GRIPPER_LIMIT_DEG)
    return (-clipped / 65.0) * MAX_FINGER_OPEN_M


def sim_finger_m_to_gripper_deg(finger_m: float) -> float:
    clipped = clamp(float(finger_m), 0.0, MAX_FINGER_OPEN_M)
    return -(clipped / MAX_FINGER_OPEN_M) * 65.0


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ContractError(f"Expected mapping at {resolved}, got {type(loaded).__name__}.")
    return loaded


def validate_workspace_pose(config: dict[str, Any], pose: list[float], *, label: str) -> None:
    bounds = config["scene"]["workspace_bounds_m"]
    names = ("x", "y", "z")
    for axis, value in zip(names, pose):
        low, high = bounds[axis]
        if float(value) < float(low) or float(value) > float(high):
            raise ContractError(
                f"{label} {axis}={float(value):.3f} m is outside workspace "
                f"{float(low):.3f}..{float(high):.3f} m"
            )


def validate_scene_config(config: dict[str, Any]) -> None:
    scene = config.get("scene", {})
    if scene.get("sim_only") is not True:
        raise ContractError("Scene config must set scene.sim_only: true.")
    active_arm = normalize_side(scene.get("active_arm", "right"))
    robot = config.get("robot", {})
    reset_pose = robot.get("reset_pose_deg", {})
    if active_arm not in reset_pose:
        raise ContractError(f"Missing reset pose for active arm {active_arm!r}.")
    if not scene.get("allow_out_of_contract_reset_pose", False):
        validate_joint_targets(active_arm, reset_pose[active_arm])

    objects = config.get("objects", [])
    if len(objects) != 4:
        raise ContractError(f"Expected exactly four objects, got {len(objects)}.")
    seen = set()
    for obj in objects:
        name = obj.get("name")
        if not name or name in seen:
            raise ContractError(f"Object names must be unique and non-empty: {name!r}")
        seen.add(name)
        pose = obj.get("spawn_pose_m")
        if not isinstance(pose, list) or len(pose) != 3:
            raise ContractError(f"{name} must define a 3-value spawn_pose_m.")
        validate_workspace_pose(config, pose, label=f"{name}.spawn_pose_m")


def object_names_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(obj["name"]) for obj in config.get("objects", []))


def limits_summary() -> dict[str, Any]:
    return {
        "safe_arm_limits_deg": SAFE_ARM_LIMITS_DEG,
        "safe_gripper_limit_deg": SAFE_GRIPPER_LIMIT_DEG,
        "max_finger_open_m": MAX_FINGER_OPEN_M,
    }
