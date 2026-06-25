#!/usr/bin/env python3
"""Collect a DENSE Isaac/placeholder-camera SmolVLA pick-and-lift dataset.

This is the corrected data path called for in
``docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md`` and the success-filtered
dataset audit. It fixes the three structural weaknesses of the old
``openarm_success_filtered_14000`` dataset:

  1. Dense rollouts, not 5 keyframes. Each episode records every control step of
     approach -> descend -> close -> lift -> hold (default 50 steps).
  2. RGB at every control step. The normal path uses the actual Isaac scene
     camera tensor; ``--camera-mode placeholder`` is available as a physics-only
     fallback for machines where RTX camera rendering is unavailable.
  3. Distinct observed state and commanded action per step. ``observation.state``
     is the measured joint state read from physics; ``action`` is the clamped IK
     joint target actually commanded that step.

It writes straight into a LeRobot dataset (so there is no giant image JSONL) and
also writes a per-episode metadata JSONL (object poses, measured rises, contact,
limit-clamp, and the dense numeric state/action trace) for auditing.

Only measured successful target-object lifts with no wrong-object lift, no
object-object collision, and no gripper/table collision are kept. Targets are
sampled with a configurable weight so the hard ``orange_ball`` class is
over-collected for better balance.

Simulation only. Never opens CAN, never moves the real robot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_scene import _isaac_paths, build_scene_cls, force_bind_robot_visual_material  # noqa: E402
from sim_contract import (  # noqa: E402
    CONFIG_DIR,
    JOINT_NAMES,
    REPO_ROOT,
    SAFE_ARM_LIMITS_DEG,
    gripper_deg_to_sim_finger_m,
    load_yaml_config,
    sim_finger_m_to_gripper_deg,
    validate_joint_targets,
    validate_scene_config,
)

CAMERA_KEY = "observation.images.camera1"
DEPTH_KEY = "observation.images.depth"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_NAMES = [*JOINT_NAMES, "gripper"]


def parse_target_quotas(raw: str | None, object_names: list[str], successes_per_target: int = 0) -> dict[str, int]:
    """Parse exact retained-success quotas keyed by target object.

    ``raw`` accepts either comma-separated counts aligned to ``object_names``
    (for example ``1,1,1,1``) or name/count pairs
    (``orange_ball=1,red_cube=1``). Missing names default to zero.
    """
    if raw is None or not str(raw).strip():
        if successes_per_target <= 0:
            return {}
        return {name: int(successes_per_target) for name in object_names}

    text = str(raw).strip()
    if "=" not in text:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
        if len(values) != len(object_names):
            raise ValueError(
                f"quota count list needs {len(object_names)} values aligned to {object_names}, got {values}"
            )
        return {name: max(0, value) for name, value in zip(object_names, values)}

    quotas = {name: 0 for name in object_names}
    valid = set(object_names)
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"quota entry {part!r} must be name=count")
        name, value = [piece.strip() for piece in part.split("=", 1)]
        if name not in valid:
            raise ValueError(f"unknown quota target {name!r}; expected one of {object_names}")
        quotas[name] = max(0, int(value))
    return quotas


def target_quotas_satisfied(kept_by_target: Counter, quotas: dict[str, int]) -> bool:
    if not quotas:
        return False
    return all(int(kept_by_target.get(name, 0)) >= int(quota) for name, quota in quotas.items())


def cap_gripper_close_deg(value_deg: float, close_cap_deg: float) -> float:
    """Cap gripper commands so they never move closer than ``close_cap_deg``.

    OpenArm convention in this repo is ``-65`` open and ``0`` fully closed, so a
    value greater than the cap (for example ``-5`` with cap ``-10``) is too close.
    """
    return min(float(value_deg), float(close_cap_deg))


def limit_action_step_deg(
    previous_deg,
    target_deg,
    *,
    max_step_deg: float,
    gripper_close_cap_deg: float,
):
    """Limit one 8D command update and cap gripper close.

    Returns ``(command, raw_max_delta, applied_max_delta, limited)`` as plain
    Python values so this can be unit-tested without Isaac.
    """
    prev = [float(v) for v in previous_deg]
    target = [float(v) for v in target_deg]
    if len(prev) != 8 or len(target) != 8:
        raise ValueError("previous_deg and target_deg must both have 8 values")
    target[-1] = cap_gripper_close_deg(target[-1], gripper_close_cap_deg)
    raw_deltas = [t - p for p, t in zip(prev, target, strict=True)]
    raw_max = max(abs(v) for v in raw_deltas)
    if max_step_deg <= 0:
        command = target
    else:
        step = float(max_step_deg)
        command = [p + max(-step, min(step, d)) for p, d in zip(prev, raw_deltas, strict=True)]
    command[-1] = cap_gripper_close_deg(command[-1], gripper_close_cap_deg)
    applied_max = max(abs(c - p) for p, c in zip(prev, command, strict=True))
    return command, raw_max, applied_max, bool(max_step_deg > 0 and raw_max > float(max_step_deg) + 1.0e-6)


def hold_pad_to_length(values: list, length: int) -> list:
    """Return exactly ``length`` entries, holding the final value if needed."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not values:
        raise ValueError("cannot pad an empty sequence")
    if len(values) >= length:
        return values[:length]
    return [*values, *([values[-1]] * (length - len(values)))]


def normalize_gripper_close_range_deg(values: list[float] | tuple[float, float] | None) -> tuple[float, float] | None:
    """Validate an optional inclusive gripper close target range in degrees."""
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--gripper-close-range-deg requires exactly two values: MIN MAX")
    lo, hi = float(values[0]), float(values[1])
    if lo > hi:
        raise ValueError("--gripper-close-range-deg MIN must be <= MAX")
    return lo, hi


def load_staged_init_csv(path: str | Path, *, expected_side: str = "left") -> list[list[float]]:
    """Load the three measured staged-init rows as 8D degree commands."""
    resolved = Path(path)
    if not resolved.exists():
        raise ValueError(f"staged-init CSV does not exist: {resolved}")
    columns = [f"{joint}.pos" for joint in JOINT_NAMES] + ["gripper.pos"]
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3:
        raise ValueError(f"{resolved} must contain exactly three staged-init rows, got {len(rows)}")
    stages: list[list[float]] = []
    for index, row in enumerate(rows, start=1):
        side = str(row.get("side", expected_side)).strip().lower()
        if side != expected_side:
            raise ValueError(f"{resolved} row {index} has side={side!r}; expected {expected_side!r}")
        missing = [name for name in columns if name not in row]
        if missing:
            raise ValueError(f"{resolved} row {index} missing columns: {missing}")
        values = [float(row[name]) for name in columns]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{resolved} row {index} contains non-finite staged-init values")
        targets = {joint: values[joint_index] for joint_index, joint in enumerate(JOINT_NAMES)}
        targets["gripper"] = values[-1]
        validate_joint_targets(expected_side, targets)
        stages.append(values)
    return stages


def interpolate_staged_init_commands(stages_deg: list[list[float]], *, max_step_deg: float) -> list[list[float]]:
    """Retimed stage1 -> stage2 -> stage3 commands obeying an 8D slew cap."""
    if len(stages_deg) != 3:
        raise ValueError("staged init requires exactly three stages")
    if max_step_deg <= 0:
        raise ValueError("max_step_deg must be positive for staged-init interpolation")
    stages = [[float(v) for v in stage] for stage in stages_deg]
    if any(len(stage) != 8 for stage in stages):
        raise ValueError("each staged-init command must contain 8 values")
    commands: list[list[float]] = [stages[0]]
    for start, end in zip(stages[:-1], stages[1:], strict=True):
        max_delta = max(abs(b - a) for a, b in zip(start, end, strict=True))
        n_sub = max(1, int(math.ceil(max_delta / float(max_step_deg) - 1.0e-9)))
        for step in range(1, n_sub + 1):
            frac = step / n_sub
            commands.append([a + (b - a) * frac for a, b in zip(start, end, strict=True)])
    return commands


def dense_phase_plan(args: argparse.Namespace) -> list[tuple[str, int]]:
    """Build the recorded command phases from parsed collector args."""
    phases: list[tuple[str, int]] = []
    if getattr(args, "record_zero_to_init", False):
        phases.append(("zero_to_init", int(args.zero_init_steps)))
    staged_init_commands = getattr(args, "_staged_init_commands", None)
    if staged_init_commands:
        phases.append(("staged_init", len(staged_init_commands)))
    phases.extend(
        [
            ("approach", int(args.approach_steps)),
            ("descend", int(args.descend_steps)),
            ("close", int(args.close_steps)),
            ("lift", int(args.lift_steps)),
            ("hold", int(args.hold_steps)),
        ]
    )
    return phases


def zero_to_init_command_deg(
    init_arm_deg,
    *,
    step_index: int,
    steps: int,
    zero_gripper_deg: float,
    init_gripper_deg: float,
) -> list[float]:
    """Return one 8D command for linear all-zero-arm -> configured-init setup."""
    if steps < 2:
        raise ValueError("zero-to-init recording needs at least 2 steps")
    if step_index < 0 or step_index >= steps:
        raise ValueError("step_index must be within the zero-to-init phase")
    arm = [float(v) for v in init_arm_deg]
    if len(arm) != 7:
        raise ValueError("init_arm_deg must contain 7 arm joint values")
    frac = float(step_index) / float(steps - 1)
    return [v * frac for v in arm] + [
        float(zero_gripper_deg) + (float(init_gripper_deg) - float(zero_gripper_deg)) * frac
    ]


class LocalNpzEpisodeDataset:
    """Small local dataset writer used when LeRobot is not installed."""

    def __init__(self, *, repo_id: str, root: Path, fps: int, robot_type: str, features: dict):
        self.repo_id = str(repo_id)
        self.root = Path(root)
        self.fps = int(fps)
        self.robot_type = str(robot_type)
        self.features = features
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[dict] = []
        self.num_episodes = 0
        self.num_frames = 0

    @classmethod
    def create(cls, *, repo_id: str, root: Path, fps: int, robot_type: str, features: dict, **_: object):
        return cls(repo_id=repo_id, root=Path(root), fps=fps, robot_type=robot_type, features=features)

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        if not self.frames:
            return
        import numpy as np  # noqa: PLC0415

        camera = np.stack([np.asarray(frame[CAMERA_KEY], dtype=np.uint8) for frame in self.frames], axis=0)
        depth = None
        if any(DEPTH_KEY in frame for frame in self.frames):
            if not all(DEPTH_KEY in frame for frame in self.frames):
                raise ValueError("depth must be present on every frame when enabled")
            depth = np.stack([np.asarray(frame[DEPTH_KEY], dtype=np.float32) for frame in self.frames], axis=0)
        state = np.stack([np.asarray(frame[STATE_KEY], dtype=np.float32) for frame in self.frames], axis=0)
        action = np.stack([np.asarray(frame[ACTION_KEY], dtype=np.float32) for frame in self.frames], axis=0)
        task = str(self.frames[0].get("task", ""))
        out = self.episodes_dir / f"episode_{self.num_episodes:06d}.npz"
        arrays = {
            "camera": camera,
            "state": state,
            "action": action,
            "task": np.asarray(task),
            "fps": np.asarray(self.fps, dtype=np.int32),
        }
        if depth is not None:
            arrays[DEPTH_KEY] = depth
        np.savez_compressed(out, **arrays)
        self.num_episodes += 1
        self.num_frames += int(camera.shape[0])
        self.frames.clear()

    def finalize(self) -> None:
        metadata = {
            "ok": True,
            "backend": "local_npz",
            "repo_id": self.repo_id,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "features": self.features,
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
            "format": "episodes/episode_000000.npz with arrays camera,state,action,task and optional depth",
            "depth_key": DEPTH_KEY if DEPTH_KEY in self.features else None,
        }
        (self.root / "meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- Better safety-check helpers (pure python; unit-tested without Isaac/GPU) ---
# These replace the two misleading manifest diagnostics:
#   * min_tcp_table_clearance_m: unconditioned on being over the table footprint, so it
#     flagged the TCP frame hanging low beside the robot base as if it were penetration.
#   * limit_exceeded: dominated by the unavoidable joint_4 zero-start clamp (reset is all
#     zeros but joint_4's safe floor is 2 deg), so it flagged ~100% of episodes.
# The in-loop collector uses torch tensors but mirrors this exact logic.

def finger_table_penetration(finger_z, over_table, table_top_z, margin_m):
    """Footprint-conditioned tabletop penetration from the actual finger body world z.

    finger_z: per-finger-body world z values (one per finger body).
    over_table: per-finger-body bools; True iff that body's xy is within the table footprint.
    Returns (min_clearance, penetrated). min_clearance = min(finger_z - table_top_z) over only
    the finger bodies that are over the table (``+inf`` if none are), so a finger dipping below
    the table plane while beside the table is NOT counted. penetrated = min_clearance < -margin_m.
    """
    clearances = [float(fz) - float(table_top_z) for fz, ot in zip(finger_z, over_table) if ot]
    if not clearances:
        return float("inf"), False
    min_clearance = min(clearances)
    return min_clearance, bool(min_clearance < -float(margin_m))


def object_pushed_down(obj_z, rest_z, margin_m):
    """True if any object is driven below its episode-start rest z by more than margin_m."""
    return any((float(rz) - float(oz)) > float(margin_m) for oz, rz in zip(obj_z, rest_z))


def refined_action_clip(jdes_deg, jclamped_deg, joint4_index, tol_deg, joint4_startup_tol_deg):
    """Genuine commanded-action clipping for one control step, ignoring the joint_4 zero-start clamp.

    jdes_deg / jclamped_deg: per-joint IK-desired vs safe-clamped angles (degrees) for one step.
    The joint_4 lower-bound correction (clip magnitude <= joint4_startup_tol_deg) is ignored because
    the all-zero reset starts joint_4 below its 2 deg safe floor and the first solves necessarily
    nudge it up. Returns (clipped, max_clip_deg) over the non-ignored joints.
    """
    max_clip = 0.0
    clipped = False
    for idx, (d, c) in enumerate(zip(jdes_deg, jclamped_deg)):
        mag = abs(float(d) - float(c))
        if idx == int(joint4_index) and mag <= float(joint4_startup_tol_deg):
            continue
        if mag > max_clip:
            max_clip = mag
        if mag > float(tol_deg):
            clipped = True
    return clipped, max_clip


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(CONFIG_DIR / "scene_openarm_dense_isaac_camera_v1.yaml"))
    p.add_argument("--dataset-root", default="synthetic_smolvla/datasets/openarm_dense_isaac_camera_v1")
    p.add_argument("--repo-id", default="local/openarm_dense_isaac_camera_v1")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--rounds", type=int, default=1, help="num_envs * rounds = source episodes")
    p.add_argument("--seed", type=int, default=12000)
    p.add_argument("--randomize", action="store_true", default=True)
    p.add_argument("--no-randomize", dest="randomize", action="store_false")
    # Dense control schedule (control steps per phase). Sum is the episode length.
    p.add_argument("--approach-steps", type=int, default=14)
    p.add_argument("--descend-steps", type=int, default=12)
    p.add_argument("--close-steps", type=int, default=8)
    p.add_argument("--lift-steps", type=int, default=12)
    p.add_argument("--hold-steps", type=int, default=4)
    p.add_argument("--substeps", type=int, default=12, help="physics steps per control step (match eval)")
    p.add_argument("--settle-steps", type=int, default=40)
    p.add_argument(
        "--record-zero-to-init",
        action="store_true",
        help="record a zero-arm setup phase before the normal grasp phases",
    )
    p.add_argument(
        "--staged-init-csv",
        default=None,
        help="optional measured three-row staged-init CSV to record before the grasp phases",
    )
    p.add_argument("--staged-init-name", default=None, help="label for the measured staged-init trajectory")
    p.add_argument(
        "--target-episode-commands",
        type=int,
        default=0,
        help="if positive, extend/reduce hold so the recorded episode has exactly this many commands",
    )
    p.add_argument(
        "--zero-init-steps",
        type=int,
        default=20,
        help="recorded commands for the zero pose -> configured init pose phase",
    )
    p.add_argument(
        "--zero-start-gripper-deg",
        type=float,
        default=0.0,
        help="gripper command at the first zero-start frame when --record-zero-to-init is enabled",
    )
    p.add_argument(
        "--init-gripper-deg",
        type=float,
        default=-50.0,
        help="gripper command at the configured init pose when --record-zero-to-init is enabled",
    )
    p.add_argument(
        "--prepose-to-ready",
        action="store_true",
        help="run a non-recorded IK warmup to an above-object ready pose, then record approach as zero->ready joint interpolation",
    )
    p.add_argument(
        "--prepose-warmup-steps",
        type=int,
        default=120,
        help="non-recorded IK iterations used by --prepose-to-ready before resetting and recording",
    )
    p.add_argument("--above-offset-m", type=float, default=0.12)
    p.add_argument(
        "--grasp-z-offset-m",
        type=float,
        default=0.0,
        help="offset added to the target object's grasp z before descend/close/lift waypoints",
    )
    p.add_argument("--lift-offset-m", type=float, default=0.05)
    p.add_argument("--lift-threshold-m", type=float, default=0.04)
    p.add_argument("--grasp-close-deg", type=float, default=0.0, help="gripper target at full close (deg, -65..0)")
    p.add_argument(
        "--open-gripper-deg",
        type=float,
        default=-65.0,
        help="gripper command used for approach/descend and episode initialization",
    )
    p.add_argument(
        "--max-gripper-close-deg",
        type=float,
        default=0.0,
        help="upper gripper command cap; use -3 to prevent fully closed gripper commands",
    )
    p.add_argument(
        "--gripper-close-range-deg",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="sample each episode's close target from [MIN, MAX] deg, capped by --max-gripper-close-deg",
    )
    p.add_argument("--contact-eps-m", type=float, default=0.02, help="tcp-object distance counted as contact")
    p.add_argument("--object-collision-margin-m", type=float, default=0.002)
    p.add_argument("--table-collision-margin-m", type=float, default=0.005)
    p.add_argument(
        "--object-sweep-threshold-m",
        type=float,
        default=0.025,
        help="reject episodes where any object slides this far on the table before lift",
    )
    # Per-episode object spawn jitter (defaults match the small-jitter contract).
    p.add_argument("--jitter-x-m", type=float, default=0.005,
                   help="uniform +/- x jitter applied to each object spawn (m)")
    p.add_argument("--jitter-y-m", type=float, default=0.003,
                   help="uniform +/- y jitter applied to each object spawn (m)")
    # Better safety checks (replace the misleading tcp-clearance / limit_exceeded diagnostics).
    p.add_argument("--finger-table-margin-m", type=float, default=0.003,
                   help="reject if a finger body penetrates below the table top by more than this (m), over the footprint")
    p.add_argument("--object-pushdown-margin-m", type=float, default=0.005,
                   help="reject if any object is driven below its rest z by more than this (m)")
    p.add_argument("--action-clip-tol-deg", type=float, default=1.0,
                   help="genuine action-clip threshold (deg) for the refined unsafe-clip reject flag")
    p.add_argument("--joint4-startup-tol-deg", type=float, default=3.0,
                   help="ignore joint_4 lower-bound clamps up to this magnitude (the unavoidable zero-start correction)")
    p.add_argument(
        "--max-arm-action-step-deg",
        type=float,
        default=0.0,
        help="deprecated compatibility alias for --max-action-step-deg when that flag is omitted",
    )
    p.add_argument(
        "--max-action-step-deg",
        type=float,
        default=None,
        help="optional 8D upsample/slew limit for recorded commands; 1 at 10 Hz means <=10 deg/sec",
    )
    p.add_argument(
        "--early-stop-on-lift",
        action="store_true",
        help="freeze each env's active command once target rise reaches --lift-threshold-m, then hold-pad through the horizon",
    )
    # Target sampling weights, aligned to config object order [orange,red,green,blue].
    p.add_argument("--target-weights", default="2.5,1,1,1", help="comma weights for object sampling")
    p.add_argument("--drop-limit-exceeded", action="store_true", help="reject episodes that hit the limit clamp")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument(
        "--dataset-backend",
        choices=("auto", "local_npz", "lerobot"),
        default="auto",
        help="dataset writer backend; local_npz writes episodes/*.npz directly",
    )
    p.add_argument(
        "--require-depth",
        action="store_true",
        help="require and persist Isaac distance_to_image_plane depth frames",
    )
    p.add_argument(
        "--camera-mode",
        choices=("isaac", "placeholder"),
        default="isaac",
        help="use real Isaac camera frames, or physics-only deterministic placeholder frames if RTX rendering is unavailable",
    )
    p.add_argument(
        "--placeholder-view",
        choices=("isaac_viewport", "robot_front"),
        default="isaac_viewport",
        help="visual style for --camera-mode placeholder; physics still comes from Isaac Lab",
    )
    p.add_argument("--experience", default="", help="optional Isaac/Kit experience file override")
    p.add_argument("--rendering-mode", default=None, choices=("performance", "balanced", "quality"))
    p.add_argument("--kit-args", default="", help="raw Omniverse Kit args, quoted as one string")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-keep", type=int, default=0, help="optional cap on kept episodes (0 = no cap)")
    p.add_argument(
        "--successes-per-target",
        type=int,
        default=0,
        help="optional exact retained-success quota for every target; 0 disables per-target quotas",
    )
    p.add_argument(
        "--target-quotas",
        default=None,
        help=(
            "optional retained-success quotas, either counts aligned to config objects "
            "(e.g. '1,1,1,1') or name=count pairs; overrides --successes-per-target"
        ),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--manifest", default="synthetic_smolvla/reports/dense_isaac_camera_v1_manifest.jsonl")
    p.add_argument("--report", default="synthetic_smolvla/reports/dense_isaac_camera_v1_collect.md")
    p.add_argument("--sample-frame-dir", default="synthetic_smolvla/reports/dense_isaac_camera_v1_samples",
                   help="dump a few PNG frames of episode 0 for visual inspection")
    return p


def _resolve(path: str) -> Path:
    rp = Path(path)
    return rp if rp.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.record_zero_to_init and args.prepose_to_ready:
        raise SystemExit("--record-zero-to-init cannot be combined with --prepose-to-ready.")
    if args.staged_init_csv and args.record_zero_to_init:
        raise SystemExit("--staged-init-csv cannot be combined with --record-zero-to-init.")
    if args.staged_init_csv and args.prepose_to_ready:
        raise SystemExit("--staged-init-csv cannot be combined with --prepose-to-ready.")
    if args.record_zero_to_init and args.zero_init_steps < 2:
        raise SystemExit("--zero-init-steps must be >= 2 when --record-zero-to-init is enabled.")
    if args.target_episode_commands < 0:
        raise SystemExit("--target-episode-commands must be >= 0.")
    if args.prepose_warmup_steps < 0:
        raise SystemExit("--prepose-warmup-steps must be >= 0.")
    if args.max_arm_action_step_deg < 0:
        raise SystemExit("--max-arm-action-step-deg must be >= 0.")
    max_action_step_deg = (
        float(args.max_action_step_deg)
        if args.max_action_step_deg is not None
        else float(args.max_arm_action_step_deg)
    )
    if max_action_step_deg < 0:
        raise SystemExit("--max-action-step-deg must be >= 0.")
    try:
        gripper_close_range_deg = normalize_gripper_close_range_deg(args.gripper_close_range_deg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = load_yaml_config(args.config)
    validate_scene_config(config)

    side = config["scene"].get("active_arm", "right")
    staged_init_stages: list[list[float]] | None = None
    staged_init_commands: list[list[float]] | None = None
    if args.staged_init_csv:
        if max_action_step_deg <= 0.0:
            raise SystemExit("--staged-init-csv requires --max-action-step-deg > 0.")
        try:
            staged_init_stages = load_staged_init_csv(args.staged_init_csv, expected_side=side)
            staged_init_commands = interpolate_staged_init_commands(
                staged_init_stages,
                max_step_deg=max_action_step_deg,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        args._staged_init_commands = staged_init_commands
        setup_commands = len(staged_init_commands)
    else:
        setup_commands = int(args.zero_init_steps) if args.record_zero_to_init else 0
    if args.target_episode_commands > 0:
        fixed_without_hold = (
            setup_commands
            + int(args.approach_steps)
            + int(args.descend_steps)
            + int(args.close_steps)
            + int(args.lift_steps)
        )
        dynamic_hold = int(args.target_episode_commands) - fixed_without_hold
        if dynamic_hold < 0:
            raise SystemExit(
                f"--target-episode-commands={args.target_episode_commands} is too short for "
                f"{fixed_without_hold} non-hold commands."
            )
        args.hold_steps = dynamic_hold
    objs = config["objects"]
    obj_names = [o["name"] for o in objs]
    instruction_for = {o["name"]: o["instruction"] for o in objs}
    spawn_for = {o["name"]: [float(v) for v in o["spawn_pose_m"]] for o in objs}
    bounds = config["scene"]["workspace_bounds_m"]
    layout_info = config["scene"].get("layout_info_cm", {})
    height_sweep_info = config["scene"].get("height_sweep", {})
    table_cfg = config["scene"]["table"]
    table_size = [float(v) for v in table_cfg["size_m"]]
    table_pose = [float(v) for v in table_cfg["pose_m"]]
    table_top_z = table_pose[2] + table_size[2] / 2.0
    table_x = (table_pose[0] - table_size[0] / 2.0, table_pose[0] + table_size[0] / 2.0)
    table_y = (table_pose[1] - table_size[1] / 2.0, table_pose[1] + table_size[1] / 2.0)
    n_obj = len(obj_names)
    object_contact_radii = []
    for obj in objs:
        if obj["shape"] == "sphere":
            object_contact_radii.append(float(obj["radius_m"]))
        else:
            sx, sy, sz = [float(v) for v in obj["size_m"]]
            object_contact_radii.append(math.sqrt(sx * sx + sy * sy + sz * sz) / 2.0)

    res = config["scene"]["camera"]["resolution"]
    if int(res[0]) != int(res[1]):
        raise SystemExit(f"Dense camera dataset expects a square camera resolution, got {res}.")
    image_size = int(res[0])
    camera_data_types = list(config["scene"].get("camera", {}).get("data_types", ["rgb"]))
    config_requests_depth = "distance_to_image_plane" in camera_data_types
    if args.require_depth and args.camera_mode != "isaac":
        raise SystemExit("--require-depth requires --camera-mode isaac.")
    if args.require_depth and not config_requests_depth:
        raise SystemExit(
            "--require-depth requires scene.camera.data_types to include distance_to_image_plane."
        )

    weights = [float(w) for w in args.target_weights.split(",")]
    if len(weights) != n_obj:
        raise SystemExit(f"--target-weights needs {n_obj} values aligned to {obj_names}, got {weights}.")
    try:
        target_quotas = parse_target_quotas(args.target_quotas, obj_names, args.successes_per_target)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if target_quotas and args.max_keep:
        quota_total = sum(target_quotas.values())
        if args.max_keep < quota_total:
            raise SystemExit(f"--max-keep={args.max_keep} is smaller than target quota total {quota_total}.")

    dataset_root = _resolve(args.dataset_root)
    manifest_path = _resolve(args.manifest)
    report_path = _resolve(args.report)
    sample_dir = _resolve(args.sample_frame_dir) if args.sample_frame_dir else None

    phase_plan = dense_phase_plan(args)
    episode_len = sum(n for _, n in phase_plan)
    if gripper_close_range_deg is None:
        effective_close_deg = min(float(args.grasp_close_deg), float(args.max_gripper_close_deg))
    else:
        effective_close_deg = min(float(gripper_close_range_deg[1]), float(args.max_gripper_close_deg))
    open_gripper_arg = float(args.init_gripper_deg) if args.record_zero_to_init else float(args.open_gripper_deg)
    open_gripper_deg = max(-65.0, cap_gripper_close_deg(open_gripper_arg, effective_close_deg))
    if effective_close_deg != float(args.grasp_close_deg):
        print(
            f"[dense] capping --grasp-close-deg {args.grasp_close_deg:.3f} to "
            f"{effective_close_deg:.3f}",
            file=sys.stderr,
            flush=True,
        )
    print(f"[dense] episode_len={episode_len} control steps, image={image_size}px, "
          f"keep-success-only, drop_limit={args.drop_limit_exceeded}, "
          f"gripper_close={effective_close_deg:.3f} deg, "
          f"gripper_open={open_gripper_deg:.3f} deg, "
          f"prepose_to_ready={args.prepose_to_ready}, "
          f"prepose_warmup_steps={args.prepose_warmup_steps}, "
          f"record_zero_to_init={args.record_zero_to_init}, "
          f"zero_init_steps={args.zero_init_steps if args.record_zero_to_init else 0}, "
          f"staged_init={bool(staged_init_commands)}, "
          f"staged_init_name={args.staged_init_name or 'n/a'}, "
          f"staged_init_steps={len(staged_init_commands) if staged_init_commands else 0}, "
          f"target_episode_commands={args.target_episode_commands or 'disabled'}, "
          f"max_action_step_deg={max_action_step_deg:.3f}, "
          f"early_stop_on_lift={args.early_stop_on_lift}, "
          f"gripper_close_range_deg={gripper_close_range_deg or 'disabled'}, "
          f"target_quotas={target_quotas or 'disabled'}, "
          f"camera_data_types={camera_data_types}, "
          f"require_depth={args.require_depth}, "
          f"dataset_backend={args.dataset_backend}",
          file=sys.stderr, flush=True)

    _isaac_paths()
    from isaaclab.app import AppLauncher

    use_isaac_camera = args.camera_mode == "isaac"
    print(
        f"[dense] launching Isaac (num_envs={args.num_envs}, camera_mode={args.camera_mode})",
        file=sys.stderr,
        flush=True,
    )
    app_launcher = AppLauncher(
        headless=True,
        enable_cameras=use_isaac_camera,
        experience=args.experience,
        rendering_mode=args.rendering_mode,
        kit_args=args.kit_args,
    )
    simulation_app = app_launcher.app

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415
    from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: PLC0415
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: PLC0415

    capture_depth = bool(use_isaac_camera and config_requests_depth)
    scene_cls = build_scene_cls(
        config,
        sim_utils=sim_utils,
        AssetBaseCfg=AssetBaseCfg,
        RigidObjectCfg=RigidObjectCfg,
        CameraCfg=CameraCfg,
        InteractiveSceneCfg=InteractiveSceneCfg,
        configclass=configclass,
        include_camera=use_isaac_camera,
    )

    N = args.num_envs
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    cam_cfg = config["scene"]["camera"]
    if use_isaac_camera:
        sim.set_camera_view(eye=cam_cfg["eye_m"], target=cam_cfg["target_m"])
    scene = InteractiveScene(scene_cls(num_envs=N, env_spacing=3.0))
    sim.reset()
    scene.reset()
    bound_robot_geoms = force_bind_robot_visual_material(config)
    if bound_robot_geoms:
        print(
            f"[dense] force-bound clean robot material to {bound_robot_geoms} geom prims",
            file=sys.stderr,
            flush=True,
        )
    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    camera = scene["camera"] if use_isaac_camera else None

    if use_isaac_camera:
        # Aim every env's camera from eye_m at target_m. build_scene_cls bakes a fixed
        # offset rotation that only points correctly for its original eye position, so
        # we override the sensor world pose here to frame the workspace for any eye.
        cam_eye = torch.tensor([float(v) for v in cam_cfg["eye_m"]], device=robot.device, dtype=torch.float32)
        cam_tgt = torch.tensor([float(v) for v in cam_cfg["target_m"]], device=robot.device, dtype=torch.float32)
        cam_eyes = cam_eye.unsqueeze(0).repeat(N, 1) + scene.env_origins
        cam_targets = cam_tgt.unsqueeze(0).repeat(N, 1) + scene.env_origins
        camera.set_world_poses_from_view(cam_eyes, cam_targets)

    arm_ids, _ = robot.find_joints([f"openarm_{side}_joint{i}" for i in range(1, 8)], preserve_order=True)
    finger_ids, _ = robot.find_joints(f"openarm_{side}_finger_joint.*")
    inactive_side = "left" if side == "right" else "right"
    inactive_arm_ids, _ = robot.find_joints(
        [f"openarm_{inactive_side}_joint{i}" for i in range(1, 8)],
        preserve_order=True,
    )
    inactive_finger_ids, _ = robot.find_joints(f"openarm_{inactive_side}_finger_joint.*")
    tcp_idx = robot.find_bodies(f"openarm_{side}_ee_tcp")[0][0]
    ee_jacobi_idx = tcp_idx - 1 if robot.is_fixed_base else tcp_idx
    # Finger body ids for the footprint-conditioned tabletop penetration check.
    finger_body_ids, finger_body_names = robot.find_bodies(f"openarm_{side}_.*finger")
    if not finger_body_ids:
        finger_body_ids, finger_body_names = robot.find_bodies(f"openarm_{side}_hand")
    gripper_table_body_ids = list(finger_body_ids)
    gripper_table_body_names = list(finger_body_names)
    hand_body_ids, hand_body_names = robot.find_bodies(f"openarm_{side}_hand")
    for body_id, body_name in zip(hand_body_ids, hand_body_names, strict=False):
        if body_id not in gripper_table_body_ids:
            gripper_table_body_ids.append(body_id)
            gripper_table_body_names.append(body_name)
    print(f"[dense] finger bodies for penetration check: {finger_body_names}", file=sys.stderr, flush=True)
    print(f"[dense] gripper bodies for table check: {gripper_table_body_names}", file=sys.stderr, flush=True)
    joint4_idx = JOINT_NAMES.index("joint_4")
    ent = SceneEntityCfg("robot", joint_names=[f"openarm_{side}_joint{i}" for i in range(1, 8)],
                         body_names=[f"openarm_{side}_ee_tcp"])
    ent.resolve(scene)

    device = robot.device
    arm_lo = torch.tensor([math.radians(SAFE_ARM_LIMITS_DEG[side][j][0]) for j in JOINT_NAMES], device=device)
    arm_hi = torch.tensor([math.radians(SAFE_ARM_LIMITS_DEG[side][j][1]) for j in JOINT_NAMES], device=device)
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(N, 1)

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
        num_envs=N, device=device,
    )

    open_m = gripper_deg_to_sim_finger_m(open_gripper_deg)
    staged_init_tensor = (
        torch.tensor(staged_init_commands, device=device, dtype=torch.float32)
        if staged_init_commands is not None
        else None
    )
    object_radius_t = torch.tensor(object_contact_radii, device=device, dtype=torch.float32)
    object_colors_u8 = torch.tensor(
        [[int(max(0, min(255, round(float(c) * 255.0)))) for c in obj.get("color_rgb", [0.8, 0.8, 0.8])] for obj in objs],
        device=device,
        dtype=torch.uint8,
    )

    def lock_inactive_arm() -> None:
        if inactive_arm_ids:
            robot.set_joint_position_target(robot.data.default_joint_pos[:, inactive_arm_ids], joint_ids=inactive_arm_ids)
        if inactive_finger_ids:
            reset_gripper = float(config["robot"]["reset_pose_deg"][inactive_side].get("gripper", -65.0))
            capped_gripper = min(reset_gripper, float(args.max_gripper_close_deg))
            finger_m = gripper_deg_to_sim_finger_m(capped_gripper)
            target = torch.full((N, len(inactive_finger_ids)), float(finger_m), device=device)
            robot.set_joint_position_target(target, joint_ids=inactive_finger_ids)

    def step_phys(n: int, render_last: bool = True) -> None:
        for k in range(n):
            lock_inactive_arm()
            scene.write_data_to_sim()
            sim.step(render=(use_isaac_camera and render_last and k == n - 1))
            scene.update(sim_dt)

    def set_gripper(finger_m: float) -> None:
        tgt = torch.full((N, len(finger_ids)), float(finger_m), device=device)
        robot.set_joint_position_target(tgt, joint_ids=finger_ids)

    def set_gripper_deg_targets(gripper_deg: torch.Tensor, *, cap_close: bool = True) -> None:
        values = [
            gripper_deg_to_sim_finger_m(
                cap_gripper_close_deg(float(v), effective_close_deg) if cap_close else float(v)
            )
            for v in gripper_deg.detach().cpu().tolist()
        ]
        target = torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, len(finger_ids))
        robot.set_joint_position_target(target, joint_ids=finger_ids)

    def obj_pos_w() -> torch.Tensor:  # [N, n_obj, 3]
        return torch.stack([scene[name].data.root_pos_w for name in obj_names], dim=1)

    def object_collision_state() -> tuple[torch.Tensor, torch.Tensor]:
        pos = obj_pos_w()
        flags = torch.zeros(N, dtype=torch.bool, device=device)
        min_dist = torch.full((N,), float("inf"), device=device)
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                dist = torch.linalg.norm(pos[:, i] - pos[:, j], dim=-1)
                threshold = object_radius_t[i] + object_radius_t[j] + float(args.object_collision_margin_m)
                flags |= dist < threshold
                min_dist = torch.minimum(min_dist, dist)
        return flags, min_dist

    def gripper_table_collision_state() -> tuple[torch.Tensor, torch.Tensor]:
        gb = robot.data.body_pose_w[:, gripper_table_body_ids, 0:3]  # [N, G, 3]
        over_table = (
            (gb[:, :, 0] >= table_x[0])
            & (gb[:, :, 0] <= table_x[1])
            & (gb[:, :, 1] >= table_y[0])
            & (gb[:, :, 1] <= table_y[1])
        )
        clearance = gb[:, :, 2] - table_top_z
        clearance_over = torch.where(over_table, clearance, torch.full_like(clearance, float("inf")))
        min_clear = clearance_over.min(dim=1).values
        return min_clear < float(args.table_collision_margin_m), min_clear

    def finger_table_state() -> tuple[torch.Tensor, torch.Tensor]:
        """Footprint-conditioned tabletop penetration from the actual finger body world z.

        Mirrors ``finger_table_penetration``: clearance is min(finger_z - table_top_z) over only
        the finger bodies whose xy is within the table footprint (``+inf`` if none), so a finger
        dipping below the table plane while beside the table is not counted as penetration.
        """
        fb = robot.data.body_pose_w[:, finger_body_ids, 0:3]  # [N, F, 3]
        fx, fy, fz = fb[:, :, 0], fb[:, :, 1], fb[:, :, 2]
        over = (fx >= table_x[0]) & (fx <= table_x[1]) & (fy >= table_y[0]) & (fy <= table_y[1])
        clearance = fz - table_top_z  # [N, F]
        clearance_over = torch.where(over, clearance, torch.full_like(clearance, float("inf")))
        min_clear = clearance_over.min(dim=1).values  # [N]
        penetrated = min_clear < -float(args.finger_table_margin_m)
        return penetrated, min_clear

    def read_rgb() -> np.ndarray:
        """Current camera tensor for all envs as uint8 [N,H,W,3]."""
        if not use_isaac_camera:
            image = np.zeros((N, image_size, image_size, 3), dtype=np.uint8)
            image[:, :, :, :] = np.array([37, 42, 46], dtype=np.uint8)
            table_top_color = np.array([176, 157, 128], dtype=np.uint8)
            table_front_color = np.array([126, 104, 80], dtype=np.uint8)
            table_edge_color = np.array([82, 64, 46], dtype=np.uint8)
            tape_color = np.array([205, 193, 170], dtype=np.uint8)
            arm_color = np.array([230, 230, 224], dtype=np.uint8)
            gripper_color = np.array([18, 19, 21], dtype=np.uint8)
            pos = obj_pos_w().detach().cpu().numpy()
            colors = object_colors_u8.detach().cpu().numpy()
            tcp = robot.data.body_pose_w[:, tcp_idx, 0:3].detach().cpu().numpy()
            root = robot.data.root_pose_w[:, 0:3].detach().cpu().numpy()
            if args.placeholder_view == "isaac_viewport":
                sky_h = int(image_size * 0.34)
                floor_color = np.array([174, 176, 176], dtype=np.uint8)
                grid_color = np.array([134, 137, 137], dtype=np.uint8)
                table_color = np.array([196, 198, 195], dtype=np.uint8)
                table_edge = np.array([112, 115, 116], dtype=np.uint8)
                arm_color = np.array([222, 225, 225], dtype=np.uint8)
                joint_color = np.array([60, 64, 68], dtype=np.uint8)
                gripper_color = np.array([18, 20, 22], dtype=np.uint8)
                focus_x = float(table_x[0] + 0.34)
                y_span = max(0.15, float(table_y[1] - table_y[0]))
                yy, xx = np.ogrid[:image_size, :image_size]

                def clampi(value: float, low: int = 0, high: int | None = None) -> int:
                    upper = image_size - 1 if high is None else high
                    return max(low, min(upper, int(round(value))))

                def project(point: np.ndarray | list[float] | tuple[float, float, float]) -> tuple[int, int, float]:
                    px_world = float(point[0])
                    py_world = float(point[1])
                    pz_world = float(point[2])
                    relx = px_world - focus_x
                    rely = py_world
                    relz = pz_world - table_top_z
                    px = image_size * 0.52 + (rely / y_span) * image_size * 0.76 - relx * image_size * 0.14
                    py = image_size * 0.60 + relx * image_size * 0.24 - relz * image_size * 1.30
                    scale = max(0.45, min(1.35, 1.10 - relx * 0.18))
                    return clampi(px), clampi(py), scale

                def draw_circle(frame: np.ndarray, cx: int, cy: int, radius: int, color: np.ndarray) -> None:
                    mask = (xx - int(cx)) ** 2 + (yy - int(cy)) ** 2 <= int(radius) ** 2
                    frame[mask] = color

                def draw_line(frame: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: np.ndarray, width: int = 2) -> None:
                    x0, y0_ = p0
                    x1_, y1_ = p1
                    steps = max(abs(x1_ - x0), abs(y1_ - y0_), 1)
                    for step in range(steps + 1):
                        t = step / steps
                        x = clampi(x0 + (x1_ - x0) * t)
                        y = clampi(y0_ + (y1_ - y0_) * t)
                        r = max(1, width)
                        frame[max(0, y - r) : min(image_size, y + r + 1), max(0, x - r) : min(image_size, x + r + 1)] = color

                def fill_convex(frame: np.ndarray, points: list[tuple[int, int]], color: np.ndarray) -> None:
                    pts = np.asarray(points, dtype=np.float32)
                    if len(pts) < 3:
                        return
                    orient = 0.0
                    for idx in range(len(pts)):
                        x0, y0_ = pts[idx]
                        x1_, y1_ = pts[(idx + 1) % len(pts)]
                        orient += x0 * y1_ - x1_ * y0_
                    mask = np.ones((image_size, image_size), dtype=bool)
                    for idx in range(len(pts)):
                        x0, y0_ = pts[idx]
                        x1_, y1_ = pts[(idx + 1) % len(pts)]
                        cross = (xx - x0) * (y1_ - y0_) - (yy - y0_) * (x1_ - x0)
                        mask &= cross >= 0 if orient < 0 else cross <= 0
                    frame[mask] = color

                table_corners = [
                    project([table_x[0], table_y[0], table_top_z])[:2],
                    project([table_x[1], table_y[0], table_top_z])[:2],
                    project([table_x[1], table_y[1], table_top_z])[:2],
                    project([table_x[0], table_y[1], table_top_z])[:2],
                ]
                for e in range(N):
                    frame = image[e]
                    for row in range(sky_h):
                        shade = row / max(1, sky_h - 1)
                        frame[row, :, :] = np.array(
                            [188 + 18 * shade, 201 + 14 * shade, 214 + 8 * shade],
                            dtype=np.uint8,
                        )
                    frame[sky_h:, :, :] = floor_color
                    horizon = max(0, sky_h - 1)
                    frame[horizon : horizon + 2, :, :] = np.array([154, 160, 164], dtype=np.uint8)
                    for offset in range(-image_size, image_size * 2, max(8, image_size // 16)):
                        draw_line(frame, (offset, sky_h), (offset + image_size, image_size - 1), grid_color, 1)
                        draw_line(frame, (offset, image_size - 1), (offset + image_size, sky_h), grid_color, 1)

                    fill_convex(frame, table_corners, table_color)
                    for idx in range(len(table_corners)):
                        draw_line(frame, table_corners[idx], table_corners[(idx + 1) % len(table_corners)], table_edge, 1)

                    base_px = project(root[e])[:2]
                    draw_circle(frame, base_px[0], base_px[1] + max(5, image_size // 28), max(10, image_size // 18), np.array([74, 78, 82], dtype=np.uint8))
                    shoulder = (base_px[0], clampi(base_px[1] - image_size * 0.16))
                    elbow = (clampi((shoulder[0] + project(tcp[e])[:2][0]) / 2 - image_size * 0.06), clampi((shoulder[1] + project(tcp[e])[:2][1]) / 2 - image_size * 0.08))
                    tx, ty, scale = project(tcp[e])
                    draw_line(frame, base_px, shoulder, arm_color, max(1, image_size // 52))
                    draw_line(frame, shoulder, elbow, arm_color, max(1, image_size // 56))
                    draw_line(frame, elbow, (tx, ty), arm_color, max(1, image_size // 64))
                    for joint in (base_px, shoulder, elbow, (tx, ty)):
                        draw_circle(frame, joint[0], joint[1], max(2, image_size // 62), joint_color)

                    order = sorted(range(n_obj), key=lambda idx: pos[e, idx, 0], reverse=True)
                    for i in order:
                        px, py, scale = project(pos[e, i])
                        radius_px = max(3, int((8 if objs[i]["shape"] == "sphere" else 7) * scale))
                        color = colors[i]
                        draw_circle(frame, px + radius_px // 3, py + radius_px, max(2, radius_px), np.array([113, 114, 111], dtype=np.uint8))
                        if objs[i]["shape"] == "sphere":
                            draw_circle(frame, px, py, radius_px, color)
                            draw_circle(frame, px - radius_px // 3, py - radius_px // 4, max(1, radius_px // 3), np.minimum(color.astype(np.int16) + 70, 255).astype(np.uint8))
                        else:
                            darker = (color.astype(np.float32) * 0.55).astype(np.uint8)
                            side_color = (color.astype(np.float32) * 0.72).astype(np.uint8)
                            frame[max(0, py - radius_px) : min(image_size, py + radius_px + 1), max(0, px - radius_px) : min(image_size, px + radius_px + 1)] = color
                            frame[max(0, py - radius_px + 2) : min(image_size, py + radius_px + 3), max(0, px + radius_px) : min(image_size, px + radius_px + 4)] = darker
                            frame[max(0, py + radius_px) : min(image_size, py + radius_px + 4), max(0, px - radius_px + 2) : min(image_size, px + radius_px + 4)] = side_color

                    jaw = max(3, int(7 * scale))
                    draw_line(frame, (tx - jaw, ty + jaw), (tx - 2, ty + 1), gripper_color, width=1)
                    draw_line(frame, (tx + jaw, ty + jaw), (tx + 2, ty + 1), gripper_color, width=1)
                return image

            y_span = max(0.15, float(table_y[1] - table_y[0]))
            visible_y0, visible_y1 = -0.34, 0.34
            surface_y = int(image_size * 0.58)
            front_y = int(image_size * 0.72)
            horizon_y = max(4, int(image_size * 0.02))
            yy, xx = np.ogrid[:image_size, :image_size]

            def clampi(value: float, low: int = 0, high: int | None = None) -> int:
                upper = image_size - 1 if high is None else high
                return max(low, min(upper, int(round(value))))

            def world_to_front_px(point: np.ndarray) -> tuple[int, int, float]:
                depth = max(0.05, float(point[0]) - float(table_x[0]))
                scale = 0.96 / (1.0 + 0.55 * depth)
                px = image_size * 0.50 + ((float(point[1]) - 0.0) / y_span) * image_size * scale
                py = surface_y - (float(point[2]) - table_top_z) * image_size * 7.0 - depth * image_size * 0.12
                return clampi(px), clampi(py), scale

            def draw_rect(frame: np.ndarray, x0: int, y0_: int, x1_: int, y1_: int, color: np.ndarray) -> None:
                xa, xb = sorted((clampi(x0), clampi(x1_)))
                ya, yb = sorted((clampi(y0_), clampi(y1_)))
                frame[ya : yb + 1, xa : xb + 1] = color

            def draw_circle(frame: np.ndarray, cx: int, cy: int, radius: int, color: np.ndarray) -> None:
                mask = (xx - int(cx)) ** 2 + (yy - int(cy)) ** 2 <= int(radius) ** 2
                frame[mask] = color

            def draw_line(frame: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: np.ndarray, width: int = 2) -> None:
                x0, y0_ = p0
                x1_, y1_ = p1
                steps = max(abs(x1_ - x0), abs(y1_ - y0_), 1)
                for step in range(steps + 1):
                    t = step / steps
                    x = clampi(x0 + (x1_ - x0) * t)
                    y = clampi(y0_ + (y1_ - y0_) * t)
                    r = max(1, width)
                    frame[max(0, y - r) : min(image_size, y + r + 1), max(0, x - r) : min(image_size, x + r + 1)] = color

            for e in range(N):
                frame = image[e]
                frame[horizon_y:front_y, :] = table_top_color
                for row in range(horizon_y, front_y, 3):
                    shade = int(7.0 * math.sin(row * 0.13) + 4.0 * math.sin(row * 0.31))
                    frame[row : row + 1, :, :] = np.clip(table_top_color.astype(np.int16) + shade, 0, 255).astype(np.uint8)
                frame[front_y:, :] = table_front_color
                draw_rect(frame, 0, front_y - 2, image_size - 1, front_y + 3, table_edge_color)

                tape_center = image_size // 2
                tape_w = image_size // 5
                draw_rect(frame, tape_center - tape_w // 2, surface_y - 2, tape_center + tape_w // 2, image_size - 14, tape_color)
                draw_line(frame, (tape_center - 16, surface_y + 14), (tape_center - 4, image_size - 36), np.array([238, 244, 245], dtype=np.uint8), 1)
                draw_line(frame, (tape_center, surface_y + 14), (tape_center + 12, image_size - 36), np.array([238, 244, 245], dtype=np.uint8), 1)
                draw_rect(frame, tape_center - 26, image_size - 34, tape_center + 26, image_size - 24, gripper_color)

                draw_circle(frame, image_size // 2 - 18, image_size // 4, max(8, image_size // 18), np.array([50, 52, 56], dtype=np.uint8))
                draw_circle(frame, image_size // 2 - 18, image_size // 4, max(5, image_size // 22), table_top_color)
                for i in range(n_obj):
                    px, py, scale = world_to_front_px(pos[e, i])
                    radius_px = max(3, int((8 if objs[i]["shape"] == "sphere" else 7) * scale))
                    color = colors[i]
                    if objs[i]["shape"] == "sphere":
                        draw_circle(frame, px + radius_px // 3, py + radius_px, radius_px, np.array([95, 76, 57], dtype=np.uint8))
                        draw_circle(frame, px, py, radius_px, color)
                        draw_circle(frame, px - radius_px // 3, py - radius_px // 4, max(1, radius_px // 3), np.minimum(color.astype(np.int16) + 70, 255).astype(np.uint8))
                    else:
                        darker = (color.astype(np.float32) * 0.55).astype(np.uint8)
                        side_color = (color.astype(np.float32) * 0.72).astype(np.uint8)
                        draw_rect(frame, px - radius_px, py - radius_px, px + radius_px, py + radius_px, color)
                        if px + radius_px + 3 < image_size:
                            frame[
                                max(0, py - radius_px + 2) : min(image_size, py + radius_px + 4),
                                max(0, px + radius_px) : min(image_size, px + radius_px + 4),
                            ] = darker
                        if py + radius_px + 3 < image_size:
                            frame[
                                max(0, py + radius_px) : min(image_size, py + radius_px + 4),
                                max(0, px - radius_px + 2) : min(image_size, px + radius_px + 4),
                            ] = side_color

                tx, ty, scale = world_to_front_px(tcp[e])
                shoulder = (clampi(image_size * 0.72 + (root[e, 1] / max(0.2, y_span)) * image_size * 0.25), image_size - 1)
                elbow = (clampi((shoulder[0] + tx) / 2), clampi((shoulder[1] + ty) / 2 + image_size * 0.08))
                draw_line(frame, shoulder, elbow, arm_color, width=max(1, image_size // 80))
                draw_line(frame, elbow, (tx, ty), arm_color, width=max(1, image_size // 90))
                jaw = max(3, int(7 * scale))
                draw_line(frame, (tx - jaw, ty + jaw), (tx - 2, ty + 1), gripper_color, width=1)
                draw_line(frame, (tx + jaw, ty + jaw), (tx + 2, ty + 1), gripper_color, width=1)
                draw_circle(frame, tx, ty, max(1, image_size // 95), np.array([245, 245, 240], dtype=np.uint8))
            return image
        out = camera.data.output["rgb"]
        rgb = out[..., :3]
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return rgb.detach().cpu().numpy()

    def read_depth() -> np.ndarray | None:
        """Current distance_to_image_plane tensor for all envs as float32."""
        if not capture_depth:
            return None
        if camera is None or "distance_to_image_plane" not in camera.data.output:
            if args.require_depth:
                raise RuntimeError("Isaac camera did not produce distance_to_image_plane depth")
            return None
        depth = camera.data.output["distance_to_image_plane"]
        return depth.detach().cpu().numpy().astype(np.float32, copy=False)

    def read_state_deg() -> np.ndarray:
        """Observed joint state [N,8] in degrees (7 arm + gripper)."""
        jpos = robot.data.joint_pos[:, ent.joint_ids]  # [N,7] rad
        finger = robot.data.joint_pos[:, finger_ids].mean(dim=-1)  # [N]
        jdeg = (jpos * 180.0 / math.pi).detach().cpu().numpy()
        gdeg = np.array([sim_finger_m_to_gripper_deg(float(f)) for f in finger.detach().cpu().tolist()])
        return np.concatenate([jdeg, gdeg[:, None]], axis=1).astype(np.float32)

    def ik_solve_clamped() -> tuple[torch.Tensor, torch.Tensor]:
        """One IK iteration toward the active command. Returns (clamped joints rad [N,7], clamp_hit [N])."""
        jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, ent.joint_ids]
        rp = robot.data.root_pose_w
        ee_w = robot.data.body_pose_w[:, tcp_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
        jpos = robot.data.joint_pos[:, ent.joint_ids]
        jdes = ik.compute(ee_pos_b, ee_quat_b, jac, jpos)
        jclamped = torch.clamp(jdes, arm_lo, arm_hi)
        hit = (jdes != jclamped).any(dim=-1)
        return jclamped, hit, jdes

    def set_ik_command(target_w: torch.Tensor) -> None:
        ik.reset()
        rp = robot.data.root_pose_w
        tpos_b, _ = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], target_w, ident)
        ee_w0 = robot.data.body_pose_w[:, tcp_idx]
        _, ee_quat_b0 = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], ee_w0[:, 0:3], ee_w0[:, 3:7])
        ik.set_command(tpos_b, ee_quat=ee_quat_b0)

    # Dataset writer (created once; episodes appended as they pass).
    import shutil  # noqa: PLC0415

    if args.overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty dataset directory: {dataset_root}")
    features = {
        CAMERA_KEY: {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]},
        STATE_KEY: {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
    }
    if capture_depth:
        features[DEPTH_KEY] = {
            "dtype": "float32",
            "shape": (image_size, image_size, 1),
            "names": ["height", "width", "channels"],
        }
    dataset_backend = ""
    if args.dataset_backend in ("auto", "lerobot"):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

            dataset = LeRobotDataset.create(
                repo_id=args.repo_id, root=dataset_root, fps=args.fps,
                robot_type="openarm_synthetic_isaac_dense", features=features,
                use_videos=False, image_writer_threads=0, image_writer_processes=0,
            )
            dataset_backend = "lerobot"
        except ModuleNotFoundError as exc:
            if exc.name != "lerobot":
                raise
            if args.dataset_backend == "lerobot":
                raise
            dataset = LocalNpzEpisodeDataset.create(
                repo_id=args.repo_id,
                root=dataset_root,
                fps=args.fps,
                robot_type="openarm_synthetic_isaac_dense",
                features=features,
            )
            dataset_backend = "local_npz"
            print(
                "[dense] lerobot is not installed in this Python environment; "
                f"writing local NPZ episodes under {dataset_root}",
                file=sys.stderr,
                flush=True,
            )
    else:
        dataset = LocalNpzEpisodeDataset.create(
            repo_id=args.repo_id,
            root=dataset_root,
            fps=args.fps,
            robot_type="openarm_synthetic_isaac_dense",
            features=features,
        )
        dataset_backend = "local_npz"
        print(
            f"[dense] writing local NPZ episodes under {dataset_root}",
            file=sys.stderr,
            flush=True,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_fh = manifest_path.open("w", encoding="utf-8")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)
    wtensor = torch.tensor(weights, dtype=torch.float32)

    kept = 0
    source_total = 0
    src_by_target: Counter = Counter()
    kept_by_target: Counter = Counter()
    wrong_total = 0
    clamp_total = 0
    object_collision_total = 0
    gripper_table_collision_total = 0
    object_sweep_total = 0
    finger_penetration_total = 0
    object_pushed_down_total = 0
    refined_clip_total = 0
    action_slew_violation_total = 0
    gripper_cap_violation_total = 0
    saved_sample = False

    def gripper_schedule_deg(phase: str, k: int, n: int, close_target_deg: torch.Tensor) -> torch.Tensor:
        if phase in ("approach", "descend"):
            return torch.full((N,), float(open_gripper_deg), device=device)
        if phase == "close":
            frac = (k + 1) / max(1, n)
            return float(open_gripper_deg) + (close_target_deg - float(open_gripper_deg)) * frac
        return close_target_deg  # lift, hold

    for rnd in range(args.rounds):
        if args.max_keep and kept >= args.max_keep:
            print(
                f"[dense] reached --max-keep={args.max_keep}; stopping before round {rnd+1}/{args.rounds}.",
                file=sys.stderr,
                flush=True,
            )
            break
        if target_quotas_satisfied(kept_by_target, target_quotas):
            print(
                f"[dense] target quotas satisfied; stopping before round {rnd+1}/{args.rounds}.",
                file=sys.stderr,
                flush=True,
            )
            break
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        origins = scene.env_origins
        ep_obj_local: dict[str, torch.Tensor] = {}
        for name in obj_names:
            base = torch.tensor(spawn_for[name], device=device)
            jit = torch.zeros((N, 3), device=device)
            if args.randomize:
                jx = torch.empty(N).uniform_(-float(args.jitter_x_m), float(args.jitter_x_m), generator=rng)
                jy = torch.empty(N).uniform_(-float(args.jitter_y_m), float(args.jitter_y_m), generator=rng)
                jit[:, 0] = jx.to(device)
                jit[:, 1] = jy.to(device)
            local = base.unsqueeze(0) + jit
            local[:, 0] = torch.clamp(local[:, 0], bounds["x"][0], bounds["x"][1])
            local[:, 1] = torch.clamp(local[:, 1], bounds["y"][0], bounds["y"][1])
            ep_obj_local[name] = local

        def write_episode_objects_to_sim() -> None:
            for obj_name in obj_names:
                asset = scene[obj_name]
                root = asset.data.default_root_state.clone()
                root[:, 0:3] = ep_obj_local[obj_name] + origins
                root[:, 3:7] = ident
                root[:, 7:] = 0.0
                asset.write_root_pose_to_sim(root[:, :7])
                asset.write_root_velocity_to_sim(root[:, 7:])

        init_arm_pos = robot.data.default_joint_pos[:, ent.joint_ids]
        zero_arm_pos = torch.zeros_like(init_arm_pos)
        write_episode_objects_to_sim()
        if args.record_zero_to_init:
            zero_pos = robot.data.default_joint_pos.clone()
            zero_vel = robot.data.default_joint_vel.clone()
            zero_pos[:, ent.joint_ids] = zero_arm_pos
            zero_pos[:, finger_ids] = float(gripper_deg_to_sim_finger_m(float(args.zero_start_gripper_deg)))
            zero_vel[:, ent.joint_ids] = 0.0
            zero_vel[:, finger_ids] = 0.0
            robot.write_joint_state_to_sim(zero_pos, zero_vel)
            robot.set_joint_position_target(zero_arm_pos, joint_ids=arm_ids)
            zero_gripper_target = torch.full((N,), float(args.zero_start_gripper_deg), device=device)
            set_gripper_deg_targets(zero_gripper_target, cap_close=False)
        elif staged_init_tensor is not None:
            stage1 = staged_init_tensor[0]
            stage1_arm_rad = (stage1[:7] * math.pi / 180.0).unsqueeze(0).repeat(N, 1)
            stage1_gripper_m = gripper_deg_to_sim_finger_m(float(stage1[-1].item()))
            staged_pos = robot.data.default_joint_pos.clone()
            staged_vel = robot.data.default_joint_vel.clone()
            staged_pos[:, ent.joint_ids] = stage1_arm_rad
            staged_pos[:, finger_ids] = float(stage1_gripper_m)
            staged_vel[:, ent.joint_ids] = 0.0
            staged_vel[:, finger_ids] = 0.0
            robot.write_joint_state_to_sim(staged_pos, staged_vel)
            robot.set_joint_position_target(stage1_arm_rad, joint_ids=arm_ids)
            stage1_gripper_target = torch.full((N,), float(stage1[-1].item()), device=device)
            set_gripper_deg_targets(stage1_gripper_target, cap_close=False)
        else:
            set_gripper(open_m)
            # Hold the active arm at its configured init (reset) pose during settle.
            # Without an explicit position target the arm free-falls under gravity from
            # an extended init pose, so the recorded episode would start from a sagged
            # pose far from the configured one. Driving the default joint pose keeps the
            # configured collision-free init pose in effect when recording begins.
            robot.set_joint_position_target(init_arm_pos, joint_ids=arm_ids)
        step_phys(args.settle_steps, render_last=True)  # also primes the camera render

        # Weighted target per env.
        target_idx = torch.multinomial(wtensor, N, replacement=True, generator=rng).to(device)
        if gripper_close_range_deg is None:
            close_target_deg = torch.full((N,), float(effective_close_deg), device=device)
        else:
            close_lo, close_hi = gripper_close_range_deg
            if abs(close_hi - close_lo) < 1.0e-9:
                sampled_close = torch.full((N,), float(close_lo), dtype=torch.float32)
            else:
                sampled_close = torch.empty(N, dtype=torch.float32).uniform_(
                    float(close_lo), float(close_hi), generator=rng
                )
            close_target_deg = torch.minimum(
                sampled_close.to(device),
                torch.full((N,), float(effective_close_deg), device=device),
            )

        def episode_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            ow_now = obj_pos_w()
            baseline_pos_now = ow_now.clone()
            baseline_now = baseline_pos_now[:, :, 2].clone()
            baseline_xy_now = baseline_pos_now[:, :, 0:2].clone()
            tw_now = ow_now.gather(1, target_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            ox_now, oy_now = tw_now[:, 0], tw_now[:, 1]
            grasp_z_now = tw_now[:, 2] + float(args.grasp_z_offset_m)
            above_now = torch.stack([ox_now, oy_now, grasp_z_now + args.above_offset_m], dim=-1)
            descend_now = torch.stack([ox_now, oy_now, grasp_z_now], dim=-1)
            lift_now = torch.stack([ox_now, oy_now, grasp_z_now + args.lift_offset_m], dim=-1)
            return baseline_pos_now, baseline_now, baseline_xy_now, tw_now, above_now, descend_now, lift_now

        baseline_pos, baseline, baseline_xy, tw, above, descend, lift = episode_geometry()

        reset_arm = robot.data.joint_pos[:, ent.joint_ids].clone()
        ready_arm = reset_arm
        if args.prepose_to_ready:
            set_ik_command(above)
            for _ in range(args.prepose_warmup_steps):
                jclamped, _, _ = ik_solve_clamped()
                robot.set_joint_position_target(jclamped, joint_ids=arm_ids)
                set_gripper(open_m)
                step_phys(1, render_last=False)
            ready_arm = robot.data.joint_pos[:, ent.joint_ids].clone()

            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            write_episode_objects_to_sim()
            set_gripper(open_m)
            step_phys(args.settle_steps, render_last=True)
            baseline_pos, baseline, baseline_xy, tw, above, descend, lift = episode_geometry()
            reset_arm = robot.data.joint_pos[:, ent.joint_ids].clone()

        # Per-env dense buffers.
        rgb_buf = [[] for _ in range(N)]
        depth_buf = [[] for _ in range(N)] if capture_depth else None
        state_buf = [[] for _ in range(N)]
        action_buf = [[] for _ in range(N)]
        contact_steps = torch.zeros(N, device=device)
        clamp_hit = torch.zeros(N, dtype=torch.bool, device=device)
        object_collision_hit = torch.zeros(N, dtype=torch.bool, device=device)
        gripper_table_collision_hit = torch.zeros(N, dtype=torch.bool, device=device)
        object_sweep_hit = torch.zeros(N, dtype=torch.bool, device=device)
        min_object_distance = torch.full((N,), float("inf"), device=device)
        min_tcp_table_clearance = torch.full((N,), float("inf"), device=device)
        max_surface_sweep = torch.zeros(N, device=device)
        # Authoritative new safety accumulators.
        finger_penetration_hit = torch.zeros(N, dtype=torch.bool, device=device)
        min_finger_table_clearance = torch.full((N,), float("inf"), device=device)
        object_pushed_down_hit = torch.zeros(N, dtype=torch.bool, device=device)
        refined_clip_hit = torch.zeros(N, dtype=torch.bool, device=device)
        max_refined_clip_deg = torch.zeros(N, device=device)
        action_step_limited_hit = torch.zeros(N, dtype=torch.bool, device=device)
        max_action_delta_deg = torch.zeros(N, device=device)
        max_raw_action_delta_deg = torch.zeros(N, device=device)
        lift_stop_step = torch.full((N,), -1, dtype=torch.int64, device=device)
        frozen = torch.zeros(N, dtype=torch.bool, device=device)
        command_step = 0
        if args.record_zero_to_init:
            initial_state_deg = np.zeros((N, len(STATE_NAMES)), dtype=np.float32)
            initial_state_deg[:, -1] = float(args.zero_start_gripper_deg)
        elif staged_init_tensor is not None:
            initial_state_deg = staged_init_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
            initial_state_deg = np.repeat(initial_state_deg[None, :], N, axis=0)
        else:
            initial_state_deg = read_state_deg()
            initial_state_deg[:, -1] = np.minimum(initial_state_deg[:, -1], effective_close_deg)
        last_action_deg = torch.tensor(initial_state_deg, device=device, dtype=torch.float32)
        max_step_deg = float(max_action_step_deg)

        def apply_action_upsampling_limit(
            target_arm_rad: torch.Tensor,
            target_gripper_deg: float | torch.Tensor,
            *,
            cap_gripper: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal last_action_deg, action_step_limited_hit, max_action_delta_deg, max_raw_action_delta_deg
            arm_target_deg = target_arm_rad * 180.0 / math.pi
            if isinstance(target_gripper_deg, torch.Tensor):
                grip = target_gripper_deg.to(device=device, dtype=torch.float32).view(N, 1)
            else:
                grip = torch.full((N, 1), float(target_gripper_deg), device=device)
            if cap_gripper:
                grip = torch.minimum(grip, torch.full((N, 1), float(effective_close_deg), device=device))
            grip_target = grip
            target = torch.cat([arm_target_deg, grip_target], dim=-1)
            target = torch.where(frozen.unsqueeze(1), last_action_deg, target)
            raw_delta = target - last_action_deg
            raw_max = torch.abs(raw_delta).max(dim=-1).values
            max_raw_action_delta_deg = torch.maximum(max_raw_action_delta_deg, raw_max)
            if max_step_deg > 0.0:
                # Stay just below the public slew cap so float32 roundoff cannot
                # turn a nominal 1.5 deg command into 1.50000x in saved actions.
                step_limit = max(0.0, max_step_deg - 1.0e-4)
                limited_delta = torch.clamp(raw_delta, min=-step_limit, max=step_limit)
                action_step_limited_hit |= raw_max > (max_step_deg + 1.0e-6)
                command = last_action_deg + limited_delta
            else:
                command = target
            if cap_gripper:
                command[:, -1] = torch.minimum(command[:, -1], torch.full((N,), effective_close_deg, device=device))
            applied_max = torch.abs(command - last_action_deg).max(dim=-1).values
            max_action_delta_deg = torch.maximum(max_action_delta_deg, applied_max)
            last_action_deg = command
            return command[:, :7] * math.pi / 180.0, command[:, -1]

        def object_sweep_state() -> tuple[torch.Tensor, torch.Tensor]:
            pos = obj_pos_w()
            xy_delta = torch.linalg.norm(pos[:, :, 0:2] - baseline_xy, dim=-1)
            still_on_table = (pos[:, :, 2] - baseline) < float(args.lift_threshold_m)
            sweep = (xy_delta > float(args.object_sweep_threshold_m)) & still_on_table
            return sweep.any(dim=-1), xy_delta.max(dim=-1).values

        def object_pushdown_state() -> torch.Tensor:
            z = obj_pos_w()[:, :, 2]  # [N, n_obj]
            return ((baseline - z) > float(args.object_pushdown_margin_m)).any(dim=-1)

        obj_hit, obj_dist = object_collision_state()
        table_hit, tcp_clearance = gripper_table_collision_state()
        sweep_hit, surface_sweep = object_sweep_state()
        fp_hit, fp_clear = finger_table_state()
        object_collision_hit |= obj_hit
        gripper_table_collision_hit |= table_hit
        object_sweep_hit |= sweep_hit
        finger_penetration_hit |= fp_hit
        object_pushed_down_hit |= object_pushdown_state()
        min_object_distance = torch.minimum(min_object_distance, obj_dist)
        min_tcp_table_clearance = torch.minimum(min_tcp_table_clearance, tcp_clearance)
        min_finger_table_clearance = torch.minimum(min_finger_table_clearance, fp_clear)
        max_surface_sweep = torch.maximum(max_surface_sweep, surface_sweep)

        phase_target = {"approach": above, "descend": descend, "close": descend, "lift": lift, "hold": lift}
        for phase, n_steps in phase_plan:
            if phase not in ("zero_to_init", "staged_init") and not (args.prepose_to_ready and phase == "approach"):
                set_ik_command(phase_target[phase])
            for k in range(n_steps):
                rgb = read_rgb()
                depth = read_depth()
                state = read_state_deg()
                cap_gripper_this_step = True
                if phase == "zero_to_init":
                    frac = k / max(1, n_steps - 1)
                    jraw = zero_arm_pos + (init_arm_pos - zero_arm_pos) * frac
                    gtarget_deg = float(args.zero_start_gripper_deg) + (
                        float(open_gripper_deg) - float(args.zero_start_gripper_deg)
                    ) * frac
                    hit = torch.zeros(N, dtype=torch.bool, device=device)
                    cap_gripper_this_step = False
                elif phase == "staged_init":
                    assert staged_init_tensor is not None
                    staged_command = staged_init_tensor[k].unsqueeze(0).repeat(N, 1)
                    jraw = staged_command[:, :7] * math.pi / 180.0
                    gtarget_deg = staged_command[:, -1]
                    hit = torch.zeros(N, dtype=torch.bool, device=device)
                    cap_gripper_this_step = False
                elif args.prepose_to_ready and phase == "approach":
                    frac = (k + 1) / max(1, n_steps)
                    jraw = reset_arm + (ready_arm - reset_arm) * frac
                    hit = torch.zeros(N, dtype=torch.bool, device=device)
                    gtarget_deg = gripper_schedule_deg(phase, k, n_steps, close_target_deg)
                elif phase in ("approach", "descend", "lift"):
                    jraw, hit, jdes = ik_solve_clamped()
                    active = ~frozen
                    clamp_hit |= hit & active
                    # Refined clip: genuine commanded-action clipping, ignoring the
                    # unavoidable joint_4 zero-start lower-bound correction.
                    clip_deg = torch.abs(jdes - jraw) * 180.0 / math.pi  # [N,7]
                    j4 = clip_deg[:, joint4_idx]
                    clip_eff = clip_deg.clone()
                    clip_eff[:, joint4_idx] = torch.where(
                        j4 <= float(args.joint4_startup_tol_deg), torch.zeros_like(j4), j4
                    )
                    step_max_clip = clip_eff.max(dim=-1).values  # [N]
                    refined_clip_hit |= (step_max_clip > float(args.action_clip_tol_deg)) & active
                    max_refined_clip_deg = torch.maximum(max_refined_clip_deg, torch.where(active, step_max_clip, torch.zeros_like(step_max_clip)))
                    gtarget_deg = gripper_schedule_deg(phase, k, n_steps, close_target_deg)
                else:  # close, hold -> hold last arm target
                    jraw = last_action_deg[:, :7] * math.pi / 180.0
                    gtarget_deg = gripper_schedule_deg(phase, k, n_steps, close_target_deg)
                jclamped, gcmd_deg = apply_action_upsampling_limit(
                    jraw, gtarget_deg, cap_gripper=cap_gripper_this_step
                )
                arm_deg = (jclamped * 180.0 / math.pi).detach().cpu().numpy()
                action = np.concatenate(
                    [arm_deg, gcmd_deg.detach().cpu().numpy()[:, None]], axis=1
                ).astype(np.float32)
                for e in range(N):
                    rgb_buf[e].append(rgb[e])
                    if depth_buf is not None:
                        assert depth is not None
                        depth_buf[e].append(depth[e])
                    state_buf[e].append(state[e])
                    action_buf[e].append(action[e])
                # contact bookkeeping: tcp close to target object in xy/z
                tcp_w = robot.data.body_pose_w[:, tcp_idx, 0:3]
                d = torch.linalg.norm(tcp_w - tw, dim=-1)
                contact_steps += (d < args.contact_eps_m).float()
                robot.set_joint_position_target(jclamped, joint_ids=arm_ids)
                set_gripper_deg_targets(gcmd_deg, cap_close=cap_gripper_this_step)
                step_phys(args.substeps, render_last=True)
                command_step += 1
                obj_hit, obj_dist = object_collision_state()
                table_hit, tcp_clearance = gripper_table_collision_state()
                sweep_hit, surface_sweep = object_sweep_state()
                fp_hit, fp_clear = finger_table_state()
                object_collision_hit |= obj_hit
                gripper_table_collision_hit |= table_hit
                object_sweep_hit |= sweep_hit
                finger_penetration_hit |= fp_hit
                object_pushed_down_hit |= object_pushdown_state()
                min_object_distance = torch.minimum(min_object_distance, obj_dist)
                min_tcp_table_clearance = torch.minimum(min_tcp_table_clearance, tcp_clearance)
                min_finger_table_clearance = torch.minimum(min_finger_table_clearance, fp_clear)
                max_surface_sweep = torch.maximum(max_surface_sweep, surface_sweep)
                if args.early_stop_on_lift:
                    current_rises = obj_pos_w()[:, :, 2] - baseline
                    current_target_rise = current_rises.gather(1, target_idx.view(-1, 1)).squeeze(1)
                    newly_frozen = (~frozen) & (current_target_rise >= float(args.lift_threshold_m))
                    lift_stop_step = torch.where(
                        newly_frozen,
                        torch.full_like(lift_stop_step, command_step),
                        lift_stop_step,
                    )
                    frozen |= newly_frozen

        final = obj_pos_w()[:, :, 2]
        rises = final - baseline
        target_rise = rises.gather(1, target_idx.view(-1, 1)).squeeze(1)
        success = target_rise >= args.lift_threshold_m
        wrong_mask = torch.ones_like(rises)
        wrong_mask.scatter_(1, target_idx.view(-1, 1), 0.0)
        wrong_any = ((rises * wrong_mask) >= args.lift_threshold_m).any(dim=-1)

        for e in range(N):
            source_total += 1
            tname = obj_names[int(target_idx[e])]
            src_by_target[tname] += 1
            is_wrong = bool(wrong_any[e].item())
            is_clamp = bool(clamp_hit[e].item())
            is_object_collision = bool(object_collision_hit[e].item())
            is_gripper_table_collision = bool(gripper_table_collision_hit[e].item())
            is_object_sweep = bool(object_sweep_hit[e].item())
            is_finger_penetration = bool(finger_penetration_hit[e].item())
            is_object_pushed_down = bool(object_pushed_down_hit[e].item())
            is_refined_clip = bool(refined_clip_hit[e].item())
            episode_actions = action_buf[e]
            gripper_cmd_max = max(float(a[-1]) for a in episode_actions)
            gripper_cmd_min = min(float(a[-1]) for a in episode_actions)
            if args.record_zero_to_init:
                post_init_start = int(args.zero_init_steps)
            elif staged_init_commands is not None:
                post_init_start = len(staged_init_commands)
            else:
                post_init_start = 0
            post_init_actions = episode_actions[post_init_start:] or episode_actions
            post_init_gripper_cmd_max = max(float(a[-1]) for a in post_init_actions)
            is_gripper_cap_violation = post_init_gripper_cmd_max > effective_close_deg + 1.0e-6
            is_action_slew_violation = (
                max_step_deg > 0.0 and float(max_action_delta_deg[e].item()) > max_step_deg + 1.0e-6
            )
            wrong_total += int(is_wrong)
            clamp_total += int(is_clamp)
            object_collision_total += int(is_object_collision)
            gripper_table_collision_total += int(is_gripper_table_collision)
            object_sweep_total += int(is_object_sweep)
            finger_penetration_total += int(is_finger_penetration)
            object_pushed_down_total += int(is_object_pushed_down)
            refined_clip_total += int(is_refined_clip)
            action_slew_violation_total += int(is_action_slew_violation)
            gripper_cap_violation_total += int(is_gripper_cap_violation)
            keep = (
                bool(success[e].item())
                and not is_wrong
                and not is_object_collision
                and not is_gripper_table_collision
                and not is_object_sweep
                and not is_finger_penetration       # authoritative tabletop penetration
                and not is_object_pushed_down        # object driven into the table
                and not is_refined_clip              # genuine unsafe action clipping (joint_4 zero-start excluded)
                and not is_action_slew_violation
                and not is_gripper_cap_violation
            )
            if args.drop_limit_exceeded and is_clamp:
                keep = False
            quota_already_filled = False
            if target_quotas and kept_by_target[tname] >= target_quotas.get(tname, 0):
                quota_already_filled = True
                keep = False
            if args.max_keep and kept >= args.max_keep:
                keep = False

            poses = {name: [round(float(v), 4) for v in ep_obj_local[name][e].tolist()] for name in obj_names}
            meta = {
                "schema_version": "openarm_dense_isaac_camera_v1",
                "source": "synthetic_smolvla.collect_dense_isaac_dataset",
                "dataset_backend": dataset_backend,
                "episode_index": source_total - 1,
                "kept": keep,
                "target_quota": int(target_quotas.get(tname, 0)) if target_quotas else 0,
                "target_quota_already_filled": quota_already_filled,
                "instruction": instruction_for[tname],
                "target_object": tname,
                "arm_side": side,
                "robot_plus_table_height_cm": layout_info.get("robot_plus_table_height"),
                "height_sweep": height_sweep_info,
                "image_size": image_size,
                "episode_len": episode_len,
                "prepose_to_ready": bool(args.prepose_to_ready),
                "prepose_warmup_steps": int(args.prepose_warmup_steps) if args.prepose_to_ready else 0,
                "record_zero_to_init": bool(args.record_zero_to_init),
                "zero_init_steps": int(args.zero_init_steps) if args.record_zero_to_init else 0,
                "zero_start_gripper_deg": round(float(args.zero_start_gripper_deg), 3) if args.record_zero_to_init else None,
                "staged_init": staged_init_commands is not None,
                "staged_init_name": args.staged_init_name,
                "staged_init_csv": str(args.staged_init_csv) if args.staged_init_csv else None,
                "staged_init_steps": len(staged_init_commands) if staged_init_commands is not None else 0,
                "staged_init_stages_deg": None if staged_init_stages is None else [
                    [round(float(v), 6) for v in stage] for stage in staged_init_stages
                ],
                "target_episode_commands": int(args.target_episode_commands) if args.target_episode_commands else None,
                "init_gripper_deg": round(float(open_gripper_deg), 3),
                "camera_mode": str(args.camera_mode),
                "placeholder_view": str(args.placeholder_view) if not use_isaac_camera else None,
                "depth_key": DEPTH_KEY if capture_depth else None,
                "depth_shape": None if depth_buf is None else list(np.asarray(depth_buf[e][0]).shape),
                "object_poses_m": poses,
                "object_rises_m": {name: round(float(rises[e, i]), 5) for i, name in enumerate(obj_names)},
                "target_rise_m": round(float(target_rise[e]), 5),
                "contact_steps": int(contact_steps[e].item()),
                "success_label": bool(success[e].item()),
                "wrong_object_lifted": is_wrong,
                "limit_exceeded": is_clamp,
                "object_collision": is_object_collision,
                "gripper_table_collision": is_gripper_table_collision,
                "object_swept_or_slid": is_object_sweep,
                "min_object_distance_m": round(float(min_object_distance[e]), 5),
                "min_tcp_table_clearance_m": round(float(min_tcp_table_clearance[e]), 5),
                "max_surface_sweep_m": round(float(max_surface_sweep[e]), 5),
                # Authoritative new safety fields (replace the misleading tcp/limit diagnostics).
                "min_finger_table_clearance_m": round(float(min_finger_table_clearance[e]), 5),
                "tabletop_penetration": is_finger_penetration,
                "object_pushed_down": is_object_pushed_down,
                "refined_action_clip": is_refined_clip,
                "max_refined_action_clip_deg": round(float(max_refined_clip_deg[e]), 4),
                "action_step_limited": bool(action_step_limited_hit[e].item()),
                "action_slew_violation": is_action_slew_violation,
                "max_action_delta_deg": round(float(max_action_delta_deg[e]), 4),
                "max_raw_action_delta_deg": round(float(max_raw_action_delta_deg[e]), 4),
                "max_action_step_limit_deg": round(float(max_step_deg), 4),
                # Backward-compatible aliases used by existing audits.
                "arm_action_step_limited": bool(action_step_limited_hit[e].item()),
                "max_arm_action_delta_deg": round(float(max_action_delta_deg[e]), 4),
                "max_arm_raw_delta_deg": round(float(max_raw_action_delta_deg[e]), 4),
                "max_arm_action_step_limit_deg": round(float(max_step_deg), 4),
                "early_stop_on_lift": bool(args.early_stop_on_lift),
                "lift_stop_step": None if int(lift_stop_step[e].item()) < 0 else int(lift_stop_step[e].item()),
                "hold_padded_after_lift": bool(args.early_stop_on_lift and int(lift_stop_step[e].item()) >= 0),
                "finger_body_names": list(finger_body_names),
                "substeps": int(args.substeps),
                "jitter_x_m": round(float(args.jitter_x_m), 5),
                "jitter_y_m": round(float(args.jitter_y_m), 5),
                "gripper_cmd_min_deg": round(float(gripper_cmd_min), 3),
                "gripper_cmd_max_deg": round(float(gripper_cmd_max), 3),
                "post_init_gripper_cmd_max_deg": round(float(post_init_gripper_cmd_max), 3),
                "open_gripper_deg": round(float(open_gripper_deg), 3),
                "gripper_close_cap_deg": round(float(effective_close_deg), 3),
                "gripper_close_range_deg": None if gripper_close_range_deg is None else [
                    round(float(gripper_close_range_deg[0]), 3),
                    round(float(gripper_close_range_deg[1]), 3),
                ],
                "gripper_close_target_deg": round(float(close_target_deg[e].item()), 3),
                "gripper_cap_violation": is_gripper_cap_violation,
                "grasp_close_deg": round(float(effective_close_deg), 3),
                "grasp_z_offset_m": round(float(args.grasp_z_offset_m), 5),
                "lift_offset_m": round(float(args.lift_offset_m), 5),
                "upsampling": {
                    "enabled": bool(max_step_deg > 0.0),
                    "target_fps": int(args.fps),
                    "max_deg_per_command": round(float(max_step_deg), 4),
                    "max_deg_per_second": round(float(max_step_deg) * float(args.fps), 4),
                    "dimensions": 8,
                    "final_commands": int(episode_len),
                },
                "state_trace_deg": [[round(float(v), 3) for v in s.tolist()] for s in state_buf[e]],
                "action_trace_deg": [[round(float(v), 3) for v in a.tolist()] for a in action_buf[e]],
            }

            if sample_dir is not None and not saved_sample:
                saved_sample = True
                sample_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from PIL import Image  # noqa: PLC0415

                    status = "kept" if keep else "failed"
                    for t in (0, episode_len // 3, 2 * episode_len // 3, episode_len - 1):
                        Image.fromarray(rgb_buf[e][t]).save(sample_dir / f"ep0_{status}_{tname}_step{t:02d}.png")
                    meta["sample_frames"] = [
                        str(sample_dir / f"ep0_{status}_{tname}_step{t:02d}.png")
                        for t in (0, episode_len // 3, 2 * episode_len // 3, episode_len - 1)
                    ]
                except Exception as exc:  # pragma: no cover
                    print(f"[dense] sample frame dump failed: {exc}", file=sys.stderr, flush=True)

            manifest_fh.write(json.dumps(meta) + "\n")

            if not keep:
                continue
            for t in range(episode_len):
                frame = {
                    CAMERA_KEY: rgb_buf[e][t],
                    STATE_KEY: state_buf[e][t],
                    ACTION_KEY: action_buf[e][t],
                    "task": instruction_for[tname],
                }
                if depth_buf is not None:
                    frame[DEPTH_KEY] = depth_buf[e][t]
                dataset.add_frame(frame)
            dataset.save_episode()
            kept += 1
            kept_by_target[tname] += 1
            if args.max_keep and kept >= args.max_keep:
                break
            if target_quotas_satisfied(kept_by_target, target_quotas):
                break

        print(f"[dense] round {rnd+1}/{args.rounds}: source={source_total} kept={kept} "
              f"(round success={float(success.float().mean()):.3f})", file=sys.stderr, flush=True)
        if target_quotas_satisfied(kept_by_target, target_quotas):
            print(f"[dense] target quotas satisfied after round {rnd+1}.", file=sys.stderr, flush=True)
            break

    dataset.finalize()
    manifest_fh.close()

    # Report.
    def rate(c: int) -> float:
        return 0.0 if source_total == 0 else c / source_total

    lines = [
        "# Dense Isaac-Camera SmolVLA Dataset (v1) — collection",
        "",
        f"Source episodes: {source_total} across {N} envs x {args.rounds} rounds.",
        f"Kept (measured success, no wrong-object): {kept} ({rate(kept):.3f}).",
        "",
        f"Each kept episode is a DENSE rollout with `{args.camera_mode}` camera frames at every",
        f"control step ({episode_len} steps/episode, {image_size}x{image_size} RGB).",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Source episodes | {source_total} | 1.000 |",
        f"| Kept successes | {kept} | {rate(kept):.3f} |",
        f"| Wrong-object lifts (source) | {wrong_total} | {rate(wrong_total):.3f} |",
        f"| Limit-clamp episodes (source) | {clamp_total} | {rate(clamp_total):.3f} |",
        f"| Object-collision episodes (source) | {object_collision_total} | {rate(object_collision_total):.3f} |",
        f"| Gripper/table collision episodes (source) | {gripper_table_collision_total} | {rate(gripper_table_collision_total):.3f} |",
        f"| Object sweep/slide episodes (source) | {object_sweep_total} | {rate(object_sweep_total):.3f} |",
        f"| Tabletop-penetration episodes (source, finger body) | {finger_penetration_total} | {rate(finger_penetration_total):.3f} |",
        f"| Object-pushed-down episodes (source) | {object_pushed_down_total} | {rate(object_pushed_down_total):.3f} |",
        f"| Refined-action-clip episodes (source) | {refined_clip_total} | {rate(refined_clip_total):.3f} |",
        f"| Action-slew violation episodes (source) | {action_slew_violation_total} | {rate(action_slew_violation_total):.3f} |",
        f"| Gripper-cap violation episodes (source) | {gripper_cap_violation_total} | {rate(gripper_cap_violation_total):.3f} |",
        "",
        "## Targets",
        "",
        "| Target | Kept | Source |",
        "|---|---:|---:|",
    ]
    for t in obj_names:
        lines.append(f"| {t} | {kept_by_target[t]} | {src_by_target[t]} |")
    lines += [
        "",
        "## Files",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Dataset backend: `{dataset_backend}`",
        f"- Repo id: `{args.repo_id}`",
        f"- Episode metadata JSONL (dense state/action + poses/rises/contact): `{manifest_path}`",
        f"- Sample frames: `{args.sample_frame_dir}`",
        "",
        "## Notes",
        "",
        "- `observation.state` is the measured joint state; `action` is the clamped IK command.",
        f"- Camera mode is `{args.camera_mode}`.",
        f"- Placeholder view is `{args.placeholder_view if not use_isaac_camera else 'n/a'}`.",
        f"- Depth capture is `{capture_depth}` with key `{DEPTH_KEY if capture_depth else 'n/a'}`.",
        "- Only successful, correct-object lifts are kept; wrong-object lifts, object collisions, gripper/table collisions, and object sweep/slide episodes are rejected.",
        f"- Prepose-to-ready is `{bool(args.prepose_to_ready)}` with `{int(args.prepose_warmup_steps) if args.prepose_to_ready else 0}` non-recorded warmup steps.",
        f"- Record zero-to-init is `{bool(args.record_zero_to_init)}` with `{int(args.zero_init_steps) if args.record_zero_to_init else 0}` recorded setup steps.",
        f"- Staged init is `{staged_init_commands is not None}` with `{len(staged_init_commands) if staged_init_commands is not None else 0}` recorded setup steps.",
        f"- Staged init name is `{args.staged_init_name or 'n/a'}`.",
        f"- Target episode commands is `{int(args.target_episode_commands) if args.target_episode_commands else 'disabled'}`.",
        f"- Recorded 8D command slew limit is `{float(max_step_deg):.3f}` deg/control step (`0` disables it).",
        f"- 8D command upsampling is `{'enabled' if max_step_deg > 0.0 else 'disabled'}`.",
        f"- Early stop on 5 cm lift is `{bool(args.early_stop_on_lift)}`.",
        f"- Gripper init/open command is `{open_gripper_deg:.3f}` deg.",
        f"- Gripper close command is capped at `{effective_close_deg:.3f}` deg.",
        f"- Gripper close target range is `{gripper_close_range_deg or 'disabled'}`.",
        f"- Lift waypoint is `{float(args.lift_offset_m):.3f}` m above the grasp waypoint.",
        f"- Grasp z offset is `{float(args.grasp_z_offset_m):.3f}` m.",
    ]
    if target_quotas:
        lines.extend([
            f"- Target success quotas: `{target_quotas}`.",
            f"- Target quotas satisfied: `{target_quotas_satisfied(kept_by_target, target_quotas)}`.",
        ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True, "source_episodes": source_total, "kept": kept, "kept_rate": rate(kept),
        "wrong_object": wrong_total, "limit_clamp": clamp_total,
        "object_collision": object_collision_total,
        "gripper_table_collision": gripper_table_collision_total,
        "object_sweep": object_sweep_total,
        "tabletop_penetration": finger_penetration_total,
        "object_pushed_down": object_pushed_down_total,
        "refined_action_clip": refined_clip_total,
        "action_slew_violation": action_slew_violation_total,
        "gripper_cap_violation": gripper_cap_violation_total,
        "grasp_close_deg": effective_close_deg,
        "open_gripper_deg": open_gripper_deg,
        "prepose_to_ready": bool(args.prepose_to_ready),
        "prepose_warmup_steps": int(args.prepose_warmup_steps) if args.prepose_to_ready else 0,
        "record_zero_to_init": bool(args.record_zero_to_init),
        "zero_init_steps": int(args.zero_init_steps) if args.record_zero_to_init else 0,
        "zero_start_gripper_deg": float(args.zero_start_gripper_deg) if args.record_zero_to_init else None,
        "staged_init": staged_init_commands is not None,
        "staged_init_name": args.staged_init_name,
        "staged_init_csv": str(args.staged_init_csv) if args.staged_init_csv else None,
        "staged_init_steps": len(staged_init_commands) if staged_init_commands is not None else 0,
        "target_episode_commands": int(args.target_episode_commands) if args.target_episode_commands else None,
        "init_gripper_deg": float(open_gripper_deg),
        "gripper_close_range_deg": gripper_close_range_deg,
        "camera_mode": str(args.camera_mode),
        "placeholder_view": str(args.placeholder_view) if not use_isaac_camera else None,
        "capture_depth": capture_depth,
        "depth_key": DEPTH_KEY if capture_depth else None,
        "dataset_backend": dataset_backend,
        "max_action_step_deg": float(max_step_deg),
        "max_arm_action_step_deg": float(max_step_deg),
        "early_stop_on_lift": bool(args.early_stop_on_lift),
        "target_quotas": target_quotas,
        "target_quotas_satisfied": target_quotas_satisfied(kept_by_target, target_quotas),
        "grasp_z_offset_m": args.grasp_z_offset_m,
        "lift_offset_m": args.lift_offset_m,
        "image_size": image_size,
        "episode_len": episode_len, "dataset_root": str(dataset_root), "repo_id": args.repo_id,
        "manifest": str(manifest_path), "report": str(report_path),
        "kept_by_target": dict(kept_by_target), "source_by_target": dict(src_by_target),
    }, indent=2), flush=True)

    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
