#!/usr/bin/env python3
"""Run the oracle joint plan in Isaac physics and MEASURE pick success.

Unlike ``collect_oracle_demos.py`` (which writes a hard-coded ``success_label``),
this evaluator actually steps Isaac Lab physics, drives the OpenArm right arm
through the oracle approach -> lower -> close -> lift plan, and labels each
episode by whether the *target* object's height genuinely increased.

Every commanded joint target is clamped and validated against the simulation
limit contract in ``sim_contract.py`` (which mirrors
``docs/reference/OPENARM_JOINT_LIMITS.md``). Nothing here moves a real robot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

# Reuse the proven scene definition and the oracle/contract logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_scene import _isaac_paths, _openarm_robot_cfg, _shape_spawn, build_scene_cls  # noqa: E402
from oracle_policy import build_oracle_steps, jitter_object_poses  # noqa: E402
from sim_contract import (  # noqa: E402
    CONFIG_DIR,
    JOINT_NAMES,
    REPO_ROOT,
    clamp_joint_targets,
    gripper_deg_to_sim_finger_m,
    load_yaml_config,
    validate_joint_targets,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "scene_openarm_four_objects.yaml"))
    parser.add_argument("--episodes-per-target", type=int, default=3)
    parser.add_argument("--randomize", action="store_true", help="jitter object poses per episode")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--settle-steps", type=int, default=40, help="physics steps to let objects rest before acting")
    parser.add_argument("--phase-steps", type=int, default=45, help="physics steps per oracle phase")
    parser.add_argument("--hold-steps", type=int, default=70, help="extra steps for the close/lift phases")
    parser.add_argument("--lift-threshold-m", type=float, default=0.04, help="target rise that counts as a pick")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", default="synthetic_smolvla/reports/physics_rollout_manifest.jsonl")
    parser.add_argument("--output", default="synthetic_smolvla/reports/physics_rollout_eval.md")
    return parser


def _resolve(path: str) -> Path:
    # Resolve relative to the repo root, not cwd: the Isaac launcher chdirs into
    # the IsaacLab tree, so cwd-relative paths would land outside this project.
    resolved = Path(path)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    active_side = config["scene"].get("active_arm", "right")
    object_names = [obj["name"] for obj in config["objects"]]
    instruction_for = {obj["name"]: obj["instruction"] for obj in config["objects"]}

    _isaac_paths()
    from isaaclab.app import AppLauncher

    print("[rollout] launching Isaac app", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415

    scene_cls = build_scene_cls(
        config,
        sim_utils=sim_utils,
        AssetBaseCfg=AssetBaseCfg,
        RigidObjectCfg=RigidObjectCfg,
        CameraCfg=CameraCfg,
        InteractiveSceneCfg=InteractiveSceneCfg,
        configclass=configclass,
    )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    sim_dt = sim.get_physics_dt()

    robot = scene["robot"]
    arm_ids, _ = robot.find_joints([f"openarm_{active_side}_joint{i}" for i in range(1, 8)], preserve_order=True)
    finger_ids, finger_names = robot.find_joints(f"openarm_{active_side}_finger_joint.*")
    print(f"[rollout] arm joint ids={arm_ids} finger joints={finger_names}", file=sys.stderr, flush=True)

    def step_n(n: int) -> None:
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    def apply_phase(joints_deg: dict[str, float], gripper_deg: float, n_steps: int) -> None:
        # Clamp + validate every commanded target against the limit contract.
        clamped = clamp_joint_targets(active_side, joints_deg)
        validate_joint_targets(active_side, {**clamped, "gripper": gripper_deg})
        arm_target = torch.tensor(
            [[math.radians(clamped[name]) for name in JOINT_NAMES]],
            device=robot.device,
            dtype=torch.float32,
        )
        finger_m = gripper_deg_to_sim_finger_m(gripper_deg)
        finger_target = torch.full((1, len(finger_ids)), float(finger_m), device=robot.device, dtype=torch.float32)
        for _ in range(n_steps):
            robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
            robot.set_joint_position_target(finger_target, joint_ids=finger_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    def object_z(name: str) -> float:
        return float(scene[name].data.root_pos_w[0, 2].item())

    def set_object_poses(poses_m: dict[str, list[float]]) -> None:
        origin = scene.env_origins[0]
        for name, pose in poses_m.items():
            asset = scene[name]
            root = asset.data.default_root_state.clone()
            root[0, 0] = float(pose[0]) + float(origin[0])
            root[0, 1] = float(pose[1]) + float(origin[1])
            root[0, 2] = float(pose[2]) + float(origin[2])
            root[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
            root[0, 7:] = 0.0
            asset.write_root_pose_to_sim(root[:, :7])
            asset.write_root_velocity_to_sim(root[:, 7:])

    records: list[dict] = []
    for target in object_names:
        steps = build_oracle_steps(instruction_for[target], arm_side=active_side)
        for ep in range(args.episodes_per_target):
            seed = args.seed + len(records)
            poses = jitter_object_poses(seed, enabled=args.randomize)

            # Reset robot to its default pose and re-place objects, then settle.
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            set_object_poses(poses)
            step_n(args.settle_steps)

            baseline = {name: object_z(name) for name in object_names}

            limit_exceeded = False
            try:
                for phase in steps:
                    n = args.hold_steps if phase.name in ("close_gripper", "lift_target") else args.phase_steps
                    apply_phase(phase.joint_targets_deg, phase.gripper_target_deg, n)
            except Exception as exc:  # contract violation or sim error
                limit_exceeded = "limit contract" in str(exc).lower()
                print(f"[rollout] phase error: {exc}", file=sys.stderr, flush=True)

            final = {name: object_z(name) for name in object_names}
            rises = {name: final[name] - baseline[name] for name in object_names}
            target_rise = rises[target]
            success = bool(target_rise > args.lift_threshold_m)
            wrong = sorted(
                name for name in object_names
                if name != target and rises[name] > args.lift_threshold_m
            )

            records.append({
                "schema_version": "openarm_synth_physics_rollout_v1",
                "source": "synthetic_smolvla.rollout_oracle_physics",
                "episode_index": len(records),
                "instruction": instruction_for[target],
                "target_object": target,
                "arm_side": active_side,
                "randomized": bool(args.randomize),
                "object_poses_m": poses,
                "baseline_z_m": {k: round(v, 5) for k, v in baseline.items()},
                "final_z_m": {k: round(v, 5) for k, v in final.items()},
                "target_rise_m": round(target_rise, 5),
                "lift_threshold_m": args.lift_threshold_m,
                "success_label": success,
                "wrong_object_lifted": bool(wrong),
                "wrong_objects": wrong,
                "limit_exceeded": bool(limit_exceeded),
            })
            print(
                f"[rollout] {target} ep{ep}: rise={target_rise:+.4f}m success={success} wrong={wrong}",
                file=sys.stderr,
                flush=True,
            )

    # Camera sanity (proves frames render during rollouts).
    try:
        camera_shape = list(scene["camera"].data.output["rgb"].shape)
    except Exception:
        camera_shape = None

    manifest_path = _resolve(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    total = len(records)
    success = sum(r["success_label"] for r in records)
    wrong = sum(r["wrong_object_lifted"] for r in records)
    limit = sum(r["limit_exceeded"] for r in records)
    by_target = Counter(r["target_object"] for r in records)
    success_by_target = Counter(r["target_object"] for r in records if r["success_label"])

    def rate(c: int) -> float:
        return 0.0 if total == 0 else c / total

    lines = [
        "# Synthetic SmolVLA Physics-Rollout Evaluation",
        "",
        "Measured by stepping Isaac Lab physics and checking the target object's",
        f"height rise (threshold {args.lift_threshold_m:.3f} m). Joint targets are",
        "clamped/validated against the OpenArm simulation limit contract.",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Episodes | {total} | 1.000 |",
        f"| Success (real lift) | {success} | {rate(success):.3f} |",
        f"| Wrong object lifted | {wrong} | {rate(wrong):.3f} |",
        f"| Limit exceeded | {limit} | {rate(limit):.3f} |",
        "",
        "## Success By Target",
        "",
        "| Target | Success | Episodes |",
        "|---|---:|---:|",
    ]
    for target in object_names:
        lines.append(f"| {target} | {success_by_target[target]} | {by_target[target]} |")
    lines.extend([
        "",
        "## Notes",
        "",
        f"- Camera RGB shape during rollout: `{camera_shape}`.",
        "- This is a *measured* label from physics, not the hard-coded oracle label.",
        f"- Settle {args.settle_steps} / phase {args.phase_steps} / hold {args.hold_steps} steps at dt={sim_dt}.",
    ])
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "episodes": total,
        "success": success,
        "success_rate": rate(success),
        "wrong_object": wrong,
        "limit_exceeded": limit,
        "manifest": str(manifest_path),
        "report": str(output_path),
        "camera_rgb_shape": camera_shape,
    }, indent=2), flush=True)

    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
