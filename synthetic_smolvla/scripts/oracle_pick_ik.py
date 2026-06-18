#!/usr/bin/env python3
"""Closed-loop IK oracle that ACTUALLY picks the requested object in Isaac physics.

Background
----------
The original oracle (``oracle_policy.py`` + ``collect_oracle_demos.py``) is a
scaffold: it emits a fixed list of joint angles and hard-codes
``success_label: True``. When those joint angles are stepped through real
physics (``rollout_oracle_physics.py``) the arm never touches anything and the
measured pick rate is 0% (``reports/logs/physics_rollout.log``): the bimanual
OpenArm was floor-mounted while the table sits at z=0.40, so the objects were
out of reach (differential IK missed the targets by 16-46 cm, see
``reports/openarm_reach_probe.json``).

This oracle replaces the open-loop joint script with a task-space state machine
driven by differential inverse kinematics, modelled on the Franka pick-and-lift
state machine in
``/home/chyanin/IsaacLab/source/hsi_pregrasp_refusal/hsi_pregrasp_refusal/state_machine.py``:

    settle -> approach above object -> descend to object -> close gripper -> lift

Each per-step IK joint solution is clamped and validated against the OpenArm
simulation limit contract (``sim_contract.py`` / ``OPENARM_JOINT_LIMITS.md``).
Success is MEASURED from the target object's true height rise, not asserted.
Nothing here opens a CAN bus or moves a real robot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_scene import _isaac_paths, build_scene_cls  # noqa: E402
from sim_contract import (  # noqa: E402
    CONFIG_DIR,
    JOINT_NAMES,
    REPO_ROOT,
    clamp_joint_targets,
    gripper_deg_to_sim_finger_m,
    load_yaml_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "scene_openarm_four_objects.yaml"))
    parser.add_argument("--episodes-per-target", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--orient-lock", action="store_true", help="EXPERIMENTAL: hold a captured wrist orientation through descend/grasp/lift. Default OFF: position-only IK is the proven mode; the captured-orientation lock regressed grasps to ~0%% (see README 'Optimization attempts').")
    parser.add_argument("--randomize", action="store_true", help="jitter object xy each episode within the reachable pocket")
    parser.add_argument("--jitter-x-m", type=float, default=0.02, help="+/- per-episode x jitter when --randomize")
    parser.add_argument("--jitter-y-m", type=float, default=0.01, help="+/- per-episode y jitter when --randomize (kept small to avoid object overlap)")
    parser.add_argument("--settle-steps", type=int, default=40)
    parser.add_argument("--ik-iters", type=int, default=70, help="IK servo iterations per state-machine phase")
    parser.add_argument("--substeps", type=int, default=2, help="physics steps per IK iteration")
    parser.add_argument("--grasp-steps", type=int, default=60, help="physics steps to hold while closing the gripper")
    parser.add_argument("--above-offset-m", type=float, default=0.12, help="pre-grasp height above the object")
    parser.add_argument("--grasp-offset-m", type=float, default=0.0, help="z offset added to the object centre when grasping")
    parser.add_argument("--lift-offset-m", type=float, default=0.15, help="how far above the object to lift")
    parser.add_argument("--lift-threshold-m", type=float, default=0.04, help="target rise that counts as a pick")
    parser.add_argument("--pos-tolerance-m", type=float, default=0.02, help="EE position tolerance to consider a phase reached")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", default="synthetic_smolvla/reports/oracle_ik_manifest.jsonl")
    parser.add_argument("--output", default="synthetic_smolvla/reports/oracle_ik_eval.md")
    return parser


def _resolve(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    side = config["scene"].get("active_arm", "right")
    object_names = [obj["name"] for obj in config["objects"]]
    instruction_for = {obj["name"]: obj["instruction"] for obj in config["objects"]}
    spawn_for = {obj["name"]: [float(v) for v in obj["spawn_pose_m"]] for obj in config["objects"]}
    bounds = config["scene"]["workspace_bounds_m"]

    def episode_object_poses(ep_seed: int) -> dict[str, list[float]]:
        """Per-episode object poses, optionally jittered inside the reachable pocket.

        x/y are clamped to the scene workspace bounds so jitter never pushes an
        object out of the right arm's reachable region; z (table height) is fixed.
        """
        rng = random.Random(ep_seed)
        poses: dict[str, list[float]] = {}
        for name in object_names:
            sp = list(spawn_for[name])
            if args.randomize:
                sp[0] = min(bounds["x"][1], max(bounds["x"][0], sp[0] + rng.uniform(-args.jitter_x_m, args.jitter_x_m)))
                sp[1] = min(bounds["y"][1], max(bounds["y"][0], sp[1] + rng.uniform(-args.jitter_y_m, args.jitter_y_m)))
            poses[name] = sp
        return poses

    _isaac_paths()
    from isaaclab.app import AppLauncher

    print("[oracle-ik] launching Isaac app", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415
    from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: PLC0415
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: PLC0415

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

    # Pick the end-effector body for the active side (prefer a tcp/hand link).
    body_names = list(robot.body_names)
    side_bodies = [b for b in body_names if side in b]
    ee_candidates = [b for b in side_bodies if any(k in b.lower() for k in ("tcp", "hand", "gripper", "tool"))]
    ee_body = ee_candidates[-1] if ee_candidates else (sorted(side_bodies)[-1] if side_bodies else body_names[-1])
    print(f"[oracle-ik] ee_body={ee_body}", file=sys.stderr, flush=True)

    ent = SceneEntityCfg(
        "robot",
        joint_names=[f"openarm_{side}_joint{i}" for i in range(1, 8)],
        body_names=[ee_body],
    )
    ent.resolve(scene)
    ee_jacobi_idx = ent.body_ids[0] - 1 if robot.is_fixed_base else ent.body_ids[0]
    finger_ids, _ = robot.find_joints(f"openarm_{side}_finger_joint.*")

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=robot.device,
    )
    # Pose-mode controller used once a grasp orientation is captured, so the wrist
    # is held in a consistent top-down pose through descend/grasp/lift instead of
    # drifting (orientation drift is what makes position-only grasps flaky and
    # position-dependent: see reports/logs/oracle_ik_closer_check.log).
    ik_pose = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=robot.device,
    )

    def step_n(n: int) -> None:
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    def ee_pos_w() -> torch.Tensor:
        return robot.data.body_pose_w[:, ent.body_ids[0], 0:3]

    def object_z(name: str) -> float:
        return float(scene[name].data.root_pos_w[0, 2].item())

    def object_xy_w(name: str):
        p = scene[name].data.root_pos_w[0]
        return float(p[0].item()), float(p[1].item())

    def set_gripper(gripper_deg: float) -> None:
        finger_m = gripper_deg_to_sim_finger_m(gripper_deg)
        target = torch.full((1, len(finger_ids)), float(finger_m), device=robot.device, dtype=torch.float32)
        robot.set_joint_position_target(target, joint_ids=finger_ids)

    def ee_quat_b_now() -> torch.Tensor:
        """Current EE orientation in the base frame (xyzw->wxyz as IsaacLab uses)."""
        rp = robot.data.root_pose_w
        ee_w = robot.data.body_pose_w[:, ent.body_ids[0]]
        _, ee_quat_b = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
        return ee_quat_b

    def servo_to(target_w_xyz, gripper_deg: float, iters: int, hold_quat_b: torch.Tensor | None = None) -> bool:
        """Drive the EE toward a world position with clamped differential IK.

        If ``hold_quat_b`` is given, pose-mode IK also holds that fixed base-frame
        orientation (a consistent top-down grasp wrist) instead of letting the
        wrist drift. Returns True if any raw IK joint solution had to be clamped
        to the OpenArm limit contract this phase.
        """
        target_w = torch.tensor([list(target_w_xyz)], device=robot.device, dtype=torch.float32)
        rp = robot.data.root_pose_w
        tpos_b, _ = subtract_frame_transforms(
            rp[:, 0:3], rp[:, 3:7], target_w, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=robot.device)
        )
        if hold_quat_b is None:
            ik.reset()
            ik.set_command(tpos_b, ee_quat=ee_quat_b_now())
            controller = ik
        else:
            ik_pose.reset()
            ik_pose.set_command(torch.cat([tpos_b, hold_quat_b], dim=-1))
            controller = ik_pose
        clamp_hit = False
        for _ in range(iters):
            jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, ent.joint_ids]
            ee_w = robot.data.body_pose_w[:, ent.body_ids[0]]
            rpc = robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(rpc[:, 0:3], rpc[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
            jpos = robot.data.joint_pos[:, ent.joint_ids]
            jdes = controller.compute(ee_pos_b, ee_quat_b, jac, jpos)

            # Clamp every commanded joint to the OpenArm simulation limit contract.
            raw_deg = {JOINT_NAMES[i]: math.degrees(float(jdes[0, i].item())) for i in range(7)}
            clamped_deg = clamp_joint_targets(side, raw_deg)
            if any(abs(clamped_deg[n] - raw_deg[n]) > 1e-6 for n in JOINT_NAMES):
                clamp_hit = True
            arm_target = torch.tensor(
                [[math.radians(clamped_deg[n]) for n in JOINT_NAMES]],
                device=robot.device,
                dtype=torch.float32,
            )
            robot.set_joint_position_target(arm_target, joint_ids=ent.joint_ids)
            set_gripper(gripper_deg)
            step_n(args.substeps)
        return clamp_hit

    # Calibrate one consistent grasp wrist orientation from a central reach above
    # the object cluster, then hold it through every descend/grasp/lift so the
    # fingers straddle the object the same way regardless of object x/y.
    grasp_quat_b = None
    if args.orient_lock:
        cx = sum(spawn_for[n][0] for n in object_names) / len(object_names)
        cy = sum(spawn_for[n][1] for n in object_names) / len(object_names)
        cz = sum(spawn_for[n][2] for n in object_names) / len(object_names)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        step_n(args.settle_steps)
        servo_to([cx, cy, cz + args.above_offset_m], -65.0, args.ik_iters)
        grasp_quat_b = ee_quat_b_now().clone()
        print(f"[oracle-ik] grasp wrist quat (base) = {[round(v,4) for v in grasp_quat_b[0].tolist()]}", file=sys.stderr, flush=True)

    records: list[dict] = []
    for target in object_names:
        for ep in range(args.episodes_per_target):
            # Reset robot + objects, then let everything settle.
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            origin = scene.env_origins[0]
            ep_poses = episode_object_poses(args.seed + len(records))
            for name in object_names:
                asset = scene[name]
                root = asset.data.default_root_state.clone()
                sp = ep_poses[name]
                root[0, 0] = sp[0] + float(origin[0])
                root[0, 1] = sp[1] + float(origin[1])
                root[0, 2] = sp[2] + float(origin[2])
                root[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
                root[0, 7:] = 0.0
                asset.write_root_pose_to_sim(root[:, :7])
                asset.write_root_velocity_to_sim(root[:, 7:])
            set_gripper(-65.0)
            step_n(args.settle_steps)

            baseline = {name: object_z(name) for name in object_names}
            ox, oy = object_xy_w(target)
            grasp_z = baseline[target] + args.grasp_offset_m

            clamp_hit = False
            # State machine: above (position-only) -> descend -> close -> lift,
            # holding the calibrated top-down wrist for descend/grasp/lift.
            clamp_hit |= servo_to([ox, oy, grasp_z + args.above_offset_m], -65.0, args.ik_iters)
            clamp_hit |= servo_to([ox, oy, grasp_z], -65.0, args.ik_iters, hold_quat_b=grasp_quat_b)
            set_gripper(0.0)  # close
            step_n(args.grasp_steps)
            clamp_hit |= servo_to([ox, oy, grasp_z + args.lift_offset_m], 0.0, args.ik_iters, hold_quat_b=grasp_quat_b)

            ee_final = [round(float(v), 4) for v in ee_pos_w()[0].tolist()]
            final = {name: object_z(name) for name in object_names}
            rises = {name: final[name] - baseline[name] for name in object_names}
            target_rise = rises[target]
            success = bool(target_rise > args.lift_threshold_m)
            wrong = sorted(
                name for name in object_names
                if name != target and rises[name] > args.lift_threshold_m
            )

            records.append({
                "schema_version": "openarm_synth_oracle_ik_v1",
                "source": "synthetic_smolvla.oracle_pick_ik",
                "episode_index": len(records),
                "instruction": instruction_for[target],
                "target_object": target,
                "arm_side": side,
                "ee_body": ee_body,
                "randomized": bool(args.randomize),
                "object_poses_m": {k: [round(c, 4) for c in v] for k, v in ep_poses.items()},
                "object_target_w": [round(ox, 4), round(oy, 4), round(grasp_z, 4)],
                "ee_final_w": ee_final,
                "baseline_z_m": {k: round(v, 5) for k, v in baseline.items()},
                "final_z_m": {k: round(v, 5) for k, v in final.items()},
                "target_rise_m": round(target_rise, 5),
                "lift_threshold_m": args.lift_threshold_m,
                "success_label": success,
                "wrong_object_lifted": bool(wrong),
                "wrong_objects": wrong,
                "limit_exceeded": bool(clamp_hit),
            })
            print(
                f"[oracle-ik] {target} ep{ep}: rise={target_rise:+.4f}m success={success} "
                f"wrong={wrong} ee_final={ee_final}",
                file=sys.stderr,
                flush=True,
            )

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
        "# Synthetic SmolVLA IK Oracle Evaluation",
        "",
        "Closed-loop differential-IK oracle (approach-above -> descend -> close -> lift).",
        f"Success is the MEASURED target rise above {args.lift_threshold_m:.3f} m in Isaac",
        "physics. Every IK joint solution is clamped to the OpenArm limit contract.",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Episodes | {total} | 1.000 |",
        f"| Success (real lift) | {success} | {rate(success):.3f} |",
        f"| Wrong object lifted | {wrong} | {rate(wrong):.3f} |",
        f"| Limit-clamp engaged | {limit} | {rate(limit):.3f} |",
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
        f"- EE body: `{ee_body}`; position-only DLS IK, {args.ik_iters} iters x {args.substeps} substeps per phase.",
        "- Labels are measured from physics, not the hard-coded scaffold label.",
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
