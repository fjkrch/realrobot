#!/usr/bin/env python3
"""Probe OpenArm kinematics: list bodies/joints, find the right end-effector,
measure where the gripper actually is, and test whether differential IK can
drive it to the table objects. Diagnostic only; no real robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_scene import _isaac_paths, build_scene_cls  # noqa: E402
from oracle_policy import DEFAULT_OBJECT_POSES_M  # noqa: E402
from sim_contract import REPO_ROOT, load_yaml_config  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="synthetic_smolvla/configs/scene_openarm_four_objects.yaml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ik-iters", type=int, default=120)
    p.add_argument("--substeps", type=int, default=3)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg_path = args.config
    config = load_yaml_config(cfg_path if Path(cfg_path).is_absolute() else REPO_ROOT / cfg_path)
    side = config["scene"].get("active_arm", "right")

    _isaac_paths()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app

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
        config, sim_utils=sim_utils, AssetBaseCfg=AssetBaseCfg, RigidObjectCfg=RigidObjectCfg,
        CameraCfg=CameraCfg, InteractiveSceneCfg=InteractiveSceneCfg, configclass=configclass,
    )
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    dt = sim.get_physics_dt()
    robot = scene["robot"]

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    print("[probe] is_fixed_base:", robot.is_fixed_base, flush=True)
    print("[probe] base pos_w:", [round(v, 4) for v in robot.data.root_pos_w[0].tolist()], flush=True)
    print("[probe] body_names:", body_names, flush=True)
    print("[probe] joint_names:", joint_names, flush=True)

    # Choose an end-effector body for the active side: prefer a hand/gripper link.
    side_bodies = [b for b in body_names if side in b]
    ee_candidates = [b for b in side_bodies if any(k in b.lower() for k in ("hand", "gripper", "tcp", "tool"))]
    if not ee_candidates:
        # fall back to the highest-numbered link on this side
        link_bodies = [b for b in side_bodies if "link" in b.lower() or "joint" in b.lower()]
        ee_candidates = [sorted(side_bodies)[-1]] if side_bodies else [body_names[-1]]
    ee_body = ee_candidates[-1]
    print("[probe] chosen ee_body:", ee_body, "| side bodies:", side_bodies, flush=True)

    # settle
    for _ in range(40):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    ent = SceneEntityCfg("robot", joint_names=[f"openarm_{side}_joint{i}" for i in range(1, 8)], body_names=[ee_body])
    ent.resolve(scene)
    ee_jacobi_idx = ent.body_ids[0] - 1 if robot.is_fixed_base else ent.body_ids[0]

    def ee_pose_w():
        p = robot.data.body_pose_w[:, ent.body_ids[0]]
        return p[0, 0:3].tolist(), p[0, 3:7].tolist()

    pos, quat = ee_pose_w()
    print("[probe] EE world pos @reset:", [round(v, 4) for v in pos], flush=True)
    print("[probe] objects (world approx):", {k: [round(c, 3) for c in v] for k, v in DEFAULT_OBJECT_POSES_M.items()}, flush=True)

    # Position-only IK to MAP the right arm's reachable region over the table.
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=robot.device,
    )

    def ik_reach(target_w_xyz, iters, substeps):
        ik.reset()
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        for _ in range(15):
            robot.set_joint_position_target(robot.data.default_joint_pos)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)
        target_w = torch.tensor([target_w_xyz], device=robot.device, dtype=torch.float32)
        rp0 = robot.data.root_pose_w
        ee_w0 = robot.data.body_pose_w[:, ent.body_ids[0]]
        tpos_b, _ = subtract_frame_transforms(rp0[:, 0:3], rp0[:, 3:7], target_w,
                                              torch.tensor([[1.0, 0, 0, 0]], device=robot.device))
        _, ee_quat_b0 = subtract_frame_transforms(rp0[:, 0:3], rp0[:, 3:7], ee_w0[:, 0:3], ee_w0[:, 3:7])
        ik.set_command(tpos_b, ee_quat=ee_quat_b0)
        for _ in range(iters):
            jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, ent.joint_ids]
            ee_w = robot.data.body_pose_w[:, ent.body_ids[0]]
            rp = robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
            jpos = robot.data.joint_pos[:, ent.joint_ids]
            jdes = ik.compute(ee_pos_b, ee_quat_b, jac, jpos)
            robot.set_joint_position_target(jdes, joint_ids=ent.joint_ids)
            for _ in range(substeps):
                scene.write_data_to_sim(); sim.step(); scene.update(dt)
        fp, _ = ee_pose_w()
        err = sum((fp[i] - target_w_xyz[i]) ** 2 for i in range(3)) ** 0.5
        return fp, err

    # Map reachability over a grid at grasp height (z=0.45) and pre-grasp (z=0.55).
    grid = {}
    for z in (0.45, 0.55):
        for x in (0.30, 0.38, 0.46):
            for y in (-0.28, -0.18, -0.08, 0.02, 0.12):
                fp, err = ik_reach([x, y, z], args.ik_iters, args.substeps)
                key = f"x{x:.2f}_y{y:+.2f}_z{z:.2f}"
                grid[key] = {"ee": [round(v, 3) for v in fp], "err": round(err, 4), "reachable": err < 0.03}
                print(f"[probe] grid {key}: err={err:.4f} reachable={err < 0.03}", flush=True)

    # Also test the current object positions at grasp height.
    results = {}
    for name, opose in DEFAULT_OBJECT_POSES_M.items():
        fp, err = ik_reach([opose[0], opose[1], 0.45], args.ik_iters, args.substeps)
        results[name] = {"target": [opose[0], opose[1], 0.45], "ee_final": [round(v, 4) for v in fp], "pos_err_m": round(err, 4)}
        print(f"[probe] IK to {name}@z0.45: err={err:.4f} m  ee={[round(v,3) for v in fp]}", flush=True)

    out = {
        "is_fixed_base": bool(robot.is_fixed_base),
        "base_pos_w": [round(v, 4) for v in robot.data.root_pos_w[0].tolist()],
        "ee_body": ee_body,
        "ee_pos_reset_w": [round(v, 4) for v in pos],
        "body_names": body_names,
        "joint_names": joint_names,
        "ik_reach_objects": results,
        "reach_grid": grid,
    }
    outpath = REPO_ROOT / "synthetic_smolvla/reports/openarm_reach_probe.json"
    outpath.write_text(json.dumps(out, indent=2) + "\n")
    print("[probe] wrote", outpath, flush=True)
    print(json.dumps({"ok": True, "ee_body": ee_body, "reach_err": {k: v["pos_err_m"] for k, v in results.items()}}, indent=2), flush=True)
    app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
