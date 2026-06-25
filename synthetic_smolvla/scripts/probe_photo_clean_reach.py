#!/usr/bin/env python3
"""Reach probe for the photo-clean LEFT-arm scene.

Loads the photo-matched scene config, drives the LEFT arm with differential IK
to each object's above/descend grasp target, and reports the IK position error
so we can pick a collision-free init pose that reaches every object.

Diagnostic only; headless; no real robot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from sim_contract import JOINT_NAMES, REPO_ROOT, SAFE_ARM_LIMITS_DEG, load_yaml_config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="synthetic_smolvla/configs/scene_openarm_real_photo_left_centered_clean_v1.yaml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ik-iters", type=int, default=160)
    p.add_argument("--above-offset-m", type=float, default=0.12)
    p.add_argument("--grasp-z-offset-m", type=float, default=0.0)
    p.add_argument("--clamp-safe-limits", action="store_true", help="clamp IK updates to the collector safe limits")
    p.add_argument(
        "--reset-arm-deg",
        default=None,
        help="optional comma-separated 7-DOF active-arm reset pose override for diagnostics",
    )
    p.add_argument("--out", default="synthetic_smolvla/reports/photo_clean_reach_probe.json")
    args = p.parse_args()

    config = load_yaml_config(args.config)
    side = config["scene"].get("active_arm", "left")

    # Isaac path bootstrap (mirror make_scene).
    import importlib
    mk = importlib.import_module("make_scene")
    mk._isaac_paths()  # noqa: SLF001

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app

    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415
    from isaaclab.scene import InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: PLC0415
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: PLC0415

    scene_cls = mk.build_scene_cls(
        config, sim_utils=sim_utils, AssetBaseCfg=AssetBaseCfg, RigidObjectCfg=RigidObjectCfg,
        CameraCfg=CameraCfg, InteractiveSceneCfg=InteractiveSceneCfg, configclass=configclass,
    )
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    arm_joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    arm_ids = [joint_names.index(n) for n in arm_joint_names]
    arm_lo = torch.tensor([math.radians(SAFE_ARM_LIMITS_DEG[side][j][0]) for j in JOINT_NAMES], device=args.device)
    arm_hi = torch.tensor([math.radians(SAFE_ARM_LIMITS_DEG[side][j][1]) for j in JOINT_NAMES], device=args.device)
    reset_override = None
    if args.reset_arm_deg:
        values = [float(item.strip()) for item in args.reset_arm_deg.split(",") if item.strip()]
        if len(values) != 7:
            raise SystemExit("--reset-arm-deg expects exactly 7 comma-separated values")
        reset_override = torch.tensor([math.radians(v) for v in values], device=args.device, dtype=torch.float32)
    # End-effector body = the hand/last link of the active side.
    side_bodies = [b for b in body_names if f"openarm_{side}" in b]
    ee_candidates = [b for b in side_bodies if any(k in b.lower() for k in ("hand", "tcp", "tool", "link7", "_7"))]
    ee_body = ee_candidates[-1] if ee_candidates else side_bodies[-1]
    ee_idx = body_names.index(ee_body)
    print(f"[probe] side={side} ee_body={ee_body}", flush=True)
    print(f"[probe] base pos_w={[round(v,4) for v in robot.data.root_pos_w[0].tolist()]}", flush=True)
    print(f"[probe] clamp_safe_limits={args.clamp_safe_limits} reset_override_deg={args.reset_arm_deg}", flush=True)

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=1, device=args.device)
    jac_idx = ee_idx - 1 if robot.is_fixed_base else ee_idx

    def ik_reach(target_w):
        for _ in range(args.ik_iters):
            ee_pos_w = robot.data.body_pose_w[:, ee_idx, 0:3]
            root_pos_w = robot.data.root_pose_w[:, 0:3]
            root_quat_w = robot.data.root_pose_w[:, 3:7]
            ee_quat_w = robot.data.body_pose_w[:, ee_idx, 3:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
            tgt_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, torch.tensor([target_w], device=args.device, dtype=torch.float32))
            ik.set_command(tgt_b, ee_pos_b, ee_quat_b)
            jac = robot.root_physx_view.get_jacobians()[:, jac_idx, 0:3, arm_ids]
            jpos = robot.data.joint_pos[:, arm_ids]
            jdes = ik.compute(ee_pos_b, robot.data.body_pose_w[:, ee_idx, 3:7], jac, jpos)
            if args.clamp_safe_limits:
                jdes = torch.clamp(jdes, arm_lo, arm_hi)
            robot.set_joint_position_target(jdes, joint_ids=arm_ids)
            robot.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
        ee_pos_w = robot.data.body_pose_w[0, ee_idx, 0:3].tolist()
        err = math.dist(ee_pos_w, target_w)
        return [round(v, 4) for v in ee_pos_w], round(err, 4)

    def reset_to_init():
        robot.reset()
        if reset_override is not None:
            pos = robot.data.default_joint_pos.clone()
            vel = robot.data.default_joint_vel.clone()
            pos[:, arm_ids] = reset_override.unsqueeze(0).expand(pos.shape[0], -1)
            vel[:, arm_ids] = 0.0
            robot.write_joint_state_to_sim(pos, vel)
            robot.set_joint_position_target(pos[:, arm_ids], joint_ids=arm_ids)
        sim.step(render=False)

    results = {}
    for obj in config["objects"]:
        name = obj["name"]
        sp = obj["spawn_pose_m"]
        gz = sp[2] + args.grasp_z_offset_m
        above = [sp[0], sp[1], gz + args.above_offset_m]
        descend = [sp[0], sp[1], gz]
        # reset to init pose between objects
        reset_to_init()
        ee_a, err_a = ik_reach(above)
        above_joints_deg = [round(float(v) * 180.0 / math.pi, 3) for v in robot.data.joint_pos[0, arm_ids].tolist()]
        ee_d, err_d = ik_reach(descend)
        descend_joints_deg = [round(float(v) * 180.0 / math.pi, 3) for v in robot.data.joint_pos[0, arm_ids].tolist()]
        reach_ok = err_a < 0.03 and err_d < 0.03
        results[name] = {"spawn": sp, "above_err": err_a, "descend_err": err_d, "reachable": reach_ok,
                         "ee_above": ee_a, "ee_descend": ee_d,
                         "above_joints_deg": above_joints_deg,
                         "descend_joints_deg": descend_joints_deg}
        print(f"[probe] {name} above_joints_deg={above_joints_deg}", flush=True)
        print(f"[probe] {name} descend_joints_deg={descend_joints_deg}", flush=True)
        print(f"[probe] {name}: above_err={err_a} descend_err={err_d} reachable={reach_ok}", flush=True)

    out = REPO_ROOT / args.out
    out.write_text(json.dumps({
        "ok": True,
        "ee_body": ee_body,
        "clamp_safe_limits": bool(args.clamp_safe_limits),
        "reset_override_deg": args.reset_arm_deg,
        "results": results,
    }, indent=2) + "\n")
    print("[probe] wrote", out, flush=True)
    print(json.dumps({"reachable": {k: v["reachable"] for k, v in results.items()}}, indent=2), flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
