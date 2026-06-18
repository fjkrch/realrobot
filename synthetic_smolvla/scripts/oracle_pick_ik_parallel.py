#!/usr/bin/env python3
"""Parallel (multi-env) IK oracle: pick the target object across many Isaac envs
at once, record per-keyframe trajectories, and label each episode by the REAL
measured lift. Built to generate a large success-filtered SmolVLA dataset fast.

It runs the same approach->descend->close->lift task-space plan as
``oracle_pick_ik.py`` but vectorized over N parallel envs (each env has its own
target object), so 4000 episodes take minutes instead of hours. Every commanded
joint target is clamped to the OpenArm simulation limit contract. Output records
use the same ``steps`` schema as ``collect_oracle_demos.py`` so the existing
``dataset_export.export_lerobot_dataset`` can turn the success-filtered manifest
straight into a LeRobot/SmolVLA dataset. Simulation only; no real robot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_scene import _isaac_paths, _openarm_robot_cfg, _shape_spawn  # noqa: E402
from sim_contract import (  # noqa: E402
    CONFIG_DIR,
    JOINT_NAMES,
    REPO_ROOT,
    SAFE_ARM_LIMITS_DEG,
    gripper_deg_to_sim_finger_m,
    load_yaml_config,
    sim_finger_m_to_gripper_deg,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(CONFIG_DIR / "scene_openarm_four_objects.yaml"))
    p.add_argument("--num-envs", type=int, default=500)
    p.add_argument("--rounds", type=int, default=8, help="num_envs * rounds = total episodes")
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--randomize", action="store_true", default=True)
    p.add_argument("--no-randomize", dest="randomize", action="store_false")
    p.add_argument("--settle-steps", type=int, default=40)
    p.add_argument("--ik-iters", type=int, default=60)
    p.add_argument("--substeps", type=int, default=2)
    p.add_argument("--grasp-steps", type=int, default=50)
    p.add_argument("--above-offset-m", type=float, default=0.12)
    p.add_argument("--lift-offset-m", type=float, default=0.15)
    p.add_argument("--lift-threshold-m", type=float, default=0.04)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--manifest", default="synthetic_smolvla/reports/oracle_parallel_all.jsonl")
    p.add_argument("--success-manifest", default="synthetic_smolvla/reports/oracle_parallel_success.jsonl")
    p.add_argument("--output", default="synthetic_smolvla/reports/oracle_parallel_eval.md")
    return p


def _resolve(path: str) -> Path:
    rp = Path(path)
    return rp if rp.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    side = config["scene"].get("active_arm", "right")
    objs = config["objects"]
    obj_names = [o["name"] for o in objs]
    instruction_for = {o["name"]: o["instruction"] for o in objs}
    spawn_for = {o["name"]: [float(v) for v in o["spawn_pose_m"]] for o in objs}
    bounds = config["scene"]["workspace_bounds_m"]
    n_obj = len(obj_names)

    _isaac_paths()
    from isaaclab.app import AppLauncher

    print(f"[par-oracle] launching Isaac (num_envs={args.num_envs}, rounds={args.rounds})", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415
    from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: PLC0415
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: PLC0415

    # Scene (no camera, for speed): ground, light, table, robot, four objects.
    table = config["scene"]["table"]
    attrs: dict = {"__annotations__": {}}
    attrs["ground"] = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    attrs["dome_light"] = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    attrs["table"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(v) for v in table["size_m"]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.50, 0.44), roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in table["pose_m"])),
    )
    attrs["robot"] = _openarm_robot_cfg(config)
    attrs["__annotations__"].update(
        {"ground": AssetBaseCfg, "dome_light": AssetBaseCfg, "table": AssetBaseCfg, "robot": type(attrs["robot"])}
    )
    for o in objs:
        attrs[o["name"]] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{o['name']}",
            spawn=_shape_spawn(sim_utils, o),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in o["spawn_pose_m"]), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        attrs["__annotations__"][o["name"]] = RigidObjectCfg
    scene_cls = configclass(type("ParallelOpenArmSceneCfg", (InteractiveSceneCfg,), attrs))

    N = args.num_envs
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(scene_cls(num_envs=N, env_spacing=2.0))
    sim.reset()
    scene.reset()
    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]

    arm_ids, _ = robot.find_joints([f"openarm_{side}_joint{i}" for i in range(1, 8)], preserve_order=True)
    finger_ids, _ = robot.find_joints(f"openarm_{side}_finger_joint.*")
    tcp_idx = robot.find_bodies(f"openarm_{side}_ee_tcp")[0][0]
    ee_jacobi_idx = tcp_idx - 1 if robot.is_fixed_base else tcp_idx
    ent = SceneEntityCfg("robot", joint_names=[f"openarm_{side}_joint{i}" for i in range(1, 8)], body_names=[f"openarm_{side}_ee_tcp"])
    ent.resolve(scene)

    device = robot.device
    arm_lo = torch.tensor([math.radians(SAFE_ARM_LIMITS_DEG[side][j][0]) for j in JOINT_NAMES], device=device)
    arm_hi = torch.tensor([math.radians(SAFE_ARM_LIMITS_DEG[side][j][1]) for j in JOINT_NAMES], device=device)
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(N, 1)

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
        num_envs=N, device=device,
    )

    def step_n(n: int) -> None:
        for _ in range(n):
            scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)

    def set_gripper(finger_m: float) -> None:
        tgt = torch.full((N, len(finger_ids)), float(finger_m), device=device)
        robot.set_joint_position_target(tgt, joint_ids=finger_ids)

    def obj_pos_w() -> torch.Tensor:  # [N, n_obj, 3]
        return torch.stack([scene[name].data.root_pos_w for name in obj_names], dim=1)

    def servo(target_w: torch.Tensor, finger_m: float, iters: int) -> torch.Tensor:
        """Vectorized position IK toward per-env target_w [N,3]. Returns clamp-hit [N] bool."""
        ik.reset()
        rp = robot.data.root_pose_w
        tpos_b, _ = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], target_w, ident)
        ee_w0 = robot.data.body_pose_w[:, tcp_idx]
        _, ee_quat_b0 = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], ee_w0[:, 0:3], ee_w0[:, 3:7])
        ik.set_command(tpos_b, ee_quat=ee_quat_b0)
        clamp_hit = torch.zeros(N, dtype=torch.bool, device=device)
        for _ in range(iters):
            jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, ent.joint_ids]
            ee_w = robot.data.body_pose_w[:, tcp_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
            jpos = robot.data.joint_pos[:, ent.joint_ids]
            jdes = ik.compute(ee_pos_b, ee_quat_b, jac, jpos)
            jclamped = torch.clamp(jdes, arm_lo, arm_hi)
            clamp_hit |= (jdes != jclamped).any(dim=-1)
            robot.set_joint_position_target(jclamped, joint_ids=arm_ids)
            set_gripper(finger_m)
            step_n(args.substeps)
        return clamp_hit

    def record_step(name: str) -> dict:
        """Per-env right-arm joints (deg) + gripper at the current state -> step dict list."""
        jpos = robot.data.joint_pos[:, ent.joint_ids]  # [N,7] rad
        finger = robot.data.joint_pos[:, finger_ids].mean(dim=-1)  # [N]
        jdeg = (jpos * 180.0 / math.pi).tolist()
        gdeg = [sim_finger_m_to_gripper_deg(float(f)) for f in finger.tolist()]
        fm = finger.tolist()
        steps = []
        for e in range(N):
            steps.append({
                "name": name,
                "action": {
                    "joint_targets_deg": {JOINT_NAMES[k]: round(jdeg[e][k], 3) for k in range(7)},
                    "gripper_target_deg": round(gdeg[e], 3),
                    "sim_finger_target_m": round(fm[e], 5),
                },
            })
        return steps

    open_m = gripper_deg_to_sim_finger_m(-65.0)
    closed_m = gripper_deg_to_sim_finger_m(0.0)

    all_records: list[dict] = []
    rng = torch.Generator(device="cpu"); rng.manual_seed(args.seed)
    target_cycle = (torch.arange(N) % n_obj)  # balanced targets per env

    for rnd in range(args.rounds):
        # Reset robot + place objects (per-env jitter), then settle.
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        origins = scene.env_origins  # [N,3]
        ep_obj_local = {}
        for name in obj_names:
            asset = scene[name]
            root = asset.data.default_root_state.clone()
            base = torch.tensor(spawn_for[name], device=device)
            jit = torch.zeros((N, 3), device=device)
            if args.randomize:
                jx = torch.empty(N).uniform_(-0.015, 0.015, generator=rng)
                jy = torch.empty(N).uniform_(-0.008, 0.008, generator=rng)
                jit[:, 0] = jx.to(device); jit[:, 1] = jy.to(device)
            local = base.unsqueeze(0) + jit
            local[:, 0] = torch.clamp(local[:, 0], bounds["x"][0], bounds["x"][1])
            local[:, 1] = torch.clamp(local[:, 1], bounds["y"][0], bounds["y"][1])
            ep_obj_local[name] = local
            root[:, 0:3] = local + origins
            root[:, 3:7] = ident
            root[:, 7:] = 0.0
            asset.write_root_pose_to_sim(root[:, :7])
            asset.write_root_velocity_to_sim(root[:, 7:])
        set_gripper(open_m)
        step_n(args.settle_steps)

        target_idx = target_cycle.to(device)
        ow = obj_pos_w()                          # [N,n_obj,3]
        baseline = ow[:, :, 2].clone()            # [N,n_obj]
        tw = ow.gather(1, target_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)  # [N,3]
        ox, oy = tw[:, 0], tw[:, 1]
        grasp_z = tw[:, 2]

        steps_per_env = [[] for _ in range(N)]

        def add(name):
            for e, s in enumerate(record_step(name)):
                steps_per_env[e].append(s)

        add("start")
        clamp = torch.zeros(N, dtype=torch.bool, device=device)
        above = torch.stack([ox, oy, grasp_z + args.above_offset_m], dim=-1)
        clamp |= servo(above, open_m, args.ik_iters); add("move_above_target")
        descend = torch.stack([ox, oy, grasp_z], dim=-1)
        clamp |= servo(descend, open_m, args.ik_iters); add("lower_to_target")
        set_gripper(closed_m); step_n(args.grasp_steps); add("close_gripper")
        lift = torch.stack([ox, oy, grasp_z + args.lift_offset_m], dim=-1)
        clamp |= servo(lift, closed_m, args.ik_iters); add("lift_target")

        final = obj_pos_w()[:, :, 2]
        rises = final - baseline                  # [N,n_obj]
        target_rise = rises.gather(1, target_idx.view(-1, 1)).squeeze(1)
        success = target_rise > args.lift_threshold_m
        wrong_mask = torch.ones_like(rises); wrong_mask.scatter_(1, target_idx.view(-1, 1), 0.0)
        wrong_any = ((rises * wrong_mask) > args.lift_threshold_m).any(dim=-1)

        for e in range(N):
            tname = obj_names[int(target_idx[e])]
            poses = {name: [round(float(v), 4) for v in ep_obj_local[name][e].tolist()] for name in obj_names}
            all_records.append({
                "schema_version": "openarm_synth_oracle_parallel_v1",
                "source": "synthetic_smolvla.oracle_pick_ik_parallel",
                "episode_index": len(all_records),
                "instruction": instruction_for[tname],
                "target_object": tname,
                "arm_side": side,
                "randomized": bool(args.randomize),
                "all_objects_visible": True,
                "visible_objects": list(obj_names),
                "object_poses_m": poses,
                "steps": steps_per_env[e],
                "target_rise_m": round(float(target_rise[e]), 5),
                "success_label": bool(success[e].item()),
                "wrong_object_lifted": bool(wrong_any[e].item()),
                "limit_exceeded": bool(clamp[e].item()),
            })
        done = len(all_records)
        sr = float(success.float().mean())
        print(f"[par-oracle] round {rnd+1}/{args.rounds}: {done} episodes, round success={sr:.3f}", file=sys.stderr, flush=True)

    # Write manifests.
    total = len(all_records)
    succ = [r for r in all_records if r["success_label"]]
    man = _resolve(args.manifest); man.parent.mkdir(parents=True, exist_ok=True)
    with man.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    sman = _resolve(args.success_manifest)
    with sman.open("w", encoding="utf-8") as fh:
        for r in succ:
            fh.write(json.dumps(r) + "\n")

    n_succ = len(succ)
    n_wrong = sum(r["wrong_object_lifted"] for r in all_records)
    by_t = Counter(r["target_object"] for r in all_records)
    succ_t = Counter(r["target_object"] for r in succ)

    def rate(c): return 0.0 if total == 0 else c / total
    lines = [
        "# Parallel IK Oracle (4-object, target-conditioned) — measured in physics",
        "",
        f"Ran {total} episodes across {N} parallel envs x {args.rounds} rounds.",
        "Success = measured target lift; dataset keeps only successful episodes.",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Episodes | {total} | 1.000 |",
        f"| Success (kept for VLA) | {n_succ} | {rate(n_succ):.3f} |",
        f"| Wrong object lifted | {n_wrong} | {rate(n_wrong):.3f} |",
        "",
        "## Success by target",
        "",
        "| Target | Success | Episodes |",
        "|---|---:|---:|",
    ]
    for t in obj_names:
        lines.append(f"| {t} | {succ_t[t]} | {by_t[t]} |")
    lines += ["", f"- All manifest: `{args.manifest}`", f"- Success-filtered manifest: `{args.success_manifest}`"]
    out = _resolve(args.output); out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True, "episodes": total, "success": n_succ, "success_rate": rate(n_succ),
        "wrong_object": n_wrong, "manifest": str(man), "success_manifest": str(sman), "report": str(out),
    }, indent=2), flush=True)

    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
