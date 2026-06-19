#!/usr/bin/env python3
"""Collect a DENSE, Isaac-camera SmolVLA pick-and-lift dataset.

This is the corrected data path called for in
``docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md`` and the success-filtered
dataset audit. It fixes the three structural weaknesses of the old
``openarm_success_filtered_14000`` dataset:

  1. Dense rollouts, not 5 keyframes. Each episode records every control step of
     approach -> descend -> close -> lift -> hold (default 50 steps).
  2. Real Isaac camera RGB at every control step (the actual scene camera tensor,
     captured per env), not the deterministic top-down placeholder. The frame
     therefore shows the arm, gripper, object, contact, and lift MOVING across an
     episode instead of one repeated static image.
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
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_scene import _isaac_paths, build_scene_cls  # noqa: E402
from sim_contract import (  # noqa: E402
    CONFIG_DIR,
    JOINT_NAMES,
    REPO_ROOT,
    SAFE_ARM_LIMITS_DEG,
    gripper_deg_to_sim_finger_m,
    load_yaml_config,
    sim_finger_m_to_gripper_deg,
    validate_scene_config,
)

CAMERA_KEY = "observation.images.camera1"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_NAMES = [*JOINT_NAMES, "gripper"]


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
        "--max-gripper-close-deg",
        type=float,
        default=0.0,
        help="upper gripper command cap; use -3 to prevent fully closed gripper commands",
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
    # Target sampling weights, aligned to config object order [orange,red,green,blue].
    p.add_argument("--target-weights", default="2.5,1,1,1", help="comma weights for object sampling")
    p.add_argument("--drop-limit-exceeded", action="store_true", help="reject episodes that hit the limit clamp")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-keep", type=int, default=0, help="optional cap on kept episodes (0 = no cap)")
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
    config = load_yaml_config(args.config)
    validate_scene_config(config)

    side = config["scene"].get("active_arm", "right")
    objs = config["objects"]
    obj_names = [o["name"] for o in objs]
    instruction_for = {o["name"]: o["instruction"] for o in objs}
    spawn_for = {o["name"]: [float(v) for v in o["spawn_pose_m"]] for o in objs}
    bounds = config["scene"]["workspace_bounds_m"]
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

    weights = [float(w) for w in args.target_weights.split(",")]
    if len(weights) != n_obj:
        raise SystemExit(f"--target-weights needs {n_obj} values aligned to {obj_names}, got {weights}.")

    dataset_root = _resolve(args.dataset_root)
    manifest_path = _resolve(args.manifest)
    report_path = _resolve(args.report)
    sample_dir = _resolve(args.sample_frame_dir) if args.sample_frame_dir else None

    phase_plan = [
        ("approach", args.approach_steps),
        ("descend", args.descend_steps),
        ("close", args.close_steps),
        ("lift", args.lift_steps),
        ("hold", args.hold_steps),
    ]
    episode_len = sum(n for _, n in phase_plan)
    effective_close_deg = min(float(args.grasp_close_deg), float(args.max_gripper_close_deg))
    if effective_close_deg != float(args.grasp_close_deg):
        print(
            f"[dense] capping --grasp-close-deg {args.grasp_close_deg:.3f} to "
            f"{effective_close_deg:.3f}",
            file=sys.stderr,
            flush=True,
        )
    print(f"[dense] episode_len={episode_len} control steps, image={image_size}px, "
          f"keep-success-only, drop_limit={args.drop_limit_exceeded}, "
          f"gripper_close={effective_close_deg:.3f} deg", file=sys.stderr, flush=True)

    _isaac_paths()
    from isaaclab.app import AppLauncher

    print(f"[dense] launching Isaac (num_envs={args.num_envs}, cameras=on)", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
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

    scene_cls = build_scene_cls(
        config,
        sim_utils=sim_utils,
        AssetBaseCfg=AssetBaseCfg,
        RigidObjectCfg=RigidObjectCfg,
        CameraCfg=CameraCfg,
        InteractiveSceneCfg=InteractiveSceneCfg,
        configclass=configclass,
    )

    N = args.num_envs
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    cam_cfg = config["scene"]["camera"]
    sim.set_camera_view(eye=cam_cfg["eye_m"], target=cam_cfg["target_m"])
    scene = InteractiveScene(scene_cls(num_envs=N, env_spacing=3.0))
    sim.reset()
    scene.reset()
    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    camera = scene["camera"]

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
    print(f"[dense] finger bodies for penetration check: {finger_body_names}", file=sys.stderr, flush=True)
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

    open_m = gripper_deg_to_sim_finger_m(-65.0)
    closed_m = gripper_deg_to_sim_finger_m(effective_close_deg)
    object_radius_t = torch.tensor(object_contact_radii, device=device, dtype=torch.float32)

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
            sim.step(render=(render_last and k == n - 1))
            scene.update(sim_dt)

    def set_gripper(finger_m: float) -> None:
        tgt = torch.full((N, len(finger_ids)), float(finger_m), device=device)
        robot.set_joint_position_target(tgt, joint_ids=finger_ids)

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
        tcp_w = robot.data.body_pose_w[:, tcp_idx, 0:3]
        over_table = (
            (tcp_w[:, 0] >= table_x[0])
            & (tcp_w[:, 0] <= table_x[1])
            & (tcp_w[:, 1] >= table_y[0])
            & (tcp_w[:, 1] <= table_y[1])
        )
        clearance = tcp_w[:, 2] - table_top_z
        return over_table & (clearance < float(args.table_collision_margin_m)), clearance

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
        """Current Isaac camera tensor for all envs as uint8 [N,H,W,3]."""
        out = camera.data.output["rgb"]
        rgb = out[..., :3]
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return rgb.detach().cpu().numpy()

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

    # LeRobot dataset (created once; episodes appended as they pass).
    import shutil  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    if args.overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty dataset directory: {dataset_root}")
    features = {
        CAMERA_KEY: {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]},
        STATE_KEY: {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, root=dataset_root, fps=args.fps,
        robot_type="openarm_synthetic_isaac_dense", features=features,
        use_videos=False, image_writer_threads=0, image_writer_processes=0,
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
    saved_sample = False

    def gripper_schedule(phase: str, k: int, n: int) -> float:
        if phase in ("approach", "descend"):
            return open_m
        if phase == "close":
            frac = (k + 1) / max(1, n)
            return open_m + (closed_m - open_m) * frac
        return closed_m  # lift, hold

    for rnd in range(args.rounds):
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        origins = scene.env_origins
        ep_obj_local: dict[str, torch.Tensor] = {}
        for name in obj_names:
            asset = scene[name]
            root = asset.data.default_root_state.clone()
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
            root[:, 0:3] = local + origins
            root[:, 3:7] = ident
            root[:, 7:] = 0.0
            asset.write_root_pose_to_sim(root[:, :7])
            asset.write_root_velocity_to_sim(root[:, 7:])
        set_gripper(open_m)
        step_phys(args.settle_steps, render_last=True)  # also primes the camera render

        # Weighted target per env.
        target_idx = torch.multinomial(wtensor, N, replacement=True, generator=rng).to(device)

        ow = obj_pos_w()
        baseline_pos = ow.clone()
        baseline = baseline_pos[:, :, 2].clone()
        baseline_xy = baseline_pos[:, :, 0:2].clone()
        tw = ow.gather(1, target_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)  # [N,3]
        ox, oy = tw[:, 0], tw[:, 1]
        grasp_z = tw[:, 2] + float(args.grasp_z_offset_m)
        above = torch.stack([ox, oy, grasp_z + args.above_offset_m], dim=-1)
        descend = torch.stack([ox, oy, grasp_z], dim=-1)
        lift = torch.stack([ox, oy, grasp_z + args.lift_offset_m], dim=-1)

        # Per-env dense buffers.
        rgb_buf = [[] for _ in range(N)]
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
        last_arm = robot.data.joint_pos[:, ent.joint_ids].clone()

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
            set_ik_command(phase_target[phase])
            for k in range(n_steps):
                rgb = read_rgb()
                state = read_state_deg()
                if phase in ("approach", "descend", "lift"):
                    jclamped, hit, jdes = ik_solve_clamped()
                    last_arm = jclamped
                    clamp_hit |= hit
                    # Refined clip: genuine commanded-action clipping, ignoring the
                    # unavoidable joint_4 zero-start lower-bound correction.
                    clip_deg = torch.abs(jdes - jclamped) * 180.0 / math.pi  # [N,7]
                    j4 = clip_deg[:, joint4_idx]
                    clip_eff = clip_deg.clone()
                    clip_eff[:, joint4_idx] = torch.where(
                        j4 <= float(args.joint4_startup_tol_deg), torch.zeros_like(j4), j4
                    )
                    step_max_clip = clip_eff.max(dim=-1).values  # [N]
                    refined_clip_hit |= step_max_clip > float(args.action_clip_tol_deg)
                    max_refined_clip_deg = torch.maximum(max_refined_clip_deg, step_max_clip)
                else:  # close, hold -> hold last arm target
                    jclamped = last_arm
                gtarget_m = gripper_schedule(phase, k, n_steps)
                gtarget_deg = sim_finger_m_to_gripper_deg(gtarget_m)
                arm_deg = (jclamped * 180.0 / math.pi).detach().cpu().numpy()
                action = np.concatenate(
                    [arm_deg, np.full((N, 1), float(gtarget_deg), dtype=np.float32)], axis=1
                ).astype(np.float32)
                for e in range(N):
                    rgb_buf[e].append(rgb[e])
                    state_buf[e].append(state[e])
                    action_buf[e].append(action[e])
                # contact bookkeeping: tcp close to target object in xy/z
                tcp_w = robot.data.body_pose_w[:, tcp_idx, 0:3]
                d = torch.linalg.norm(tcp_w - tw, dim=-1)
                contact_steps += (d < args.contact_eps_m).float()
                robot.set_joint_position_target(jclamped, joint_ids=arm_ids)
                set_gripper(gtarget_m)
                step_phys(args.substeps, render_last=True)
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

        final = obj_pos_w()[:, :, 2]
        rises = final - baseline
        target_rise = rises.gather(1, target_idx.view(-1, 1)).squeeze(1)
        success = target_rise > args.lift_threshold_m
        wrong_mask = torch.ones_like(rises)
        wrong_mask.scatter_(1, target_idx.view(-1, 1), 0.0)
        wrong_any = ((rises * wrong_mask) > args.lift_threshold_m).any(dim=-1)

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
            wrong_total += int(is_wrong)
            clamp_total += int(is_clamp)
            object_collision_total += int(is_object_collision)
            gripper_table_collision_total += int(is_gripper_table_collision)
            object_sweep_total += int(is_object_sweep)
            finger_penetration_total += int(is_finger_penetration)
            object_pushed_down_total += int(is_object_pushed_down)
            refined_clip_total += int(is_refined_clip)
            keep = (
                bool(success[e].item())
                and not is_wrong
                and not is_object_collision
                and not is_gripper_table_collision
                and not is_object_sweep
                and not is_finger_penetration       # authoritative tabletop penetration
                and not is_object_pushed_down        # object driven into the table
                and not is_refined_clip              # genuine unsafe action clipping (joint_4 zero-start excluded)
            )
            if args.drop_limit_exceeded and is_clamp:
                keep = False
            if args.max_keep and kept >= args.max_keep:
                keep = False

            poses = {name: [round(float(v), 4) for v in ep_obj_local[name][e].tolist()] for name in obj_names}
            meta = {
                "schema_version": "openarm_dense_isaac_camera_v1",
                "source": "synthetic_smolvla.collect_dense_isaac_dataset",
                "episode_index": source_total - 1,
                "kept": keep,
                "instruction": instruction_for[tname],
                "target_object": tname,
                "arm_side": side,
                "image_size": image_size,
                "episode_len": episode_len,
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
                "finger_body_names": list(finger_body_names),
                "substeps": int(args.substeps),
                "jitter_x_m": round(float(args.jitter_x_m), 5),
                "jitter_y_m": round(float(args.jitter_y_m), 5),
                "gripper_cmd_min_deg": round(float(min(float(a[-1]) for a in action_buf[e])), 3),
                "gripper_cmd_max_deg": round(float(max(float(a[-1]) for a in action_buf[e])), 3),
                "gripper_close_cap_deg": round(float(args.max_gripper_close_deg), 3),
                "grasp_close_deg": round(float(effective_close_deg), 3),
                "grasp_z_offset_m": round(float(args.grasp_z_offset_m), 5),
                "lift_offset_m": round(float(args.lift_offset_m), 5),
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
                dataset.add_frame({
                    CAMERA_KEY: rgb_buf[e][t],
                    STATE_KEY: state_buf[e][t],
                    ACTION_KEY: action_buf[e][t],
                    "task": instruction_for[tname],
                })
            dataset.save_episode()
            kept += 1
            kept_by_target[tname] += 1

        print(f"[dense] round {rnd+1}/{args.rounds}: source={source_total} kept={kept} "
              f"(round success={float(success.float().mean()):.3f})", file=sys.stderr, flush=True)

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
        "Each kept episode is a DENSE rollout with real Isaac camera frames at every",
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
        f"- LeRobot dataset root: `{dataset_root}`",
        f"- LeRobot repo id: `{args.repo_id}`",
        f"- Episode metadata JSONL (dense state/action + poses/rises/contact): `{manifest_path}`",
        f"- Sample frames: `{args.sample_frame_dir}`",
        "",
        "## Notes",
        "",
        "- `observation.state` is the measured joint state; `action` is the clamped IK command.",
        "- Frames are the real Isaac scene camera (not the placeholder renderer); they move across the episode.",
        "- Only successful, correct-object lifts are kept; wrong-object lifts, object collisions, gripper/table collisions, and object sweep/slide episodes are rejected.",
        f"- Gripper close command is capped at `{effective_close_deg:.3f}` deg.",
        f"- Lift waypoint is `{float(args.lift_offset_m):.3f}` m above the grasp waypoint.",
        f"- Grasp z offset is `{float(args.grasp_z_offset_m):.3f}` m.",
    ]
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
        "grasp_close_deg": effective_close_deg,
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
