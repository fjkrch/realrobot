#!/usr/bin/env python3
"""Watch the trained SmolVLA drive the OpenArm in the Isaac Sim viewer.

Loads the just-trained V1 SmolVLA checkpoint and runs it closed-loop in the
four-object Isaac scene: each control step the policy receives the (in-domain)
top-down RGB + joint state + language instruction it was trained on, predicts an
8-D joint/gripper target (degrees), which is clamped to the OpenArm simulation
limit contract and applied to the arm in live physics. Run NON-headless to watch.

Nothing here touches a real robot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_scene import _isaac_paths, build_scene_cls  # noqa: E402
from oracle_policy import DEFAULT_OBJECT_POSES_M  # noqa: E402
from dataset_export import render_synthetic_camera  # noqa: E402
from sim_contract import (  # noqa: E402
    JOINT_NAMES,
    REPO_ROOT,
    SAFE_GRIPPER_LIMIT_DEG,
    clamp,
    clamp_joint_targets,
    gripper_deg_to_sim_finger_m,
    load_yaml_config,
)

DEFAULT_CKPT = (
    "synthetic_smolvla/checkpoints/smolvla_openarm_synth_v1/checkpoints/003000/pretrained_model"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="synthetic_smolvla/configs/scene_openarm_four_objects.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--headless", action="store_true", help="run without the viewer window")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps-per-instruction", type=int, default=50)
    parser.add_argument("--substeps", type=int, default=12, help="physics steps applied per policy action")
    parser.add_argument("--settle-steps", type=int, default=40)
    parser.add_argument("--lift-threshold-m", type=float, default=0.04)
    parser.add_argument("--loops", type=int, default=1, help="how many times to cycle through all instructions")
    return parser


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def patch_checkpoint(ckpt_dir: Path) -> Path:
    """Create a sibling dir with the same weights but a config.json that carries
    the ``type: smolvla`` discriminator lerobot's loader requires.

    Non-destructive: the big weight files are symlinked, not copied.
    """
    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("type") == "smolvla":
        return ckpt_dir
    fixed = ckpt_dir.parent / (ckpt_dir.name + "_typed")
    fixed.mkdir(exist_ok=True)
    for item in ckpt_dir.iterdir():
        if item.name == "config.json":
            continue
        link = fixed / item.name
        if not link.exists():
            link.symlink_to(item.resolve())
    cfg = {"type": "smolvla", **cfg}
    (fixed / "config.json").write_text(json.dumps(cfg, indent=2))
    return fixed


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    side = config["scene"].get("active_arm", "right")
    object_names = [obj["name"] for obj in config["objects"]]
    instruction_for = {obj["name"]: obj["instruction"] for obj in config["objects"]}

    ckpt = _abs(args.checkpoint)
    fixed_ckpt = patch_checkpoint(ckpt)
    print(f"[demo] using checkpoint {fixed_ckpt}", file=sys.stderr, flush=True)

    _isaac_paths()
    from isaaclab.app import AppLauncher

    print(f"[demo] launching Isaac (headless={args.headless})", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.utils.control_utils import predict_action  # noqa: PLC0415

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[demo] loading SmolVLA policy ...", file=sys.stderr, flush=True)
    policy = SmolVLAPolicy.from_pretrained(str(fixed_ckpt))
    policy.to(dev)
    policy.eval()
    cfg = policy.config
    cfg.pretrained_path = str(fixed_ckpt)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(fixed_ckpt),
        preprocessor_overrides={"device_processor": {"device": str(dev)}},
    )

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
    cam = config["scene"]["camera"]
    sim.set_camera_view(eye=cam["eye_m"], target=cam["target_m"])
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    arm_ids, _ = robot.find_joints([f"openarm_{side}_joint{i}" for i in range(1, 8)], preserve_order=True)
    finger_ids, _ = robot.find_joints(f"openarm_{side}_finger_joint.*")

    reset_deg = config["robot"]["reset_pose_deg"][side]
    reset_state = np.asarray(
        [float(reset_deg[j]) for j in JOINT_NAMES] + [float(reset_deg.get("gripper", -65.0))],
        dtype=np.float32,
    )

    def place_objects() -> None:
        origin = scene.env_origins[0]
        for name in object_names:
            pose = DEFAULT_OBJECT_POSES_M[name]
            asset = scene[name]
            root = asset.data.default_root_state.clone()
            root[0, 0] = float(pose[0]) + float(origin[0])
            root[0, 1] = float(pose[1]) + float(origin[1])
            root[0, 2] = float(pose[2]) + float(origin[2])
            root[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
            root[0, 7:] = 0.0
            asset.write_root_pose_to_sim(root[:, :7])
            asset.write_root_velocity_to_sim(root[:, 7:])

    def apply(state8: np.ndarray, n: int) -> None:
        arm = torch.tensor(
            [[float(np.radians(state8[i])) for i in range(7)]], device=robot.device, dtype=torch.float32
        )
        fm = gripper_deg_to_sim_finger_m(float(state8[7]))
        fingers = torch.full((1, len(finger_ids)), float(fm), device=robot.device, dtype=torch.float32)
        for _ in range(n):
            robot.set_joint_position_target(arm, joint_ids=arm_ids)
            robot.set_joint_position_target(fingers, joint_ids=finger_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    def object_z(name: str) -> float:
        return float(scene[name].data.root_pos_w[0, 2].item())

    results = []
    for loop in range(args.loops):
        for target in object_names:
            instruction = instruction_for[target]
            record = {
                "target_object": target,
                "object_poses_m": DEFAULT_OBJECT_POSES_M,
                "visible_objects": object_names,  # V1 was trained all-visible
            }
            image = render_synthetic_camera(record, image_size=96)

            # Fresh start: robot to reset pose, objects replaced, settle.
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            place_objects()
            apply(reset_state, args.settle_steps)
            baseline = object_z(target)

            policy.reset()
            pre.reset()
            post.reset()
            state = reset_state.copy()
            print(f"\n[demo] >>> instruction: '{instruction}'  (target {target})", file=sys.stderr, flush=True)
            for t in range(args.steps_per_instruction):
                obs = {
                    "observation.images.camera1": image,
                    "observation.state": state.astype(np.float32),
                }
                action = predict_action(obs, policy, dev, pre, post, use_amp=False, task=instruction)
                a = action.detach().cpu().numpy().astype(np.float32)
                clamped = clamp_joint_targets(side, {JOINT_NAMES[i]: float(a[i]) for i in range(7)})
                grip = clamp(float(a[7]), *SAFE_GRIPPER_LIMIT_DEG)
                state = np.asarray([clamped[j] for j in JOINT_NAMES] + [grip], dtype=np.float32)
                apply(state, args.substeps)

            rise = object_z(target) - baseline
            success = bool(rise > args.lift_threshold_m)
            results.append({"target": target, "rise_m": round(rise, 5), "success": success})
            print(
                f"[demo] {target}: target rose {rise:+.4f} m -> picked={success} "
                f"(final joints deg={np.round(state,1).tolist()})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.6)  # brief pause so each pick is visible

    print(json.dumps({"ok": True, "results": results}, indent=2), flush=True)
    if not args.headless:
        print("[demo] holding viewer for 8 s ...", file=sys.stderr, flush=True)
        for _ in range(int(8.0 / sim_dt)):
            sim.step()
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
