#!/usr/bin/env python3
"""Evaluate a trained SmolVLA OpenArm policy in Isaac physics.

Runs closed-loop simulated pick-and-lift trials. Each trial gives the policy the
same observation schema used during training: deterministic top-down RGB,
8-D joint state, and the language task. The predicted 8-D action is clamped to
the OpenArm simulation contract before being applied to Isaac.

Simulation only. This never touches a real robot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_export import render_synthetic_camera  # noqa: E402
from make_scene import _isaac_paths, build_scene_cls  # noqa: E402
from sim_contract import (  # noqa: E402
    JOINT_NAMES,
    REPO_ROOT,
    SAFE_GRIPPER_LIMIT_DEG,
    clamp,
    clamp_joint_targets,
    gripper_deg_to_sim_finger_m,
    load_yaml_config,
    sim_finger_m_to_gripper_deg,
    validate_scene_config,
)


DEFAULT_CKPT = (
    "synthetic_smolvla/checkpoints/smolvla_openarm_dense_isaac_camera_v1/"
    "checkpoints/020000/pretrained_model"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--viewer", dest="headless", action="store_false", help="open the Isaac viewer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps-per-trial", type=int, default=50)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--settle-steps", type=int, default=40)
    parser.add_argument("--lift-threshold-m", type=float, default=0.04)
    parser.add_argument("--jitter-x-m", type=float, default=0.015)
    parser.add_argument("--jitter-y-m", type=float, default=0.008)
    parser.add_argument("--record-action-trace", action="store_true")
    # The corrected pipeline trains on REAL Isaac camera frames, so eval must feed
    # the same real camera (captured per step). --placeholder-camera reverts to the
    # old deterministic top-down renderer for evaluating legacy placeholder models.
    parser.add_argument("--real-camera", dest="real_camera", action="store_true", default=True)
    parser.add_argument("--placeholder-camera", dest="real_camera", action="store_false")
    parser.add_argument("--output-jsonl", default="synthetic_smolvla/reports/dense_isaac_camera_v1_eval.jsonl")
    parser.add_argument("--output-md", default="synthetic_smolvla/reports/dense_isaac_camera_v1_eval.md")
    return parser


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def patch_checkpoint(ckpt_dir: Path) -> Path:
    """Add the `type: smolvla` discriminator expected by LeRobot if missing."""
    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
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
    (fixed / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fixed


def _jittered_poses(
    config: dict[str, Any],
    *,
    rng: random.Random,
    jitter_x_m: float,
    jitter_y_m: float,
) -> dict[str, list[float]]:
    bounds = config["scene"]["workspace_bounds_m"]
    poses = {}
    for obj in config["objects"]:
        x, y, z = [float(v) for v in obj["spawn_pose_m"]]
        x = clamp(x + rng.uniform(-jitter_x_m, jitter_x_m), *bounds["x"])
        y = clamp(y + rng.uniform(-jitter_y_m, jitter_y_m), *bounds["y"])
        poses[obj["name"]] = [x, y, z]
    return poses


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def write_report(
    records: list[dict[str, Any]],
    path: Path,
    *,
    checkpoint: Path,
    jsonl_path: Path,
    error: BaseException | None = None,
) -> None:
    total = len(records)
    success = sum(1 for r in records if r["success_label"])
    wrong = sum(1 for r in records if r["wrong_object_lifted"])
    by_target = Counter(r["target_object"] for r in records)
    succ_by_target = Counter(r["target_object"] for r in records if r["success_label"])
    wrong_by_target = Counter(r["target_object"] for r in records if r["wrong_object_lifted"])
    avg_rise = sum(float(r["target_rise_m"]) for r in records) / total if total else 0.0

    lines = [
        "# SmolVLA Isaac Policy Evaluation",
        "",
        "Closed-loop policy rollout in Isaac physics. Simulation only; no real robot motion.",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Trials | {total} | 1.000 |",
        f"| Success | {success} | {_rate(success, total):.3f} |",
        f"| Wrong object lifted | {wrong} | {_rate(wrong, total):.3f} |",
        "",
        f"- Average target rise: `{avg_rise:.5f} m`",
        f"- Checkpoint: `{checkpoint}`",
        f"- JSONL trials: `{jsonl_path}`",
    ]
    if error is not None:
        lines.extend(
            [
                f"- Evaluation error: `{type(error).__name__}: {error}`",
                "",
                "This report is partial because rollout stopped before all requested trials completed.",
            ]
        )
    lines.extend(
        [
            "",
            "## By Target",
            "",
            "| Target | Success | Trials | Success Rate | Wrong Object Lifts |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target in sorted(by_target):
        trials = by_target[target]
        lines.append(
            f"| {target} | {succ_by_target[target]} | {trials} | "
            f"{_rate(succ_by_target[target], trials):.3f} | {wrong_by_target[target]} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Success means the requested target object rose above the lift threshold.",
            "- Wrong-object lift is measured from non-target object rises.",
            "- RGB is the real Isaac scene camera captured per step (default), matching the dense training dataset.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    validate_scene_config(config)

    side = config["scene"].get("active_arm", "right")
    object_names = [obj["name"] for obj in config["objects"]]
    instruction_for = {obj["name"]: obj["instruction"] for obj in config["objects"]}

    ckpt = patch_checkpoint(_abs(args.checkpoint))
    output_jsonl = _abs(args.output_jsonl)
    output_md = _abs(args.output_md)
    rng = random.Random(args.seed)

    print(f"[eval-vla] using checkpoint {ckpt}", file=sys.stderr, flush=True)
    _isaac_paths()
    from isaaclab.app import AppLauncher

    print(f"[eval-vla] launching Isaac (headless={args.headless})", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415
    from lerobot.utils.control_utils import predict_action  # noqa: PLC0415

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[eval-vla] loading policy", file=sys.stderr, flush=True)
    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy.to(dev)
    policy.eval()
    policy_cfg = policy.config
    policy_cfg.pretrained_path = str(ckpt)
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt),
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

    # Real Isaac camera (matches the dense dataset) or legacy placeholder renderer.
    scene_camera = scene["camera"]
    if args.real_camera:
        eye_w = torch.tensor([float(v) for v in cam["eye_m"]], device=robot.device).unsqueeze(0) + scene.env_origins
        tgt_w = torch.tensor([float(v) for v in cam["target_m"]], device=robot.device).unsqueeze(0) + scene.env_origins
        scene_camera.set_world_poses_from_view(eye_w, tgt_w)
        cam_res = cam["resolution"]
        if int(cam_res[0]) != int(cam_res[1]):
            raise SystemExit(f"Real-camera eval expects a square camera resolution, got {cam_res}.")
        image_size = int(cam_res[0])
    else:
        image_size = 96

    def read_rgb() -> "np.ndarray":
        out = scene_camera.data.output["rgb"][0]
        rgb = out[..., :3]
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return rgb.detach().cpu().numpy()

    reset_deg = config["robot"]["reset_pose_deg"][side]
    reset_state = np.asarray(
        [float(reset_deg[j]) for j in JOINT_NAMES] + [float(reset_deg.get("gripper", -65.0))],
        dtype=np.float32,
    )

    def place_objects(poses: dict[str, list[float]]) -> None:
        origin = scene.env_origins[0]
        for name, pose in poses.items():
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
            [[float(np.radians(state8[i])) for i in range(7)]],
            device=robot.device,
            dtype=torch.float32,
        )
        fm = gripper_deg_to_sim_finger_m(float(state8[7]))
        fingers = torch.full((1, len(finger_ids)), float(fm), device=robot.device, dtype=torch.float32)
        for _ in range(n):
            robot.set_joint_position_target(arm, joint_ids=arm_ids)
            robot.set_joint_position_target(fingers, joint_ids=finger_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    def read_state() -> np.ndarray:
        joints = robot.data.joint_pos[:, arm_ids][0].detach().cpu().numpy()
        fingers = robot.data.joint_pos[:, finger_ids][0].detach().cpu().numpy()
        gripper_deg = sim_finger_m_to_gripper_deg(float(fingers.mean()))
        return np.asarray([float(np.degrees(v)) for v in joints] + [gripper_deg], dtype=np.float32)

    def object_z(name: str) -> float:
        return float(scene[name].data.root_pos_w[0, 2].item())

    started = time.time()
    records: list[dict[str, Any]] = []
    error: BaseException | None = None
    print(f"[eval-vla] entering rollout loop for {args.trials} trials", file=sys.stderr, flush=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")
    try:
        for trial in range(args.trials):
            target = object_names[trial % len(object_names)]
            instruction = instruction_for[target]
            print(
                f"[eval-vla] starting {trial + 1}/{args.trials}: {instruction}",
                file=sys.stderr,
                flush=True,
            )
            poses = _jittered_poses(
                config,
                rng=rng,
                jitter_x_m=args.jitter_x_m,
                jitter_y_m=args.jitter_y_m,
            )
            image_record = {
                "target_object": target,
                "object_poses_m": poses,
                "visible_objects": object_names,
            }

            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            place_objects(poses)
            apply(reset_state, args.settle_steps)
            baseline = {name: object_z(name) for name in object_names}

            def current_image():
                if args.real_camera:
                    return read_rgb()
                return render_synthetic_camera(image_record, image_size=image_size)

            policy.reset()
            pre.reset()
            post.reset()
            state = read_state()
            image = current_image()
            clamp_events = 0
            final_command = state.copy()
            action_trace: list[dict[str, Any]] = []
            for _ in range(args.steps_per_trial):
                obs = {
                    "observation.images.camera1": image,
                    "observation.state": state.astype(np.float32),
                }
                action = predict_action(obs, policy, dev, pre, post, use_amp=False, task=instruction)
                raw_np = action.detach().cpu().numpy().astype(np.float32)
                if raw_np.shape[-1] != len(JOINT_NAMES) + 1:
                    raise ValueError(f"Expected action last dim 8, got shape {raw_np.shape}")
                raw = raw_np.reshape(-1, raw_np.shape[-1])[0]
                arm_raw = {JOINT_NAMES[i]: float(raw[i]) for i in range(7)}
                arm_clamped = clamp_joint_targets(side, arm_raw)
                grip = clamp(float(raw[7]), *SAFE_GRIPPER_LIMIT_DEG)
                if any(abs(arm_clamped[j] - arm_raw[j]) > 1e-5 for j in JOINT_NAMES):
                    clamp_events += 1
                if abs(grip - float(raw[7])) > 1e-5:
                    clamp_events += 1
                command = np.asarray([arm_clamped[j] for j in JOINT_NAMES] + [grip], dtype=np.float32)
                apply(command, args.substeps)
                observed = read_state()
                if args.record_action_trace:
                    action_trace.append(
                        {
                            "command_deg": [round(float(v), 3) for v in command.tolist()],
                            "observed_state_deg": [round(float(v), 3) for v in observed.tolist()],
                        }
                    )
                state = observed
                image = current_image()
                final_command = command.copy()

            final_z = {name: object_z(name) for name in object_names}
            rises = {name: final_z[name] - baseline[name] for name in object_names}
            target_rise = rises[target]
            success = target_rise > args.lift_threshold_m
            wrong = any(rise > args.lift_threshold_m for name, rise in rises.items() if name != target)
            record = {
                "schema_version": "openarm_smolvla_isaac_eval_v1",
                "source": "synthetic_smolvla.eval_vla_isaac",
                "trial_index": trial,
                "instruction": instruction,
                "target_object": target,
                "arm_side": side,
                "randomized": True,
                "all_objects_visible": True,
                "visible_objects": list(object_names),
                "object_poses_m": {name: [round(float(v), 5) for v in pose] for name, pose in poses.items()},
                "target_rise_m": round(float(target_rise), 5),
                "object_rises_m": {name: round(float(rise), 5) for name, rise in rises.items()},
                "success_label": bool(success),
                "wrong_object_lifted": bool(wrong),
                "limit_clamp_events": int(clamp_events),
                "final_action_deg": [round(float(v), 3) for v in final_command.tolist()],
                "final_observed_state_deg": [round(float(v), 3) for v in state.tolist()],
            }
            if args.record_action_trace:
                record["action_trace"] = action_trace
            records.append(record)
            with output_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"[eval-vla] {trial + 1}/{args.trials} {target}: "
                f"rise={target_rise:+.4f} success={success} wrong={wrong}",
                file=sys.stderr,
                flush=True,
            )
    except BaseException as exc:
        error = exc
        print(f"[eval-vla] ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        _write_jsonl(records, output_jsonl)
        write_report(records, output_md, checkpoint=ckpt, jsonl_path=output_jsonl, error=error)
        success_count = sum(1 for r in records if r["success_label"])
        wrong_count = sum(1 for r in records if r["wrong_object_lifted"])
        print(
            json.dumps(
                {
                    "ok": error is None,
                    "trials": len(records),
                    "success": success_count,
                    "success_rate": _rate(success_count, len(records)),
                    "wrong_object": wrong_count,
                    "wrong_object_rate": _rate(wrong_count, len(records)),
                    "duration_sec": time.time() - started,
                    "jsonl": str(output_jsonl),
                    "report": str(output_md),
                    "error": None if error is None else f"{type(error).__name__}: {error}",
                },
                indent=2,
            ),
            flush=True,
        )
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
