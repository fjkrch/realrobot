#!/usr/bin/env python3
"""Visually re-run a saved NPZ episode in Isaac Sim (simulation only).

Loads a kept episode's 8D action trace (7 left-arm joints + gripper, degrees),
rebuilds the same photo-clean scene, and drives the LEFT arm through the saved
commands at the recorded rate so you can watch the grasp in the RTX viewer.

No real robot, no CAN, no SSH. The right arm stays hidden/locked.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from sim_contract import (  # noqa: E402
    REPO_ROOT,
    gripper_deg_to_sim_finger_m,
    load_yaml_config,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", required=True, help="path to episode_XXXXXX.npz")
    p.add_argument("--config", required=True, help="scene config the episode was collected against")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--rendering-mode", default="quality", choices=("performance", "balanced", "quality"))
    p.add_argument("--substeps", type=int, default=20, help="physics substeps per command (20 @dt=0.005 = 10 Hz)")
    p.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pace visible replay to wall-clock rate",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="wall-clock command replay rate when --realtime is enabled",
    )
    p.add_argument("--loops", type=int, default=3, help="how many times to replay the trajectory")
    p.add_argument("--settle-steps", type=int, default=30)
    p.add_argument("--capture-dir", default=None, help="optional directory for replay camera PPM captures")
    p.add_argument("--capture-count", type=int, default=10, help="captures to save during the first replay loop")
    p.add_argument("--capture-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
    p.add_argument(
        "--camera-view",
        default="config",
        choices=("config", "front", "robot"),
        help=(
            "viewer/capture camera: config uses the configured pose as a free viewport, "
            "front uses the visual front-check pose, robot locks the GUI viewport to the "
            "actual Isaac scene camera sensor"
        ),
    )
    p.add_argument("--camera-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--camera-target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--kit-args", default="--/app/window/width=1100 --/app/window/height=760")
    args = p.parse_args()
    if args.rate_hz <= 0.0:
        p.error("--rate-hz must be positive")
    if args.capture_dir and args.capture_count <= 0:
        p.error("--capture-count must be positive when --capture-dir is provided")

    ep = Path(args.episode)
    ep = ep if ep.is_absolute() else REPO_ROOT / ep
    data = np.load(ep, allow_pickle=True)
    action = np.asarray(data["action"], dtype=np.float32)  # [T,8] deg
    task = str(data["task"])
    T = action.shape[0]
    print(f"[replay] episode={ep.name} commands={T} task={task!r}", flush=True)

    config = load_yaml_config(args.config)
    if args.capture_resolution is not None:
        config["scene"]["camera"]["resolution"] = [int(args.capture_resolution[0]), int(args.capture_resolution[1])]
    side = config["scene"].get("active_arm", "left")
    cam_cfg = config["scene"]["camera"]

    def camera_pose() -> tuple[list[float], list[float]]:
        if args.camera_eye is not None or args.camera_target is not None:
            if args.camera_eye is None or args.camera_target is None:
                p.error("--camera-eye and --camera-target must be passed together")
            return [float(v) for v in args.camera_eye], [float(v) for v in args.camera_target]
        if args.camera_view == "front":
            base = [float(v) for v in config.get("robot", {}).get("base_pose_m", [0.38, 0.08, 0.60])]
            return [base[0] - 0.82, base[1], base[2] + 0.24], [base[0] + 0.08, base[1], base[2] + 0.15]
        return [float(v) for v in cam_cfg["eye_m"]], [float(v) for v in cam_cfg["target_m"]]

    cam_eye, cam_target = camera_pose()

    capture_dir = None
    capture_indices: set[int] = set()
    capture_manifest: list[dict[str, object]] = []
    if args.capture_dir:
        capture_dir = Path(args.capture_dir)
        if not capture_dir.is_absolute():
            capture_dir = REPO_ROOT / capture_dir
        capture_dir.mkdir(parents=True, exist_ok=True)
        capture_total = min(int(args.capture_count), T)
        capture_indices = set(int(v) for v in np.linspace(0, T - 1, capture_total, dtype=int).tolist())

    import make_scene as mk  # noqa: PLC0415
    mk._isaac_paths()  # noqa: SLF001
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=True,
        rendering_mode=args.rendering_mode,
        kit_args=args.kit_args,
    )
    simulation_app = app_launcher.app

    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415

    scene_cls = mk.build_scene_cls(
        config, sim_utils=sim_utils, AssetBaseCfg=AssetBaseCfg, RigidObjectCfg=RigidObjectCfg,
        CameraCfg=CameraCfg, InteractiveSceneCfg=InteractiveSceneCfg, configclass=configclass,
    )
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim.set_camera_view(eye=cam_eye, target=cam_target)
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    scene_camera = scene["camera"]

    def aim_scene_camera() -> None:
        eye_w = (
            torch.tensor(cam_eye, device=robot.device, dtype=torch.float32).unsqueeze(0)
            + scene.env_origins
        )
        target_w = (
            torch.tensor(cam_target, device=robot.device, dtype=torch.float32).unsqueeze(0)
            + scene.env_origins
        )
        scene_camera.set_world_poses_from_view(eye_w, target_w)

    def scene_camera_prim_path() -> str | None:
        sensor_prims = getattr(scene_camera, "_sensor_prims", None)
        if sensor_prims:
            return sensor_prims[0].GetPath().pathString
        view = getattr(scene_camera, "_view", None)
        prims = getattr(view, "prims", None)
        if prims:
            return prims[0].GetPath().pathString
        return None

    def lock_viewport_to_scene_camera() -> None:
        if args.headless or args.camera_view != "robot":
            return
        camera_prim_path = scene_camera_prim_path()
        if not camera_prim_path:
            print("[replay] warning: could not resolve scene camera prim path for robot camera view", flush=True)
            return
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if hasattr(viewport, "set_active_camera"):
                viewport.set_active_camera(camera_prim_path)
            else:
                viewport.camera_path = camera_prim_path
            print(f"[replay] viewport locked to robot camera sensor {camera_prim_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - replay can still continue with sensor captures
            print(f"[replay] warning: could not lock viewport to robot camera sensor: {exc}", flush=True)

    aim_scene_camera()
    lock_viewport_to_scene_camera()
    bound_robot_geoms = mk.force_bind_robot_visual_material(config)
    if bound_robot_geoms:
        print(f"[replay] force-bound clean robot material to {bound_robot_geoms} geom prims", flush=True)

    def apply_robot_visuals() -> None:
        # scene.reset() can refresh Fabric state from the original robot asset.
        # Re-apply after resets so hidden arms and per-subset bindings survive.
        mk.force_bind_robot_visual_material(config)

    joint_names = list(robot.joint_names)
    arm_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    arm_ids = [joint_names.index(n) for n in arm_names]
    finger_ids, _ = robot.find_joints(f"openarm_{side}_finger_joint.*")

    def drive(arm_deg: np.ndarray, gripper_deg: float) -> None:
        arm_rad = torch.tensor(arm_deg * math.pi / 180.0, device=args.device, dtype=torch.float32).unsqueeze(0)
        robot.set_joint_position_target(arm_rad, joint_ids=arm_ids)
        finger_m = float(gripper_deg_to_sim_finger_m(float(gripper_deg)))
        finger_t = torch.full((1, len(finger_ids)), finger_m, device=args.device, dtype=torch.float32)
        robot.set_joint_position_target(finger_t, joint_ids=finger_ids)
        robot.write_data_to_sim()

    def reset_robot_to_command(arm_deg: np.ndarray, gripper_deg: float) -> None:
        """Place the simulated robot exactly at a saved command before rendering."""
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        arm_rad = torch.tensor(arm_deg * math.pi / 180.0, device=args.device, dtype=torch.float32).unsqueeze(0)
        joint_pos[:, arm_ids] = arm_rad
        joint_vel[:, arm_ids] = 0.0
        finger_m = float(gripper_deg_to_sim_finger_m(float(gripper_deg)))
        joint_pos[:, finger_ids] = finger_m
        joint_vel[:, finger_ids] = 0.0
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.reset()
        drive(arm_deg, gripper_deg)

    def save_capture(loop_index: int, command_index: int) -> None:
        if capture_dir is None:
            return
        rgb = scene_camera.data.output["rgb"][0].detach().cpu().numpy()
        image = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
        height, width = image.shape[:2]
        out = capture_dir / f"loop{loop_index + 1:02d}_cmd{command_index:04d}.ppm"
        with out.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(image.tobytes())
        capture_manifest.append(
            {
                "loop": loop_index + 1,
                "command_index": int(command_index),
            "path": str(out),
            "camera_eye": cam_eye,
            "camera_target": cam_target,
        }
        )

    # settle at the first saved command, which is zero pose for zero-start episodes.
    reset_robot_to_command(action[0, :7], float(action[0, 7]))
    for _ in range(args.settle_steps):
        sim.step(render=True)
        scene.update(sim.get_physics_dt())

    dt = sim.get_physics_dt()
    realtime_replay = args.realtime and not args.headless
    command_period_s = 1.0 / args.rate_hz
    for loop in range(args.loops):
        print(f"[replay] loop {loop + 1}/{args.loops}", flush=True)
        # reset objects, then place the robot at the first saved command before rendering.
        scene.reset()
        aim_scene_camera()
        lock_viewport_to_scene_camera()
        reset_robot_to_command(action[0, :7], float(action[0, 7]))
        apply_robot_visuals()
        for _ in range(args.settle_steps):
            sim.step(render=True)
            scene.update(dt)
        for t in range(T):
            command_start = time.perf_counter()
            drive(action[t, :7], float(action[t, 7]))
            for _ in range(args.substeps):
                sim.step(render=True)
                scene.update(dt)
            if realtime_replay:
                remaining_s = command_period_s - (time.perf_counter() - command_start)
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
            if loop == 0 and t in capture_indices:
                save_capture(loop, t)
        # brief hold so you can see the lifted result
        for _ in range(40):
            sim.step(render=True)
            scene.update(dt)

    if capture_dir is not None:
        manifest = {
            "episode": str(ep),
            "config": str(args.config),
            "commands": int(T),
            "capture_count": len(capture_manifest),
            "camera_view": str(args.camera_view),
            "camera_eye": cam_eye,
            "camera_target": cam_target,
            "scene_camera_prim_path": scene_camera_prim_path(),
            "captures": capture_manifest,
        }
        (capture_dir / "capture_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[replay] wrote {len(capture_manifest)} captures to {capture_dir}", flush=True)

    print("[replay] done; keeping viewer open. Close the window or Ctrl-C to exit.", flush=True)
    while simulation_app.is_running() and not args.headless:
        sim.step(render=True)
        scene.update(dt)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
