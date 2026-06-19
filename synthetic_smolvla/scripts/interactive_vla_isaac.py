#!/usr/bin/env python3
"""Interactive SmolVLA Isaac demo with typed tasks and random resets.

Default behavior is simulation only. This script opens Isaac, waits for a typed task such as
``pick up the red cube``, runs one closed-loop VLA rollout, reports the measured
lift result, then resets the robot and objects to a fresh randomized pose for
the next typed task. Optional dry-run and explicitly confirmed real mirror sinks
can observe the same clamped policy commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_vla_isaac import _abs, _jittered_poses, patch_checkpoint  # noqa: E402
from language import LanguageError, instruction_for_object, parse_target_object  # noqa: E402
from make_scene import _isaac_paths, build_scene_cls  # noqa: E402
from mirror_sinks import (  # noqa: E402
    DEFAULT_FIRST_TARGET_TOLERANCE_DEG,
    DEFAULT_REAL_HELPER,
    DEFAULT_REAL_HOST,
    DEFAULT_REAL_REPO,
    DEFAULT_REAL_USER,
    DEFAULT_START_POSE_TOLERANCE_DEG,
    REQUIRED_REAL_CONFIRMATION,
    CommandContext,
    CompositeSink,
    DryRunMirrorSink,
    MirrorSafetyError,
    RealMirrorConfig,
    RealMirrorSink,
    SimSink,
    start_pose_from_config,
)
from sim_contract import (  # noqa: E402
    JOINT_NAMES,
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
    "checkpoints/015000/pretrained_model"
)


def default_real_port_for_side(side: str) -> str:
    return "can0" if side == "right" else "can1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--headless", action="store_true", default=True, help="run without opening the Isaac viewer")
    parser.add_argument("--viewer", dest="headless", action="store_false", help="open the Isaac viewer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--viewer-eye-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="viewer-only starting camera eye in meters; defaults to scene.camera.eye_m",
    )
    parser.add_argument(
        "--viewer-target-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="viewer-only starting camera target in meters; defaults to scene.camera.target_m",
    )
    parser.add_argument(
        "--policy-device",
        default=None,
        help="device for SmolVLA inference; use cpu with --viewer on 8 GB GPUs",
    )
    parser.add_argument("--steps-per-task", type=int, default=50)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument(
        "--render-every-sim-steps",
        type=int,
        default=1,
        help=(
            "render/camera-update interval in physics steps; default 1 renders every physics step. "
            "Increase this only when you want a lower render rate."
        ),
    )
    parser.add_argument(
        "--sim-camera-size-px",
        type=int,
        default=None,
        help=(
            "override the Isaac RGB camera to a square lower resolution, for example 128. "
            "Images are resized back to the original config resolution before policy inference."
        ),
    )
    parser.add_argument("--settle-steps", type=int, default=40)
    parser.add_argument("--lift-threshold-m", type=float, default=0.04)
    parser.add_argument("--jitter-x-m", type=float, default=0.015)
    parser.add_argument("--jitter-y-m", type=float, default=0.008)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument(
        "--use-typed-language",
        action="store_true",
        help="send the exact typed text to SmolVLA; default canonicalizes to the training instruction",
    )
    parser.add_argument(
        "--save-debug-frames",
        default=None,
        help="directory for initial/final RGB PPM frames and, with --save-depth, normalized depth PGM frames",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="also request distance_to_image_plane and save/print depth summaries",
    )
    parser.add_argument(
        "--mirror-dry-run",
        default=None,
        help="write timestamped clamped policy commands to this JSONL path without touching the real robot",
    )
    parser.add_argument(
        "--mirror-real",
        action="store_true",
        help="enable the guarded real OpenArm mirror path; requires --prepare-real-start-pose and exact --real-confirm",
    )
    parser.add_argument("--real-confirm", default="", help="exact real-motion acknowledgement phrase")
    parser.add_argument(
        "--prepare-real-start-pose",
        action="store_true",
        help="move the real arm to the config reset pose before any real mirror target",
    )
    parser.add_argument(
        "--read-real-state",
        action="store_true",
        help="read current real arm+gripper state through the guarded helper",
    )
    parser.add_argument(
        "--real-preflight-only",
        action="store_true",
        help="after real state/start-pose preflight, close the helper and exit before accepting typed tasks",
    )
    parser.add_argument(
        "--real-replay-after-sim-success",
        action="store_true",
        help="run the task in sim first, save the command trajectory, then replay it on the real arm only after sim success",
    )
    parser.add_argument(
        "--save-trajectory-dir",
        default="synthetic_smolvla/reports/interactive_vla_saved_trajectories",
        help="directory for per-task command trajectories and summaries",
    )
    parser.add_argument("--mirror-rate-hz", type=float, default=2.0, help="maximum real target send rate")
    parser.add_argument(
        "--max-joint-delta-deg",
        type=float,
        default=3.0,
        help="refuse real mirror targets with a larger arm-joint step delta",
    )
    parser.add_argument(
        "--real-helper-max-rel-deg",
        type=float,
        default=None,
        help=(
            "OpenArmFollower max_relative_target sent to the helper. "
            "Use a larger value for direct init while keeping --max-joint-delta-deg for VLA sampling."
        ),
    )
    parser.add_argument(
        "--start-pose-max-joint-delta-deg",
        type=float,
        default=None,
        help="compatibility option; normal start-pose preparation uses the guarded helper prepare_start path",
    )
    parser.add_argument(
        "--start-pose-gripper-max-delta-deg",
        type=float,
        default=None,
        help="compatibility option; normal start-pose preparation uses the guarded helper prepare_start path",
    )
    parser.add_argument(
        "--start-pose-rate-hz",
        type=float,
        default=None,
        help="compatibility option; normal start-pose preparation uses the guarded helper prepare_start path",
    )
    parser.add_argument(
        "--disable-gripper-real",
        dest="disable_gripper_real",
        action="store_true",
        default=True,
        help="do not send gripper commands to the real robot (default)",
    )
    parser.add_argument(
        "--enable-gripper-real",
        dest="disable_gripper_real",
        action="store_false",
        help="allow real gripper commands; not recommended for first mirror tests",
    )
    parser.add_argument("--watchdog-timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--hold-interval-sec",
        type=float,
        default=0.2,
        help="how often to resend the prepared real target while waiting at the prompt",
    )
    parser.add_argument("--real-side", choices=["left", "right"], default=None, help="real arm side; defaults to scene active arm")
    parser.add_argument("--real-port", default=None, help="real CAN port; defaults from --real-side")
    parser.add_argument("--real-host", default=DEFAULT_REAL_HOST)
    parser.add_argument("--real-user", default=DEFAULT_REAL_USER)
    parser.add_argument("--real-repo", default=DEFAULT_REAL_REPO)
    parser.add_argument("--real-helper", default=DEFAULT_REAL_HELPER)
    parser.add_argument("--real-request-timeout-sec", type=float, default=8.0)
    parser.add_argument("--real-connect-timeout-sec", type=float, default=5.0)
    parser.add_argument("--real-start-pose-tolerance-deg", type=float, default=DEFAULT_START_POSE_TOLERANCE_DEG)
    parser.add_argument(
        "--real-start-pose-timeout-sec",
        type=float,
        default=25.0,
        help="seconds to let the normal helper prepare_start path reach the configured start pose",
    )
    parser.add_argument(
        "--real-start-pose-hold-sec",
        type=float,
        default=0.3,
        help="seconds the real start pose must stay within tolerance before accepting tasks",
    )
    parser.add_argument(
        "--real-start-pose-samples",
        type=int,
        default=1,
        help="number of sampled prepare_start targets for init; 1 means direct/default init",
    )
    parser.add_argument(
        "--real-start-pose-duration-sec",
        type=float,
        default=None,
        help="stream init targets for this many seconds, using --start-pose-rate-hz if set",
    )
    parser.add_argument("--first-real-target-tolerance-deg", type=float, default=DEFAULT_FIRST_TARGET_TOLERANCE_DEG)
    parser.add_argument("--real-connect-retries", type=int, default=3)
    parser.add_argument("--real-connect-retry-delay-sec", type=float, default=1.5)
    return parser


def _build_real_config(args: argparse.Namespace, *, side: str) -> RealMirrorConfig:
    real_side = args.real_side or side
    real_port = args.real_port or default_real_port_for_side(real_side)
    return RealMirrorConfig(
        side=real_side,
        port=real_port,
        confirm=args.real_confirm,
        rate_hz=args.mirror_rate_hz,
        max_joint_delta_deg=args.max_joint_delta_deg,
        watchdog_timeout_sec=args.watchdog_timeout_sec,
        disable_gripper_real=args.disable_gripper_real,
        helper_max_relative_target_deg=args.real_helper_max_rel_deg,
        start_pose_max_joint_delta_deg=args.start_pose_max_joint_delta_deg,
        start_pose_gripper_max_delta_deg=args.start_pose_gripper_max_delta_deg,
        start_pose_rate_hz=args.start_pose_rate_hz,
        host=args.real_host,
        user=args.real_user,
        repo=args.real_repo,
        helper=args.real_helper,
        connect_timeout_sec=args.real_connect_timeout_sec,
        request_timeout_sec=args.real_request_timeout_sec,
        start_pose_tolerance_deg=args.real_start_pose_tolerance_deg,
        start_pose_timeout_sec=args.real_start_pose_timeout_sec,
        start_pose_hold_sec=args.real_start_pose_hold_sec,
        start_pose_samples=args.real_start_pose_samples,
        start_pose_duration_sec=args.real_start_pose_duration_sec,
        first_target_tolerance_deg=args.first_real_target_tolerance_deg,
        connect_retries=args.real_connect_retries,
        connect_retry_delay_sec=args.real_connect_retry_delay_sec,
        hold_interval_sec=args.hold_interval_sec,
    )


def _validate_real_flags(args: argparse.Namespace) -> None:
    real_requested = args.mirror_real or args.read_real_state or args.prepare_real_start_pose
    if args.prepare_real_start_pose and not args.mirror_real:
        raise SystemExit("--prepare-real-start-pose can move the real robot and therefore requires --mirror-real.")
    if args.real_preflight_only and not args.mirror_real:
        raise SystemExit("--real-preflight-only requires --mirror-real.")
    if args.real_replay_after_sim_success and not args.mirror_real:
        raise SystemExit("--real-replay-after-sim-success requires --mirror-real.")
    if args.mirror_real and not args.prepare_real_start_pose:
        raise SystemExit("--mirror-real requires --prepare-real-start-pose before streaming policy targets.")
    if real_requested and args.real_confirm != REQUIRED_REAL_CONFIRMATION:
        raise SystemExit(
            "Refusing real robot access. Pass --real-confirm "
            f"{REQUIRED_REAL_CONFIRMATION!r} only while physically at the robot with e-stop ready."
        )


def _write_ppm(path: Path, rgb: "Any") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    with path.open("wb") as handle:
        handle.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        handle.write(rgb[..., :3].astype("uint8").tobytes())


def _write_pgm(path: Path, depth: "Any") -> dict[str, float]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.asarray(depth, dtype=np.float32).squeeze()
    finite = np.isfinite(d) & (d > 0.0)
    if finite.any():
        lo = float(d[finite].min())
        hi = float(d[finite].max())
        span = max(hi - lo, 1e-6)
        img = np.zeros(d.shape, dtype=np.uint8)
        img[finite] = np.clip((d[finite] - lo) / span * 255.0, 0.0, 255.0).astype(np.uint8)
        mean = float(d[finite].mean())
    else:
        lo = hi = mean = 0.0
        img = np.zeros(d.shape, dtype=np.uint8)
    h, w = img.shape[:2]
    with path.open("wb") as handle:
        handle.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        handle.write(img.tobytes())
    return {"min_m": lo, "max_m": hi, "mean_m": mean}


def _slug(text: str | None) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", (text or "unknown").strip().lower()).strip("_")
    return safe or "unknown"


def _write_trajectory(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    _validate_real_flags(args)
    config = load_yaml_config(args.config)
    validate_scene_config(config)
    policy_rgb_resolution = tuple(int(v) for v in config["scene"]["camera"]["resolution"])
    if args.sim_camera_size_px is not None:
        if args.sim_camera_size_px <= 0:
            raise SystemExit("--sim-camera-size-px must be a positive integer.")
        config["scene"]["camera"]["resolution"] = [int(args.sim_camera_size_px), int(args.sim_camera_size_px)]
    if args.save_depth:
        config["scene"]["camera"]["data_types"] = ["rgb", "distance_to_image_plane"]

    side = config["scene"].get("active_arm", "right")
    real_side = args.real_side or side
    object_names = [obj["name"] for obj in config["objects"]]
    rng = random.Random(args.seed)
    ckpt = patch_checkpoint(_abs(args.checkpoint))

    if args.read_real_state and not args.mirror_real:
        real = RealMirrorSink(_build_real_config(args, side=side))
        try:
            state = real.read_state()
            print(
                json.dumps(
                    {
                        "real_state_deg": [round(float(v), 5) for v in state],
                        "real_side": real_side,
                        "real_gripper_disabled": args.disable_gripper_real,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        finally:
            real.close()
        return 0

    if args.mirror_real:
        print("[interactive-vla] REAL MIRROR ARMED after exact confirmation; preflight required.", file=sys.stderr, flush=True)
        if args.real_replay_after_sim_success:
            print(
                "[interactive-vla] sim-first replay mode: real arm holds while sim searches; "
                "real replay happens only after sim success.",
                file=sys.stderr,
                flush=True,
            )
    else:
        print("[interactive-vla] simulation only; no real robot commands are used.", file=sys.stderr, flush=True)
    if args.mirror_dry_run:
        print(f"[interactive-vla] dry-run mirror trace: {_abs(args.mirror_dry_run)}", file=sys.stderr, flush=True)
    if args.sim_camera_size_px is not None:
        print(
            "[interactive-vla] sim camera resolution override: "
            f"{args.sim_camera_size_px}x{args.sim_camera_size_px}; "
            f"policy RGB resized to {policy_rgb_resolution[0]}x{policy_rgb_resolution[1]}",
            file=sys.stderr,
            flush=True,
        )
    print(f"[interactive-vla] using checkpoint {ckpt}", file=sys.stderr, flush=True)

    _isaac_paths()
    from isaaclab.app import AppLauncher

    print(f"[interactive-vla] launching Isaac (headless={args.headless})", file=sys.stderr, flush=True)
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

    policy_device = args.policy_device or args.device
    if str(policy_device).startswith("cuda") and not torch.cuda.is_available():
        policy_device = "cpu"
    if not args.headless and str(policy_device).startswith("cuda"):
        print(
            "[interactive-vla] warning: --viewer plus --policy-device cuda can OOM on the 8 GB GPU. "
            "Use --policy-device cpu for the viewer workflow.",
            file=sys.stderr,
            flush=True,
        )
    dev = torch.device(policy_device)
    print("[interactive-vla] loading policy", file=sys.stderr, flush=True)
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
    viewer_eye_m = args.viewer_eye_m if args.viewer_eye_m is not None else cam["eye_m"]
    viewer_target_m = args.viewer_target_m if args.viewer_target_m is not None else cam["target_m"]
    sim.set_camera_view(eye=viewer_eye_m, target=viewer_target_m)
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    sim_dt = sim.get_physics_dt()
    render_every_sim_steps = max(1, int(args.render_every_sim_steps))
    sim_step_count = 0
    print(
        f"[interactive-vla] Isaac render interval: every {render_every_sim_steps} physics step(s) "
        "(final step of each command always renders)",
        file=sys.stderr,
        flush=True,
    )

    robot = scene["robot"]
    arm_ids, _ = robot.find_joints([f"openarm_{side}_joint{i}" for i in range(1, 8)], preserve_order=True)
    finger_ids, _ = robot.find_joints(f"openarm_{side}_finger_joint.*")
    scene_camera = scene["camera"]
    eye_w = torch.tensor([float(v) for v in cam["eye_m"]], device=robot.device).unsqueeze(0) + scene.env_origins
    tgt_w = torch.tensor([float(v) for v in cam["target_m"]], device=robot.device).unsqueeze(0) + scene.env_origins
    scene_camera.set_world_poses_from_view(eye_w, tgt_w)

    reset_deg = config["robot"]["reset_pose_deg"][side]
    reset_state = np.asarray(
        [float(reset_deg[j]) for j in JOINT_NAMES] + [float(reset_deg.get("gripper", -65.0))],
        dtype=np.float32,
    )
    debug_dir = _abs(args.save_debug_frames) if args.save_debug_frames else None

    def read_rgb() -> "np.ndarray":
        out = scene_camera.data.output["rgb"][0]
        rgb = out[..., :3]
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        image = rgb.detach().cpu().numpy()
        if image.shape[1] != policy_rgb_resolution[0] or image.shape[0] != policy_rgb_resolution[1]:
            from PIL import Image  # noqa: PLC0415

            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            image = np.asarray(
                Image.fromarray(image).resize(
                    (policy_rgb_resolution[0], policy_rgb_resolution[1]),
                    resample=resample,
                ),
                dtype=np.uint8,
            )
        return image

    def read_depth() -> "np.ndarray | None":
        if "distance_to_image_plane" not in scene_camera.data.output:
            return None
        return scene_camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy()

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

    def apply_sim(state8: "np.ndarray | list[float]", n: int) -> None:
        nonlocal sim_step_count
        arm = torch.tensor(
            [[float(np.radians(state8[i])) for i in range(7)]],
            device=robot.device,
            dtype=torch.float32,
        )
        fm = gripper_deg_to_sim_finger_m(float(state8[7]))
        fingers = torch.full((1, len(finger_ids)), float(fm), device=robot.device, dtype=torch.float32)
        for step_offset in range(n):
            robot.set_joint_position_target(arm, joint_ids=arm_ids)
            robot.set_joint_position_target(fingers, joint_ids=finger_ids)
            scene.write_data_to_sim()
            sim_step_count += 1
            should_render = (sim_step_count % render_every_sim_steps == 0) or (step_offset == n - 1)
            sim.step(render=should_render)
            scene.update(sim_dt)

    def read_state() -> "np.ndarray":
        joints = robot.data.joint_pos[:, arm_ids][0].detach().cpu().numpy()
        fingers = robot.data.joint_pos[:, finger_ids][0].detach().cpu().numpy()
        gripper_deg = sim_finger_m_to_gripper_deg(float(fingers.mean()))
        return np.asarray([float(np.degrees(v)) for v in joints] + [gripper_deg], dtype=np.float32)

    def object_z(name: str) -> float:
        return float(scene[name].data.root_pos_w[0, 2].item())

    def reset_random_scene() -> dict[str, list[float]]:
        poses = _jittered_poses(config, rng=rng, jitter_x_m=args.jitter_x_m, jitter_y_m=args.jitter_y_m)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        place_objects(poses)
        apply_sim(reset_state, args.settle_steps)
        print("\n[interactive-vla] reset to random object poses:", file=sys.stderr, flush=True)
        for name in object_names:
            x, y, z = poses[name]
            print(f"  {name}: [{x:+.3f}, {y:+.3f}, {z:+.3f}] m", file=sys.stderr, flush=True)
        return poses

    def maybe_save_frame(task_index: int, phase: str) -> None:
        if debug_dir is None:
            return
        rgb_path = debug_dir / f"task{task_index:03d}_{phase}_rgb.ppm"
        _write_ppm(rgb_path, read_rgb())
        print(f"[interactive-vla] saved {rgb_path}", file=sys.stderr, flush=True)
        if args.save_depth:
            depth = read_depth()
            if depth is not None:
                depth_path = debug_dir / f"task{task_index:03d}_{phase}_depth.pgm"
                stats = _write_pgm(depth_path, depth)
                print(
                    f"[interactive-vla] saved {depth_path} "
                    f"(depth min/mean/max={stats['min_m']:.3f}/{stats['mean_m']:.3f}/{stats['max_m']:.3f} m)",
                    file=sys.stderr,
                    flush=True,
                )

    poses = reset_random_scene()
    sinks = [SimSink(lambda command: apply_sim(command, args.substeps))]
    if args.mirror_dry_run:
        sinks.append(DryRunMirrorSink(args.mirror_dry_run, side=side, disable_gripper_real=args.disable_gripper_real))
    real_sink: RealMirrorSink | None = None
    if args.mirror_real:
        real_config = _build_real_config(args, side=side)
        start_pose_deg = start_pose_from_config(args.config, real_config.side)
        real_sink = RealMirrorSink(real_config)
        try:
            ready = real_sink.start()
            print(
                "[interactive-vla] real helper ready; current state read back.",
                file=sys.stderr,
                flush=True,
            )
            if args.read_real_state:
                print(
                    json.dumps(
                        {
                            "real_state_deg": [round(float(v), 5) for v in ready.get("state_deg", [])],
                            "real_side": real_config.side,
                            "real_gripper_disabled": args.disable_gripper_real,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            print(
                "[interactive-vla] moving real arm to configured start pose "
                f"(side={real_config.side}, gripper_disabled={args.disable_gripper_real})",
                file=sys.stderr,
                flush=True,
            )
            real_sink.prepare_start_pose(start_pose_deg)
            summary = real_sink.audit_prepared_start_pose(start_pose_deg)
            print(
                "[interactive-vla] real start-pose audit passed: "
                + json.dumps(summary, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            if args.real_replay_after_sim_success:
                print(
                    "[interactive-vla] real arm prepared and holding; live mirror disabled until sim success.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                sinks.append(real_sink)
        except Exception:
            real_sink.close()
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
            raise
    command_sink = CompositeSink(sinks)
    if args.real_preflight_only:
        print("[interactive-vla] real preflight-only requested; exiting before typed tasks.", file=sys.stderr, flush=True)
        command_sink.close()
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        return 0
    task_index = 0
    awaiting_reset_after_task = False
    try:
        while True:
            if awaiting_reset_after_task:
                prompt = "\nTask finished and holding. Type reset to start again, hold, or q to release: "
            else:
                prompt = "\nType task (orange/red/green/blue, reset, hold, or q to release): "
            typed = input(prompt).strip()
            if typed.lower() in {"q", "quit", "exit"}:
                if real_sink is not None:
                    print(
                        "[interactive-vla] q/quit requested; closing real helper releases torque.",
                        file=sys.stderr,
                        flush=True,
                    )
                break
            if not typed:
                continue
            if typed.lower() in {"h", "hold", "wait", "stay"}:
                print("[interactive-vla] holding current pose; no new policy command.", file=sys.stderr, flush=True)
                continue
            if typed.lower() in {"r", "reset", "home", "start"}:
                if real_sink is not None:
                    print(
                        "[interactive-vla] reset requested; moving real arm back to configured start pose.",
                        file=sys.stderr,
                        flush=True,
                    )
                    real_sink.prepare_start_pose(start_pose_from_config(args.config, real_sink.config.side))
                poses = reset_random_scene()
                print(
                    "[interactive-vla] reset complete; robot is at start pose and holding.",
                    file=sys.stderr,
                    flush=True,
                )
                awaiting_reset_after_task = False
                continue
            if awaiting_reset_after_task:
                print(
                    "[interactive-vla] previous task is finished and holding. "
                    "Type reset before starting a new task, hold to keep holding, or q to release.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            task_index += 1
            target: str | None
            try:
                target = parse_target_object(typed)
            except LanguageError as exc:
                target = None
                print(
                    f"[interactive-vla] free-form command; no single scoring target inferred ({exc})",
                    file=sys.stderr,
                    flush=True,
                )
            instruction = typed if args.use_typed_language or target is None else instruction_for_object(target)
            print(
                f"[interactive-vla] task {task_index}: typed={typed!r}, policy_instruction={instruction!r}",
                file=sys.stderr,
                flush=True,
            )
            maybe_save_frame(task_index, "initial")
            baseline = {name: object_z(name) for name in object_names}
            policy.reset()
            pre.reset()
            post.reset()
            state = read_state()
            clamp_events = 0
            trajectory_records: list[dict[str, Any]] = []
            for step_index in range(args.steps_per_task):
                obs = {
                    "observation.images.camera1": read_rgb(),
                    "observation.state": state.astype(np.float32),
                }
                action = predict_action(obs, policy, dev, pre, post, use_amp=False, task=instruction)
                raw_np = action.detach().cpu().numpy().astype(np.float32)
                raw = raw_np.reshape(-1, raw_np.shape[-1])[0]
                arm_raw = {JOINT_NAMES[i]: float(raw[i]) for i in range(7)}
                arm_clamped = clamp_joint_targets(side, arm_raw)
                grip = clamp(float(raw[7]), *SAFE_GRIPPER_LIMIT_DEG)
                if any(abs(arm_clamped[j] - arm_raw[j]) > 1e-5 for j in JOINT_NAMES):
                    clamp_events += 1
                if abs(grip - float(raw[7])) > 1e-5:
                    clamp_events += 1
                command = np.asarray([arm_clamped[j] for j in JOINT_NAMES] + [grip], dtype=np.float32)
                command_list = [float(value) for value in command]
                raw_command_list = [float(value) for value in raw[:8]]
                trajectory_records.append(
                    {
                        "type": "command",
                        "source": "vla_policy",
                        "sequence": step_index,
                        "task_index": task_index,
                        "step_index": step_index,
                        "typed_task": typed,
                        "policy_instruction": instruction,
                        "target_object": target,
                        "side": side,
                        "observation_state_deg": [round(float(value), 6) for value in state.tolist()],
                        "raw_policy_command_deg": [round(float(value), 6) for value in raw_command_list],
                        "clamped_policy_command_deg": [round(float(value), 6) for value in command_list],
                        "command_deg": [round(float(value), 6) for value in command_list],
                        "command_deg_note": "same as clamped_policy_command_deg; used for sim apply and optional real replay",
                        "real_gripper_disabled": args.disable_gripper_real,
                        "gripper_sent_to_real": False,
                    }
                )
                command_sink.emit(
                    command_list,
                    CommandContext(
                        task_index=task_index,
                        step_index=step_index,
                        typed_task=typed,
                        policy_instruction=instruction,
                        target_object=target,
                    ),
                )
                state = read_state()

            final_z = {name: object_z(name) for name in object_names}
            rises = {name: final_z[name] - baseline[name] for name in object_names}
            success = None if target is None else rises[target] > args.lift_threshold_m
            wrong = None if target is None else any(
                rise > args.lift_threshold_m for name, rise in rises.items() if name != target
            )
            trajectory_dir = _abs(args.save_trajectory_dir)
            trajectory_name = f"task{task_index:03d}_{_slug(target or typed)}"
            trajectory_path = trajectory_dir / f"{trajectory_name}_commands.jsonl"
            summary_path = trajectory_dir / f"{trajectory_name}_summary.json"
            _write_trajectory(trajectory_records, trajectory_path)
            maybe_save_frame(task_index, "final")
            result_payload = {
                "task_index": task_index,
                "target": target,
                "success": None if success is None else bool(success),
                "wrong_object_lifted": None if wrong is None else bool(wrong),
                "target_rise_m": None if target is None else round(float(rises[target]), 5),
                "object_rises_m": {name: round(float(rise), 5) for name, rise in rises.items()},
                "limit_clamp_events": int(clamp_events),
                "object_poses_m": {name: [round(float(v), 5) for v in pose] for name, pose in poses.items()},
                "trajectory_jsonl": str(trajectory_path),
                "trajectory_summary": str(summary_path),
            }
            real_replay_status = "not_requested"
            if args.real_replay_after_sim_success and real_sink is not None:
                if success is True and wrong is False:
                    print(
                        "[interactive-vla] sim success; saved trajectory is ready for real replay.",
                        file=sys.stderr,
                        flush=True,
                    )
                    answer = input(
                        "Type RUN to replay this saved trajectory on the real robot, "
                        "or anything else to keep holding and skip: "
                    ).strip()
                    if answer == "RUN":
                        print(
                            "[interactive-vla] RUN confirmed; replaying saved trajectory on real arm, "
                            "then holding final pose.",
                            file=sys.stderr,
                            flush=True,
                        )
                        real_sink.prepare_start_pose(start_pose_from_config(args.config, real_sink.config.side))
                        for record in trajectory_records:
                            real_sink.emit(
                                [float(value) for value in record["command_deg"]],
                                CommandContext(
                                    task_index=task_index,
                                    step_index=int(record["step_index"]),
                                    typed_task=typed,
                                    policy_instruction=instruction,
                                    target_object=target,
                                ),
                            )
                        real_replay_status = "operator_confirmed_replayed_and_holding"
                    else:
                        print(
                            "[interactive-vla] real replay skipped by operator; arm keeps holding.",
                            file=sys.stderr,
                            flush=True,
                        )
                        real_replay_status = "operator_skipped_real_replay"
                else:
                    print(
                        "[interactive-vla] sim did not produce a clean success; real replay skipped and arm keeps holding.",
                        file=sys.stderr,
                        flush=True,
                    )
                    real_replay_status = "skipped_sim_not_successful"
            result_payload["real_replay_status"] = real_replay_status
            _write_summary(result_payload, summary_path)
            print(json.dumps(result_payload, indent=2, sort_keys=True), flush=True)
            awaiting_reset_after_task = True
            if real_sink is not None:
                print(
                    "[interactive-vla] task finished; real arm is holding its current commanded pose. "
                    "The sim scene is also holding the final pose. Type reset before another task, "
                    "hold to keep holding, or q to release.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    "[interactive-vla] task finished; sim scene is holding the final pose. "
                    "Type reset before another task, hold to keep holding, or q to quit.",
                    file=sys.stderr,
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[interactive-vla] interrupted", file=sys.stderr, flush=True)
    except MirrorSafetyError as exc:
        print(f"\n[interactive-vla] real mirror safety abort: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        try:
            command_sink.close()
        except Exception as exc:  # noqa: BLE001 - cleanup warning before Isaac close
            print(f"[interactive-vla] mirror cleanup warning: {exc}", file=sys.stderr, flush=True)
        if args.real_replay_after_sim_success and real_sink is not None:
            try:
                real_sink.close()
            except Exception as exc:  # noqa: BLE001 - cleanup warning before Isaac close
                print(f"[interactive-vla] real replay cleanup warning: {exc}", file=sys.stderr, flush=True)
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
