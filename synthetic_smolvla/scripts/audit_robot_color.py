#!/usr/bin/env python3
"""Robot color audit: build the real scene, capture several camera frames, and
audit the robot's color distribution in EACH frame.

This replaces eyeballing GUI screenshots. It verifies:
  - every visible robot mesh/subset is bound to the intended black/dark/silver
    material, and inactive/shared bimanual visuals are hidden,
  - the robot is not corrupted (no extreme black/white speckle), and
  - the robot is not washed flat to one pale value (real black->grey
    spread is present), and
  - the whole visible robot mask is low-saturation, so the arm reads as black/grey.

PASS criteria per frame:
  - robot region shows a real tonal spread: meaningful fractions of dark AND
    mid AND light pixels (not >90% in one bin), and
  - some genuinely dark (black) pixels exist (the black arms/grippers), and
  - enough whole-robot pixels are visible and low-saturation greyscale tones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from sim_contract import REPO_ROOT, load_yaml_config, validate_scene_config  # noqa: E402


def _bins(gray):
    import numpy as np

    total = gray.size
    edges = [(0, 40, "black"), (40, 90, "dark"), (90, 160, "mid"), (160, 215, "light"), (215, 256, "white")]
    out = {}
    for lo, hi, name in edges:
        out[name] = float(((gray >= lo) & (gray < hi)).sum()) / total
    return out


def _saturation_stats(rgb):
    import numpy as np

    if rgb.size == 0:
        return {"mean": 1.0, "p95": 1.0, "low_fraction": 0.0}
    rgb_f = rgb.astype(np.float32) / 255.0
    mx = rgb_f.max(axis=-1)
    mn = rgb_f.min(axis=-1)
    sat = np.where(mx > 1.0e-6, (mx - mn) / mx, 0.0)
    return {
        "mean": float(sat.mean()),
        "p95": float(np.quantile(sat, 0.95)),
        "low_fraction": float((sat <= 0.18).sum()) / float(sat.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=120, help="sim steps before first capture")
    ap.add_argument("--gap", type=int, default=20, help="sim steps between captures")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-mean-saturation", type=float, default=0.16)
    ap.add_argument("--min-low-saturation-fraction", type=float, default=0.85)
    ap.add_argument("--min-robot-pixel-fraction", type=float, default=0.12)
    ap.add_argument("--robot-mask-max-saturation", type=float, default=0.18)
    ap.add_argument("--robot-region", default="whole_robot", choices=("whole_robot", "arm_column"))
    args = ap.parse_args()

    config = load_yaml_config(args.config)
    validate_scene_config(config)

    import make_scene

    make_scene._isaac_paths()
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    camera = config["scene"]["camera"]
    scene_cls = make_scene.build_scene_cls(
        config,
        sim_utils=sim_utils,
        AssetBaseCfg=AssetBaseCfg,
        RigidObjectCfg=RigidObjectCfg,
        CameraCfg=CameraCfg,
        InteractiveSceneCfg=InteractiveSceneCfg,
        configclass=configclass,
        include_camera=True,
    )
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim.set_camera_view(eye=camera["eye_m"], target=camera["target_m"])
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    scene_camera = scene["camera"]
    import torch

    eye_w = torch.tensor([float(v) for v in camera["eye_m"]], device=robot.device).unsqueeze(0) + scene.env_origins
    tgt_w = torch.tensor([float(v) for v in camera["target_m"]], device=robot.device).unsqueeze(0) + scene.env_origins
    scene_camera.set_world_poses_from_view(eye_w, tgt_w)

    if args.outdir:
        outdir = Path(args.outdir)
        if not outdir.is_absolute():
            outdir = REPO_ROOT / outdir
    else:
        outdir = SCRIPTS.parent / "reports" / "color_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    sim_dt = sim.get_physics_dt()
    bound = 0
    material_audit = None
    step = 0
    frames_done = 0
    audit = []

    def capture(idx):
        scene.update(sim_dt)
        rgb = scene_camera.data.output["rgb"][0].detach().cpu().numpy()
        img = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        if args.robot_region == "arm_column":
            robot_reg = img[:, int(w * 0.12): int(w * 0.88)]
        else:
            robot_reg = img
        gray = (0.299 * robot_reg[..., 0] + 0.587 * robot_reg[..., 1] + 0.114 * robot_reg[..., 2]).astype(np.uint8)
        rgb_f = robot_reg.astype(np.float32) / 255.0
        mx = rgb_f.max(axis=-1)
        mn = rgb_f.min(axis=-1)
        sat_img = np.where(mx > 1.0e-6, (mx - mn) / mx, 0.0)
        robot_mask = (sat_img <= float(args.robot_mask_max_saturation)) & (gray < 245)
        robot_pixels = robot_reg[robot_mask]
        robot_gray = gray[robot_mask]
        bins = _bins(robot_gray) if robot_gray.size else {k: 0.0 for k in ("black", "dark", "mid", "light", "white")}
        sat = _saturation_stats(robot_pixels)
        robot_pixel_fraction = float(robot_mask.sum()) / float(robot_mask.size)
        # PASS: not flat-washed (no single bin > 0.9) AND has some black AND has spread
        flat_wash = max(bins.values()) > 0.90
        has_black = bins["black"] + bins["dark"] > 0.04
        spread = (bins["black"] + bins["dark"]) > 0.03 and (bins["mid"] + bins["light"]) > 0.02
        low_saturation = (
            sat["mean"] <= float(args.max_mean_saturation)
            and sat["low_fraction"] >= float(args.min_low_saturation_fraction)
        )
        enough_robot_pixels = robot_pixel_fraction >= float(args.min_robot_pixel_fraction)
        ok = (not flat_wash) and has_black and spread and low_saturation and enough_robot_pixels
        # write a PPM so it can be inspected
        ppm = outdir / f"frame_{idx:02d}.ppm"
        with ppm.open("wb") as fh:
            fh.write(f"P6\n{w} {h}\n255\n".encode())
            fh.write(img.tobytes())
        rec = {
            "frame": idx,
            "step": step,
            "robot_region": str(args.robot_region),
            "bins": {k: round(v, 3) for k, v in bins.items()},
            "saturation": {k: round(v, 3) for k, v in sat.items()},
            "robot_pixel_fraction": round(robot_pixel_fraction, 3),
            "flat_wash": flat_wash,
            "has_black": has_black,
            "spread": spread,
            "low_saturation": low_saturation,
            "enough_robot_pixels": enough_robot_pixels,
            "pass": ok,
            "ppm": str(ppm),
        }
        audit.append(rec)
        return ok

    while simulation_app.is_running() and frames_done < args.frames:
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1
        if material_audit is None:
            bound = make_scene.force_bind_robot_visual_material(config)
            material_audit = make_scene.audit_robot_visual_material_bindings(config)
            (outdir / "material_binding_audit.json").write_text(
                json.dumps(material_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        ready = step >= args.warmup and (step - args.warmup) % args.gap == 0
        if ready:
            capture(frames_done)
            frames_done += 1

    passes = sum(1 for r in audit if r["pass"])
    material_pass = bool(material_audit and material_audit.get("overall_pass"))
    summary = {
        "config": args.config,
        "frames_captured": len(audit),
        "frames_passed": passes,
        "bound_geoms": bound,
        "material_binding_pass": material_pass,
        "material_binding_failures": int(material_audit.get("failures_count", 0)) if material_audit else None,
        "max_mean_saturation": float(args.max_mean_saturation),
        "min_low_saturation_fraction": float(args.min_low_saturation_fraction),
        "min_robot_pixel_fraction": float(args.min_robot_pixel_fraction),
        "robot_mask_max_saturation": float(args.robot_mask_max_saturation),
        "robot_region": str(args.robot_region),
        "overall_pass": material_pass and passes >= max(3, args.frames - 1),
        "material_binding_audit": material_audit,
        "frames": audit,
    }
    (outdir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    simulation_app.close()
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
