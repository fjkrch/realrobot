#!/usr/bin/env python3
"""Validate or launch the four-object OpenArm Isaac scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from sim_contract import (
    CONFIG_DIR,
    JOINT_NAMES,
    PROJECT_ROOT,
    gripper_deg_to_sim_finger_m,
    limits_summary,
    load_yaml_config,
    validate_scene_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "scene_openarm_four_objects.yaml"),
        help="scene YAML to validate",
    )
    parser.add_argument("--manifest", default=None, help="optional path for a validated scene manifest JSON")
    parser.add_argument("--dry-run", action="store_true", help="validate config only and do not import Isaac")
    parser.add_argument("--steps", type=int, default=240, help="number of Isaac sim steps to run")
    parser.add_argument("--headless", action="store_true", help="run Isaac without opening the viewer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-camera-rgb", default=None, help="optional PPM path for the scene camera RGB frame")
    return parser


def _write_manifest(config: dict, manifest_path: str | None, status: str) -> Path | None:
    manifest = {
        "scene": config["scene"],
        "robot": config["robot"],
        "objects": config["objects"],
        "simulation_limit_contract": limits_summary(),
        "status": status,
    }
    if not manifest_path:
        return None
    resolved = Path(manifest_path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def _isaac_paths() -> None:
    isaaclab_root = Path("/home/chyanin/IsaacLab")
    for path in [
        isaaclab_root / "source" / "isaaclab",
        isaaclab_root / "source" / "isaaclab_assets",
        isaaclab_root / "source" / "isaaclab_tasks",
        PROJECT_ROOT,
    ]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _openarm_robot_cfg(config: dict):
    from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG, OPENARM_UNI_HIGH_PD_CFG

    # World mount pose for the base. The floor mount (default 0,0,0) cannot reach
    # the table at z=0.40, so the scene config may raise the base to table height.
    base_pose = config.get("robot", {}).get("base_pose_m")

    setup = config["scene"].get("openarm_setup", "bimanual")
    if setup == "bimanual":
        robot_cfg = OPENARM_BI_HIGH_PD_CFG.copy()
        joint_pos = {}
        for side in ("left", "right"):
            reset = config["robot"]["reset_pose_deg"][side]
            for index, joint in enumerate(JOINT_NAMES, start=1):
                joint_pos[f"openarm_{side}_joint{index}"] = math.radians(float(reset[joint]))
            joint_pos[f"openarm_{side}_finger_joint.*"] = gripper_deg_to_sim_finger_m(
                float(reset.get("gripper", -65.0))
            )
        robot_cfg.init_state.joint_pos = joint_pos
    else:
        robot_cfg = OPENARM_UNI_HIGH_PD_CFG.copy()
        reset = config["robot"]["reset_pose_deg"][config["scene"].get("active_arm", "right")]
        joint_pos = {
            f"openarm_joint{idx}": math.radians(float(reset[name]))
            for idx, name in enumerate(JOINT_NAMES, start=1)
        }
        joint_pos["openarm_finger_joint.*"] = gripper_deg_to_sim_finger_m(float(reset.get("gripper", -65.0)))
        robot_cfg.init_state.joint_pos = joint_pos
    if base_pose is not None:
        robot_cfg.init_state.pos = tuple(float(v) for v in base_pose)
    return robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _shape_spawn(sim_utils, obj: dict):
    material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=tuple(float(c) for c in obj["color_rgb"]),
        roughness=0.55,
    )
    common = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.03 if obj["shape"] == "sphere" else 0.05),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        "visual_material": material,
    }
    if obj["shape"] == "sphere":
        return sim_utils.SphereCfg(radius=float(obj["radius_m"]), **common)
    return sim_utils.CuboidCfg(size=tuple(float(v) for v in obj["size_m"]), **common)


def _add_visual_cuboid(
    attrs: dict,
    *,
    name: str,
    sim_utils,
    AssetBaseCfg,
    size_m: list[float] | tuple[float, float, float],
    pose_m: list[float] | tuple[float, float, float],
    color_rgb: list[float] | tuple[float, float, float],
    roughness: float = 0.8,
    collision: bool = False,
) -> None:
    spawn_kwargs = {
        "size": tuple(float(v) for v in size_m),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(float(c) for c in color_rgb),
            roughness=float(roughness),
        ),
    }
    if collision:
        spawn_kwargs["collision_props"] = sim_utils.CollisionPropertiesCfg()
    attrs[name] = AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(**spawn_kwargs),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in pose_m)),
    )
    attrs["__annotations__"][name] = AssetBaseCfg


def _unit_noise(index: int, salt: float) -> float:
    value = math.sin((index + 1) * salt) * 43758.5453123
    return value - math.floor(value)


def _add_real_appearance_assets(attrs: dict, config: dict, *, sim_utils, AssetBaseCfg) -> None:
    scene_config = config["scene"]
    appearance = scene_config.get("appearance", {})
    table = scene_config["table"]
    table_size = [float(v) for v in table["size_m"]]
    table_pose = [float(v) for v in table["pose_m"]]

    floor = appearance.get("floor")
    if floor and floor.get("enabled", True):
        floor_size = [float(v) for v in floor.get("size_m", [5.0, 4.0, 0.02])]
        floor_pose = [float(v) for v in floor.get("pose_m", [1.25, 0.0, -0.008])]
        _add_visual_cuboid(
            attrs,
            name="real_floor",
            sim_utils=sim_utils,
            AssetBaseCfg=AssetBaseCfg,
            size_m=floor_size,
            pose_m=floor_pose,
            color_rgb=floor.get("color_rgb", [0.50, 0.50, 0.50]),
            roughness=float(floor.get("roughness", 0.95)),
            collision=False,
        )
        speckles = floor.get("speckles", {})
        count = int(speckles.get("count", 0))
        if count > 0:
            margin = float(speckles.get("margin_m", 0.15))
            min_size = float(speckles.get("min_size_m", 0.006))
            max_size = float(speckles.get("max_size_m", 0.020))
            thickness = float(speckles.get("thickness_m", 0.001))
            cmin = [float(v) for v in speckles.get("color_rgb_min", [0.38, 0.38, 0.38])]
            cmax = [float(v) for v in speckles.get("color_rgb_max", [0.68, 0.68, 0.68])]
            top_z = floor_pose[2] + floor_size[2] / 2.0 + float(speckles.get("z_offset_m", 0.002))
            region = speckles.get("region_m", {})
            if region:
                x_low, x_high = [float(v) for v in region.get("x", [floor_pose[0] - floor_size[0] / 2.0, floor_pose[0] + floor_size[0] / 2.0])]
                y_low, y_high = [float(v) for v in region.get("y", [floor_pose[1] - floor_size[1] / 2.0, floor_pose[1] + floor_size[1] / 2.0])]
            else:
                x_low = floor_pose[0] - floor_size[0] / 2.0 + margin
                x_high = floor_pose[0] + floor_size[0] / 2.0 - margin
                y_low = floor_pose[1] - floor_size[1] / 2.0 + margin
                y_high = floor_pose[1] + floor_size[1] / 2.0 - margin
            usable_x = max(0.1, x_high - x_low)
            usable_y = max(0.1, y_high - y_low)
            for index in range(count):
                x = x_low + _unit_noise(index, 12.9898) * usable_x
                y = y_low + _unit_noise(index, 78.233) * usable_y
                sx = min_size + _unit_noise(index, 37.719) * (max_size - min_size)
                sy = min_size + _unit_noise(index, 19.917) * (max_size - min_size)
                shade = _unit_noise(index, 53.123)
                color = [lo + shade * (hi - lo) for lo, hi in zip(cmin, cmax)]
                _add_visual_cuboid(
                    attrs,
                    name=f"floor_speckle_{index:03d}",
                    sim_utils=sim_utils,
                    AssetBaseCfg=AssetBaseCfg,
                    size_m=[sx, sy, thickness],
                    pose_m=[x, y, top_z],
                    color_rgb=color,
                    roughness=float(speckles.get("roughness", 1.0)),
                    collision=False,
                )

    rim = appearance.get("table_rim")
    if rim and rim.get("enabled", True):
        thickness = float(rim.get("thickness_m", 0.035))
        height = float(rim.get("height_m", table_size[2]))
        color = rim.get("color_rgb", [0.02, 0.02, 0.02])
        roughness = float(rim.get("roughness", 0.75))
        z = table_pose[2] + float(rim.get("z_offset_m", 0.0))
        sides = tuple(str(side).lower() for side in rim.get("sides", ["front"]))
        rim_specs = {
            "front": (
                [thickness, table_size[1] + 2.0 * thickness, height],
                [table_pose[0] - table_size[0] / 2.0 - thickness / 2.0, table_pose[1], z],
            ),
            "back": (
                [thickness, table_size[1] + 2.0 * thickness, height],
                [table_pose[0] + table_size[0] / 2.0 + thickness / 2.0, table_pose[1], z],
            ),
            "left": (
                [table_size[0], thickness, height],
                [table_pose[0], table_pose[1] - table_size[1] / 2.0 - thickness / 2.0, z],
            ),
            "right": (
                [table_size[0], thickness, height],
                [table_pose[0], table_pose[1] + table_size[1] / 2.0 + thickness / 2.0, z],
            ),
        }
        for side in sides:
            if side not in rim_specs:
                continue
            size, pose = rim_specs[side]
            _add_visual_cuboid(
                attrs,
                name=f"table_{side}_rim",
                sim_utils=sim_utils,
                AssetBaseCfg=AssetBaseCfg,
                size_m=size,
                pose_m=pose,
                color_rgb=color,
                roughness=roughness,
                collision=False,
            )

    for asset_name in ("wall", "baseboard"):
        asset = appearance.get(asset_name)
        if asset and asset.get("enabled", True):
            _add_visual_cuboid(
                attrs,
                name=f"real_{asset_name}",
                sim_utils=sim_utils,
                AssetBaseCfg=AssetBaseCfg,
                size_m=asset["size_m"],
                pose_m=asset["pose_m"],
                color_rgb=asset.get("color_rgb", [0.85, 0.85, 0.82]),
                roughness=float(asset.get("roughness", 0.85)),
                collision=False,
            )


def build_scene_cls(
    config: dict,
    *,
    sim_utils,
    AssetBaseCfg,
    RigidObjectCfg,
    CameraCfg,
    InteractiveSceneCfg,
    configclass,
):
    """Build the InteractiveSceneCfg subclass for the four-object OpenArm scene.

    Extracted so both the scene launcher and the physics-rollout evaluator use a
    single, identical scene definition (same robot, table, camera, objects).
    """
    scene_config = config["scene"]
    table = scene_config["table"]
    camera = scene_config["camera"]
    table_color = tuple(float(c) for c in table.get("color_rgb", [0.55, 0.50, 0.44]))
    # Optional config-driven dome lighting. Defaults preserve the original look; a
    # lower intensity lets a tinted surface (e.g. a cardboard box) read as its
    # diffuse color instead of being overexposed to white.
    lighting = scene_config.get("appearance", {}).get("lighting", {})
    dome_intensity = float(lighting.get("intensity", 3000.0))
    dome_color = tuple(float(c) for c in lighting.get("color", [0.75, 0.75, 0.75]))

    attrs = {
        "__annotations__": {},
        "__doc__": "Synthetic OpenArm four-object scene.",
        "ground": AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()),
        "dome_light": AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=dome_intensity, color=dome_color),
        ),
        "table": AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=tuple(float(v) for v in table["size_m"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=table_color,
                    roughness=0.7,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in table["pose_m"])),
        ),
        "robot": _openarm_robot_cfg(config),
        "camera": CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            width=int(camera["resolution"][0]),
            height=int(camera["resolution"][1]),
            data_types=list(camera.get("data_types", ["rgb"])),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(camera["focal_length_mm"]),
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in camera["eye_m"]),
                # Fixed table-facing ROS camera orientation reused from the local VLA scene.
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        ),
    }
    attrs["__annotations__"].update(
        {
            "ground": AssetBaseCfg,
            "dome_light": AssetBaseCfg,
            "table": AssetBaseCfg,
            "robot": type(attrs["robot"]),
            "camera": CameraCfg,
        }
    )
    _add_real_appearance_assets(attrs, config, sim_utils=sim_utils, AssetBaseCfg=AssetBaseCfg)
    for obj in config["objects"]:
        name = obj["name"]
        attrs[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=_shape_spawn(sim_utils, obj),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(v) for v in obj["spawn_pose_m"]),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        attrs["__annotations__"][name] = RigidObjectCfg

    return configclass(type("SyntheticOpenArmSceneCfg", (InteractiveSceneCfg,), attrs))


def run_isaac_scene(
    config: dict,
    *,
    steps: int,
    headless: bool,
    device: str,
    manifest_path: str | None = None,
    save_camera_rgb: str | None = None,
) -> dict:
    _isaac_paths()
    from isaaclab.app import AppLauncher

    print("[openarm_scene] launching Isaac app", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(headless=headless, enable_cameras=True)
    simulation_app = app_launcher.app

    print("[openarm_scene] importing Isaac Lab APIs", file=sys.stderr, flush=True)
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sensors import CameraCfg  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415

    camera = config["scene"]["camera"]
    scene_cls = build_scene_cls(
        config,
        sim_utils=sim_utils,
        AssetBaseCfg=AssetBaseCfg,
        RigidObjectCfg=RigidObjectCfg,
        CameraCfg=CameraCfg,
        InteractiveSceneCfg=InteractiveSceneCfg,
        configclass=configclass,
    )
    print("[openarm_scene] creating simulation context", file=sys.stderr, flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=camera["eye_m"], target=camera["target_m"])
    print("[openarm_scene] creating interactive scene", file=sys.stderr, flush=True)
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    print("[openarm_scene] resetting sim", file=sys.stderr, flush=True)
    sim.reset()
    print("[openarm_scene] resetting scene", file=sys.stderr, flush=True)
    scene.reset()

    robot = scene["robot"]
    scene_camera = scene["camera"]
    try:
        import torch

        eye_w = torch.tensor(
            [float(v) for v in camera["eye_m"]],
            device=robot.device,
            dtype=torch.float32,
        ).unsqueeze(0) + scene.env_origins
        target_w = torch.tensor(
            [float(v) for v in camera["target_m"]],
            device=robot.device,
            dtype=torch.float32,
        ).unsqueeze(0) + scene.env_origins
        scene_camera.set_world_poses_from_view(eye_w, target_w)
    except Exception as exc:  # noqa: BLE001 - viewport camera still works if sensor pose update is unavailable
        print(f"[openarm_scene] warning: failed to aim scene camera sensor: {exc}", file=sys.stderr, flush=True)

    object_positions = {}
    for obj in config["objects"]:
        asset = scene[obj["name"]]
        object_positions[obj["name"]] = [float(v) for v in asset.data.root_pos_w[0].detach().cpu().tolist()]

    print("[openarm_scene] stepping simulation", file=sys.stderr, flush=True)
    step_count = 0
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running() and step_count < max(1, steps):
        if sim.is_stopped():
            break
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1

    print("[openarm_scene] reading camera", file=sys.stderr, flush=True)
    camera_shape = None
    camera_rgb_path = None
    try:
        rgb = scene["camera"].data.output["rgb"]
        camera_shape = list(rgb.shape)
        if save_camera_rgb:
            import numpy as np

            image = rgb[0].detach().cpu().numpy()
            image = np.clip(image[..., :3], 0, 255).astype(np.uint8)
            out = Path(save_camera_rgb)
            if not out.is_absolute():
                out = Path.cwd() / out
            out.parent.mkdir(parents=True, exist_ok=True)
            height, width = image.shape[:2]
            with out.open("wb") as handle:
                handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
                handle.write(image.tobytes())
            camera_rgb_path = str(out)
    except Exception:
        camera_shape = None
    result = {
        "steps": step_count,
        "objects": object_positions,
        "camera_rgb_shape": camera_shape,
        "camera_rgb_path": camera_rgb_path,
        "headless": headless,
        "device": device,
    }
    written_manifest = _write_manifest(config, manifest_path, "isaac_scene_launched")
    payload = {
        "ok": True,
        "manifest": None if written_manifest is None else str(written_manifest),
        "objects": len(config["objects"]),
        "status": "isaac_scene_launched",
        "isaac": result,
    }
    print(json.dumps(payload, indent=2), flush=True)
    print("[openarm_scene] closing Isaac app", file=sys.stderr, flush=True)
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    result["_reported"] = True
    return result


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.config)
    validate_scene_config(config)

    if args.dry_run:
        manifest_path = _write_manifest(config, args.manifest, "validated_config_only")
        print(
            json.dumps(
                {
                    "ok": True,
                    "manifest": None if manifest_path is None else str(manifest_path),
                    "objects": len(config["objects"]),
                    "status": "validated_config_only",
                },
                indent=2,
            )
        )
        return 0

    result = run_isaac_scene(
        config,
        steps=args.steps,
        headless=args.headless,
        device=args.device,
        manifest_path=args.manifest,
        save_camera_rgb=args.save_camera_rgb,
    )
    if result.pop("_reported", False):
        return 0
    manifest_path = _write_manifest(config, args.manifest, "isaac_scene_launched")
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": None if manifest_path is None else str(manifest_path),
                "objects": len(config["objects"]),
                "status": "isaac_scene_launched",
                "isaac": result,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
