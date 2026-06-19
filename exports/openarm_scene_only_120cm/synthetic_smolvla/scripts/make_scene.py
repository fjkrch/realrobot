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

    attrs = {
        "__annotations__": {},
        "__doc__": "Synthetic OpenArm four-object scene.",
        "ground": AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()),
        "dome_light": AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
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
    try:
        camera_shape = list(scene["camera"].data.output["rgb"].shape)
    except Exception:
        camera_shape = None
    result = {
        "steps": step_count,
        "objects": object_positions,
        "camera_rgb_shape": camera_shape,
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
