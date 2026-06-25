#!/usr/bin/env python3
"""Validate or launch the four-object OpenArm Isaac scene."""

from __future__ import annotations

import argparse
import json
import math
import os
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

PXR_VIEWER_KIT_ARGS = (
    "--enable omni.hydra.pxr",
    "--disable omni.hydra.rtx",
    "--/renderer/enabled=pxr",
    "--/renderer/active=pxr",
    "--/renderer/warnOnRtxInit=true",
    "--/app/useFabricSceneDelegate=false",
    "--/renderer/gpuEnumeration/glInterop/enabled=true",
    "--/app/window/dpiScaleOverride=1.0",
    "--/app/window/scaleToMonitor=false",
    "--/app/renderer/resolution/width=960",
    "--/app/renderer/resolution/height=540",
    "--/app/window/width=1100",
    "--/app/window/height=760",
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
    parser.add_argument(
        "--disable-scene-camera",
        action="store_true",
        help="launch the scene without an Isaac camera sensor; useful for GUI inspection when RTX camera rendering crashes",
    )
    parser.add_argument("--experience", default="", help="optional Isaac/Kit experience file override")
    parser.add_argument("--rendering-mode", default=None, choices=("performance", "balanced", "quality"))
    parser.add_argument(
        "--viewer-renderer",
        default="rtx",
        choices=("rtx", "pxr"),
        help="GUI viewport renderer; pxr avoids the local RTX viewport crash but disables scene camera capture",
    )
    parser.add_argument(
        "--viewer-camera",
        default="real",
        choices=("real", "wide"),
        help="viewport camera pose: real uses the configured robot-view camera; wide shows the whole table scene",
    )
    parser.add_argument("--kit-args", default="", help="raw Omniverse Kit args, quoted as one string")
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
    env_root = os.environ.get("ISAACLAB_ROOT")
    if env_root:
        isaaclab_root = Path(env_root)
    else:
        candidates = [
            Path("/home/chayanin/Downloads/IsaacLab"),
            Path("/home/chayanin/IsaacLab"),
            Path("/home/chyanin/IsaacLab"),
        ]
        isaaclab_root = next((path for path in candidates if path.exists()), candidates[0])
    for path in [
        isaaclab_root / "source" / "isaaclab",
        isaaclab_root / "source" / "isaaclab_assets",
        isaaclab_root / "source" / "isaaclab_tasks",
        PROJECT_ROOT,
    ]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _merge_kit_args(*chunks: str | tuple[str, ...]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, tuple):
            parts.extend(item for item in chunk if item)
        elif chunk.strip():
            parts.append(chunk.strip())
    return " ".join(parts)


def _wide_view_camera_pose(config: dict) -> tuple[list[float], list[float]]:
    scene = config["scene"]
    table = scene["table"]
    table_pose = [float(v) for v in table["pose_m"]]
    table_size = [float(v) for v in table["size_m"]]
    object_positions = [[float(v) for v in obj["spawn_pose_m"]] for obj in config["objects"]]
    robot_pose = [float(v) for v in config.get("robot", {}).get("base_pose_m", [0.0, 0.0, 0.0])]

    min_x = min([table_pose[0] - table_size[0] / 2.0, robot_pose[0], *(pos[0] for pos in object_positions)])
    max_x = max([table_pose[0] + table_size[0] / 2.0, robot_pose[0], *(pos[0] for pos in object_positions)])
    min_y = min([table_pose[1] - table_size[1] / 2.0, robot_pose[1], *(pos[1] for pos in object_positions)])
    max_y = max([table_pose[1] + table_size[1] / 2.0, robot_pose[1], *(pos[1] for pos in object_positions)])

    target = [
        (min_x + max_x) / 2.0 - 0.12,
        (min_y + max_y) / 2.0,
        table_pose[2] + table_size[2] / 2.0,
    ]
    scene_width = max(max_x - min_x, max_y - min_y)
    eye = [
        target[0],
        min_y - max(1.9, scene_width * 1.35),
        target[2] + max(1.15, scene_width * 0.8),
    ]
    return eye, target


def _viewer_camera_pose(config: dict, mode: str) -> tuple[list[float], list[float]]:
    camera = config["scene"]["camera"]
    if mode == "wide":
        return _wide_view_camera_pose(config)
    return [float(v) for v in camera["eye_m"]], [float(v) for v in camera["target_m"]]


def _openarm_robot_cfg(config: dict):
    import isaaclab.sim as sim_utils
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
    material = config.get("scene", {}).get("appearance", {}).get("robot_material", {})
    if material and material.get("enabled", True):
        robot_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(float(v) for v in material.get("color_rgb", [0.78, 0.78, 0.76])),
            roughness=float(material.get("roughness", 0.48)),
            metallic=float(material.get("metallic", 0.0)),
        )
        robot_cfg.spawn.visual_material_path = str(material.get("path", "clean_robot_material"))
    return robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


REAL_ARM_BLACK_MATERIAL_PATH = "/World/Looks/OpenArmRealBlack"
REAL_ARM_DARK_MATERIAL_PATH = "/World/Looks/OpenArmRealDarkGrey"
REAL_ARM_SILVER_MATERIAL_PATH = "/World/Looks/OpenArmRealSilver"
REAL_ARM_MATERIAL_PATHS = (
    REAL_ARM_BLACK_MATERIAL_PATH,
    REAL_ARM_DARK_MATERIAL_PATH,
    REAL_ARM_SILVER_MATERIAL_PATH,
)
DEFAULT_CAMERA_HORIZONTAL_APERTURE_MM = 20.955


def _camera_horizontal_aperture_mm(camera: dict) -> float:
    return float(camera.get("horizontal_aperture_mm", DEFAULT_CAMERA_HORIZONTAL_APERTURE_MM))


def _camera_focal_length_mm(camera: dict) -> float:
    if "horizontal_fov_deg" not in camera:
        return float(camera["focal_length_mm"])
    horizontal_fov_deg = float(camera["horizontal_fov_deg"])
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError(f"camera.horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}")
    horizontal_aperture_mm = _camera_horizontal_aperture_mm(camera)
    return horizontal_aperture_mm / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def _is_shared_bimanual_body_visual_path(prim_path: str) -> bool:
    """The body_link0 mesh contains the two-arm shoulder shell."""
    return "openarm_body_link0_visual" in prim_path.lower()


def _is_real_arm_proxy_path(prim_path: str) -> bool:
    return "real_arm_solid_proxy" in prim_path.lower()


def _is_inactive_robot_visual_path(prim_path: str, hidden_arm: str) -> bool:
    return f"/openarm_{hidden_arm}_" in prim_path.lower()


def _real_arm_material_for_path(prim_path: str, active_arm: str) -> str:
    lower = prim_path.lower()
    active_prefix = f"openarm_{active_arm}_"
    if "finger" in lower or f"{active_prefix}link7" in lower:
        return REAL_ARM_BLACK_MATERIAL_PATH
    if (
        f"{active_prefix}link1" in lower
        or f"{active_prefix}link2" in lower
        or f"{active_prefix}link3" in lower
        or f"{active_prefix}link4" in lower
        or f"{active_prefix}link5" in lower
        or f"{active_prefix}link6" in lower
    ):
        return REAL_ARM_BLACK_MATERIAL_PATH
    if "openarm_body_link" in lower or "link0" in lower:
        return REAL_ARM_DARK_MATERIAL_PATH
    return REAL_ARM_BLACK_MATERIAL_PATH


def _should_hide_robot_visual_path(prim_path: str, *, hidden_arm: str, hide_inactive: bool) -> bool:
    if not hide_inactive:
        return False
    return _is_inactive_robot_visual_path(prim_path, hidden_arm) or _is_shared_bimanual_body_visual_path(prim_path)


def _visual_owner_path(prim_path: str) -> str:
    parts = prim_path.split("/")
    for index, part in enumerate(parts):
        if part == "visuals" and index + 1 < len(parts):
            return "/".join(parts[: index + 2])
    return prim_path


def _robot_link_owner_path(prim_path: str) -> str:
    parts = prim_path.split("/")
    for index, part in enumerate(parts):
        if part == "Robot" and index + 1 < len(parts):
            return "/".join(parts[: index + 2])
    return prim_path


def _robot_material_output_color(diffuse: tuple[float, float, float], palette: str) -> tuple[float, float, float]:
    """Map the original flat OpenArm material color to a clean render-safe tone."""
    lum = sum(diffuse) / 3.0
    if palette == "black_grey_real_arm":
        if lum <= 0.18:
            tone = 0.010
        elif lum <= 0.35:
            tone = 0.025
        elif lum <= 0.70:
            tone = 0.62
        elif lum <= 0.92:
            tone = 0.72
        else:
            tone = 0.82
        return (tone, tone, tone)

    if lum <= 0.30:
        tone = 0.03
    elif lum <= 0.70:
        tone = 0.30
    elif lum <= 0.92:
        tone = 0.62
    else:
        tone = 0.93
    base = max(1e-4, lum)
    return tuple(max(0.0, min(1.0, c / base * tone)) for c in diffuse)


def _neutralize_robot_materials(stage, robot_root, *, roughness, metallic, palette: str) -> int:
    """Rewrite each robot Material to a clean UsdPreviewSurface, keeping its color.

    Root cause of the RTX black/white patch corruption: the official OpenArm USD
    ships ``OmniPBR.mdl`` materials. On this local RTX path the MDL surface does
    not resolve cleanly (each shader also carries ``emissive_intensity=10000``),
    which leaks through as noisy black/white speckle. The materials themselves are
    NOT textured -- every one is a flat ``diffuse_color_constant`` (the official
    robot is a two-tone black + grey design, e.g. 0.094, 0.247, 0.627, 0.796,
    0.984...). So the fix must PRESERVE each material's real color and only swap
    the broken MDL shader for a clean UsdPreviewSurface.

    For every Material under the robot we: (1) read its OmniPBR
    ``diffuse_color_constant`` before deleting anything, (2) remove the MDL shader
    children and clear the ``outputs:mdl:*`` connections so RTX cannot fall back
    to the broken MDL surface, and (3) author a UsdPreviewSurface using that same
    color (emissive off) wired to ``outputs:surface``. Geometry is untouched.
    """
    from pxr import Sdf, Usd, UsdShade

    neutralized = 0
    for prim in Usd.PrimRange(robot_root):
        if not prim.IsA(UsdShade.Material):
            continue
        mat = UsdShade.Material(prim)
        mat_path = prim.GetPath()

        # 1) Recover the material's intended flat color from the OmniPBR shader
        #    before we delete it. Fall back to a neutral grey if absent.
        diffuse = None
        for child in prim.GetChildren():
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            inp = shader.GetInput("diffuse_color_constant")
            if inp is not None and inp.Get() is not None:
                diffuse = tuple(float(v) for v in inp.Get())
                break
        if diffuse is None:
            diffuse = (0.627, 0.627, 0.627)

        # 2) Remove every Shader child (OmniPBR/MDL) and clear the MDL render
        #    context output so RTX cannot resolve the broken MDL surface.
        for child in list(prim.GetChildren()):
            if child.IsA(UsdShade.Shader):
                stage.RemovePrim(child.GetPath())
        for output_name in ("outputs:mdl:surface", "outputs:mdl:displacement", "outputs:mdl:volume"):
            out = prim.GetAttribute(output_name)
            if out and out.IsValid():
                out.ClearConnections()

        # 3) Author a clean UsdPreviewSurface. The black_grey_real_arm palette
        #    intentionally drops hue and keeps only black/grey tonal structure
        #    so the Isaac robot resembles the user's real arm photo.
        out_color = _robot_material_output_color(diffuse, palette)
        shader_path = mat_path.AppendChild("CleanPreviewSurface")
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(out_color)
        lum = sum(diffuse) / 3.0
        surface_roughness = 0.35 if palette == "black_grey_real_arm" and lum > 0.35 else float(roughness)
        surface_metallic = 0.35 if palette == "black_grey_real_arm" and lum > 0.35 else float(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(surface_roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(surface_metallic))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        neutralized += 1
    return neutralized


def force_bind_robot_visual_material(config: dict) -> int:
    """Give every spawned OpenArm visual geom a clean look without losing color.

    The official OpenArm USD is NOT textured: each Material is a flat
    ``diffuse_color_constant`` (a two-tone black + grey design). The RTX black/
    white corruption came from the OmniPBR ``.mdl`` surface not resolving locally,
    not from the colors. So by default we PRESERVE the real two-tone robot by:
      1. Neutralizing each robot Material in place -- swap the broken MDL shader
         for a clean UsdPreviewSurface that keeps the material's original color.
      2. Leaving each mesh bound to its own (now-clean) material.

    Set ``robot_material.force_single_color: true`` to instead override every
    visual geom with one shared flat color (the old behavior).
    """
    material = config.get("scene", {}).get("appearance", {}).get("robot_material", {})
    if not material or not material.get("enabled", True):
        return 0

    import isaaclab.sim as sim_utils
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    material_path = str(material.get("force_bind_path", "/World/Looks/CleanOpenArmRobot"))
    color = tuple(float(v) for v in material.get("color_rgb", [0.78, 0.78, 0.75]))
    roughness = float(material.get("roughness", 0.50))
    metallic = float(material.get("metallic", 0.0))
    palette = str(material.get("palette", "preserve_openarm")).strip()
    force_single_color = bool(material.get("force_single_color", False))
    hide_inactive = bool(material.get("hide_inactive_arm", False))
    active_arm = str(config.get("scene", {}).get("active_arm", "right")).strip().lower()
    hidden_arm = "left" if active_arm == "right" else "right"
    real_arm_palette = palette == "black_grey_real_arm"
    if force_single_color and not stage.GetPrimAtPath(material_path).IsValid():
        material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=roughness,
            metallic=metallic,
        )
        material_cfg.func(material_path, material_cfg)

    def define_preview_material(path: str, mat_color: tuple[float, float, float], mat_roughness: float, mat_metallic: float) -> None:
        mat = UsdShade.Material.Define(stage, path)
        shader_path = mat.GetPath().AppendChild("Shader")
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(mat_color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(mat_roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(mat_metallic))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    if real_arm_palette:
        for path, mat_color, mat_roughness, mat_metallic in (
            (REAL_ARM_BLACK_MATERIAL_PATH, (0.026, 0.026, 0.028), 0.82, 0.0),
            (REAL_ARM_DARK_MATERIAL_PATH, (0.055, 0.055, 0.058), 0.70, 0.0),
            (REAL_ARM_SILVER_MATERIAL_PATH, (0.72, 0.72, 0.70), 0.32, 0.45),
        ):
            define_preview_material(path, mat_color, mat_roughness, mat_metallic)

    def bind_material_tree(prim_path: str, bind_material_path: str) -> bool:
        root = stage.GetPrimAtPath(prim_path)
        mat_prim = stage.GetPrimAtPath(bind_material_path)
        if not root.IsValid() or not mat_prim.IsValid():
            return False
        material_api = UsdShade.Material(mat_prim)
        for target in list(Usd.PrimRange(root, Usd.TraverseInstanceProxies())):
            if target == root or target.IsA(UsdGeom.Subset):
                UsdShade.MaterialBindingAPI.Apply(target).Bind(
                    material_api,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )
        return True

    def clear_direct_material_binding(prim) -> None:
        rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
        if rel and rel.IsValid():
            rel.ClearTargets(True)
        for prop in list(prim.GetProperties()):
            if prop.GetName().startswith("material:binding"):
                prim.RemoveProperty(prop.GetName())

    def hide_visual_tree(prim_path: str) -> None:
        candidate_paths = {prim_path}
        parts = prim_path.split("/")
        for index, part in enumerate(parts):
            if (
                part == f"openarm_{hidden_arm}_"
                or part.startswith(f"openarm_{hidden_arm}_")
                or part == "openarm_body_link0_visual"
            ):
                candidate_paths.add("/".join(parts[: index + 1]))
        for candidate_path in sorted(candidate_paths, key=len):
            root = stage.GetPrimAtPath(candidate_path)
            if not root.IsValid():
                continue
            if root.GetName().endswith("_visual") or root.IsA(UsdGeom.Gprim):
                root.SetActive(False)
                continue
            for target in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if target.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(target).MakeInvisible()

    def create_solid_mesh_proxy(prim_path: str, bind_material_path: str) -> bool:
        source_prim = stage.GetPrimAtPath(prim_path)
        if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Mesh):
            return False
        source_mesh = UsdGeom.Mesh(source_prim)
        points = source_mesh.GetPointsAttr().Get()
        counts = source_mesh.GetFaceVertexCountsAttr().Get()
        indices = source_mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None:
            return False
        mins = [min(float(point[i]) for point in points) for i in range(3)]
        maxs = [max(float(point[i]) for point in points) for i in range(3)]
        center = [(lo + hi) * 0.5 for lo, hi in zip(mins, maxs, strict=True)]
        inflate = 1.025
        proxy_points = [
            Gf.Vec3f(
                center[0] + (float(point[0]) - center[0]) * inflate,
                center[1] + (float(point[1]) - center[1]) * inflate,
                center[2] + (float(point[2]) - center[2]) * inflate,
            )
            for point in points
        ]

        link_owner = stage.GetPrimAtPath(_robot_link_owner_path(prim_path))
        if not link_owner.IsValid():
            return False
        proxy_name = f"real_arm_solid_proxy_{source_prim.GetParent().GetName()}"
        proxy_path = link_owner.GetPath().AppendChild(proxy_name)
        proxy_mesh = UsdGeom.Mesh.Define(stage, proxy_path)
        proxy_prim = proxy_mesh.GetPrim()
        proxy_mesh.CreatePointsAttr(proxy_points)
        proxy_mesh.CreateFaceVertexCountsAttr(counts)
        proxy_mesh.CreateFaceVertexIndicesAttr(indices)
        normals = source_mesh.GetNormalsAttr().Get()
        if normals is not None:
            proxy_mesh.CreateNormalsAttr(normals)
            interpolation = source_mesh.GetNormalsInterpolation()
            if interpolation:
                proxy_mesh.SetNormalsInterpolation(interpolation)
        proxy_mins = [min(float(point[i]) for point in proxy_points) for i in range(3)]
        proxy_maxs = [max(float(point[i]) for point in proxy_points) for i in range(3)]
        proxy_mesh.CreateExtentAttr([
            Gf.Vec3f(*proxy_mins),
            Gf.Vec3f(*proxy_maxs),
        ])
        orientation = source_mesh.GetOrientationAttr().Get()
        if orientation is not None:
            proxy_mesh.CreateOrientationAttr(orientation)
        subdivision = source_mesh.GetSubdivisionSchemeAttr().Get()
        if subdivision is not None:
            proxy_mesh.CreateSubdivisionSchemeAttr(subdivision)
        proxy_mesh.CreateDoubleSidedAttr(True)

        xform_cache = UsdGeom.XformCache()
        source_world = xform_cache.GetLocalToWorldTransform(source_prim)
        owner_world = xform_cache.GetLocalToWorldTransform(link_owner)
        local_from_owner = source_world * owner_world.GetInverse()
        proxy_xform = UsdGeom.Xformable(proxy_prim)
        proxy_xform.ClearXformOpOrder()
        proxy_xform.AddTransformOp().Set(local_from_owner)
        UsdGeom.Imageable(proxy_prim).MakeVisible()
        return bind_material_tree(proxy_path.pathString, bind_material_path)

    robot_roots = [
        prim
        for prim in stage.TraverseAll()
        if prim.GetName() == "Robot" and prim.GetPath().pathString.startswith("/World/envs/")
    ]
    bound = 0
    neutralized = 0
    for robot_root in robot_roots:
        robot_path = robot_root.GetPath().pathString
        sim_utils.make_uninstanceable(robot_path, stage=stage)
        for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies()):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
        # The spawn-time fallback material is bound at the Robot root with
        # strongerThanDescendants. Keep the material prim, but clear that root
        # relationship so per-link black/silver bindings can actually resolve.
        clear_direct_material_binding(robot_root)

        # Stage 1: replace each broken MDL material with a clean UsdPreviewSurface
        # that keeps the material's ORIGINAL color (two-tone robot preserved).
        neutralized += _neutralize_robot_materials(
            stage, robot_root, roughness=roughness, metallic=metallic, palette=palette
        )

        visual_geoms: list[str] = []
        for prim in Usd.PrimRange(robot_root):
            prim_path = prim.GetPath().pathString
            if "/collisions" in prim_path:
                continue
            if prim.IsA(UsdGeom.Gprim) and "/visuals" in prim_path and not _is_real_arm_proxy_path(prim_path):
                visual_geoms.append(prim_path)

        # Stage 2: hide the inactive arm; optionally override to one flat color.
        for prim_path in visual_geoms:
            if _should_hide_robot_visual_path(prim_path, hidden_arm=hidden_arm, hide_inactive=hide_inactive):
                hide_visual_tree(prim_path)
                continue
            if force_single_color:
                success = bind_material_tree(prim_path, material_path)
                if success:
                    bound += 1
            elif real_arm_palette:
                success = create_solid_mesh_proxy(prim_path, _real_arm_material_for_path(prim_path, active_arm))
                if success:
                    hide_visual_tree(_visual_owner_path(prim_path))
                    bound += 1
            else:
                bound += 1

    print(
        f"[openarm_scene] neutralized {neutralized} robot materials in place",
        file=sys.stderr,
        flush=True,
    )

    # Force the viewport/Fabric to re-resolve materials after the rewrite.
    try:
        import omni.kit.app

        omni.kit.app.get_app().update()
    except Exception:  # noqa: BLE001 - refresh is best-effort
        pass
    return bound


def audit_robot_visual_material_bindings(config: dict) -> dict:
    """Inspect the live USD stage for robot visual material/visibility mistakes."""
    material = config.get("scene", {}).get("appearance", {}).get("robot_material", {})
    palette = str(material.get("palette", "preserve_openarm")).strip()
    if palette != "black_grey_real_arm":
        return {
            "enabled": False,
            "palette": palette,
            "overall_pass": True,
            "records": [],
            "failures": [],
        }

    import omni.usd
    from pxr import Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    hide_inactive = bool(material.get("hide_inactive_arm", False))
    active_arm = str(config.get("scene", {}).get("active_arm", "right")).strip().lower()
    hidden_arm = "left" if active_arm == "right" else "right"

    def direct_targets(prim) -> list[str]:
        rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
        if rel and rel.IsValid():
            return [str(target) for target in rel.GetTargets()]
        return []

    def computed_material_path(prim) -> str | None:
        bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if bound_material:
            return bound_material.GetPath().pathString
        return None

    records: list[dict] = []
    failures: list[dict] = []
    for robot_root in [
        prim
        for prim in stage.TraverseAll()
        if prim.GetName() == "Robot" and prim.GetPath().pathString.startswith("/World/envs/")
    ]:
        for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies()):
            prim_path = prim.GetPath().pathString
            if "/collisions" in prim_path:
                continue
            if "/visuals" not in prim_path and not _is_real_arm_proxy_path(prim_path):
                continue
            if not prim.IsA(UsdGeom.Gprim):
                continue
            visibility = str(UsdGeom.Imageable(prim).ComputeVisibility())
            if _is_real_arm_proxy_path(prim_path):
                expected_hidden = _should_hide_robot_visual_path(
                    prim_path,
                    hidden_arm=hidden_arm,
                    hide_inactive=hide_inactive,
                )
            else:
                expected_hidden = True
            expected_material = None if expected_hidden else _real_arm_material_for_path(prim_path, active_arm)
            root_targets = direct_targets(prim)
            root_computed = computed_material_path(prim)
            subset_records = []
            subset_ok = True
            for subset in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if subset == prim or not subset.IsA(UsdGeom.Subset):
                    continue
                targets = direct_targets(subset)
                computed = computed_material_path(subset)
                subset_record = {
                    "path": subset.GetPath().pathString,
                    "targets": targets,
                    "computed_material": computed,
                }
                subset_records.append(subset_record)
                if not expected_hidden and (targets != [expected_material] or computed != expected_material):
                    subset_ok = False

            if expected_hidden:
                ok = visibility == "invisible"
                reason = "hidden" if ok else "expected invisible"
            else:
                root_ok = root_targets == [expected_material]
                computed_ok = root_computed == expected_material
                visible_ok = visibility != "invisible"
                ok = visible_ok and root_ok and computed_ok and subset_ok
                reason = "ok"
                if not visible_ok:
                    reason = "expected visible"
                elif not root_ok:
                    reason = "root material mismatch"
                elif not computed_ok:
                    reason = "computed material mismatch"
                elif not subset_ok:
                    reason = "subset material mismatch"

            record = {
                "path": prim_path,
                "visibility": visibility,
                "expected_hidden": expected_hidden,
                "expected_material": expected_material,
                "root_targets": root_targets,
                "computed_material": root_computed,
                "subset_count": len(subset_records),
                "subsets": subset_records,
                "pass": ok,
                "reason": reason,
            }
            records.append(record)
            if not ok:
                failures.append(record)

    return {
        "enabled": True,
        "palette": palette,
        "active_arm": active_arm,
        "hidden_arm": hidden_arm,
        "hide_inactive_arm": hide_inactive,
        "allowed_material_paths": list(REAL_ARM_MATERIAL_PATHS),
        "records_checked": len(records),
        "failures_count": len(failures),
        "overall_pass": len(records) > 0 and not failures,
        "failures": failures,
        "records": records,
    }


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

    for index, marker in enumerate(appearance.get("table_markers", []) or []):
        if not marker or not marker.get("enabled", True):
            continue
        marker_name = str(marker.get("name", f"table_marker_{index:02d}"))
        safe_name = "".join(char if char.isalnum() or char == "_" else "_" for char in marker_name)
        _add_visual_cuboid(
            attrs,
            name=f"table_marker_{safe_name}",
            sim_utils=sim_utils,
            AssetBaseCfg=AssetBaseCfg,
            size_m=marker["size_m"],
            pose_m=marker["pose_m"],
            color_rgb=marker.get("color_rgb", [0.85, 0.82, 0.72]),
            roughness=float(marker.get("roughness", 0.85)),
            collision=bool(marker.get("collision", False)),
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
    include_camera: bool = True,
    use_distant_light: bool = False,
):
    """Build the InteractiveSceneCfg subclass for the four-object OpenArm scene.

    Extracted so both the scene launcher and the physics-rollout evaluator use a
    single, identical scene definition (same robot, table, camera, objects).
    """
    scene_config = config["scene"]
    table = scene_config["table"]
    camera = scene_config.get("camera", {})
    table_color = tuple(float(c) for c in table.get("color_rgb", [0.55, 0.50, 0.44]))
    # Optional config-driven dome lighting. Defaults preserve the original look; a
    # lower intensity lets a tinted surface (e.g. a cardboard box) read as its
    # diffuse color instead of being overexposed to white.
    lighting = scene_config.get("appearance", {}).get("lighting", {})
    dome_intensity = float(lighting.get("intensity", 3000.0))
    dome_color = tuple(float(c) for c in lighting.get("color", [0.75, 0.75, 0.75]))
    ground_config = scene_config.get("appearance", {}).get("ground", {})
    ground_enabled = ground_config.get("enabled", True)
    light_spawn = (
        sim_utils.DistantLightCfg(intensity=dome_intensity, color=dome_color)
        if use_distant_light
        else sim_utils.DomeLightCfg(intensity=dome_intensity, color=dome_color)
    )

    attrs = {
        "__annotations__": {},
        "__doc__": "Synthetic OpenArm four-object scene.",
        "dome_light": AssetBaseCfg(prim_path="/World/Light", spawn=light_spawn),
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
    }
    if ground_enabled:
        attrs["ground"] = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    attrs["__annotations__"].update({
        "dome_light": AssetBaseCfg,
        "table": AssetBaseCfg,
        "robot": type(attrs["robot"]),
    })
    if ground_enabled:
        attrs["__annotations__"]["ground"] = AssetBaseCfg
    if include_camera:
        attrs["camera"] = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            width=int(camera["resolution"][0]),
            height=int(camera["resolution"][1]),
            data_types=list(camera.get("data_types", ["rgb"])),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(_camera_focal_length_mm(camera)),
                focus_distance=400.0,
                horizontal_aperture=float(_camera_horizontal_aperture_mm(camera)),
                clipping_range=(0.1, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in camera["eye_m"]),
                # Fixed table-facing ROS camera orientation reused from the local VLA scene.
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )
        attrs["__annotations__"]["camera"] = CameraCfg
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
    include_scene_camera: bool = True,
    experience: str = "",
    rendering_mode: str | None = None,
    kit_args: str = "",
    use_distant_light: bool = False,
    viewer_eye_m: list[float] | None = None,
    viewer_target_m: list[float] | None = None,
) -> dict:
    if save_camera_rgb and not include_scene_camera:
        raise SystemExit("--save-camera-rgb requires the scene camera; remove --disable-scene-camera.")

    _isaac_paths()
    from isaaclab.app import AppLauncher

    print("[openarm_scene] launching Isaac app", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(
        headless=headless,
        enable_cameras=include_scene_camera,
        experience=experience,
        rendering_mode=rendering_mode,
        kit_args=kit_args,
    )
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
        include_camera=include_scene_camera,
        use_distant_light=use_distant_light,
    )
    print("[openarm_scene] creating simulation context", file=sys.stderr, flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    viewer_eye = viewer_eye_m if viewer_eye_m is not None else camera["eye_m"]
    viewer_target = viewer_target_m if viewer_target_m is not None else camera["target_m"]
    sim.set_camera_view(eye=viewer_eye, target=viewer_target)
    print("[openarm_scene] creating interactive scene", file=sys.stderr, flush=True)
    scene = InteractiveScene(scene_cls(num_envs=1, env_spacing=2.0))
    print("[openarm_scene] resetting sim", file=sys.stderr, flush=True)
    sim.reset()
    print("[openarm_scene] resetting scene", file=sys.stderr, flush=True)
    scene.reset()

    robot = scene["robot"]
    if include_scene_camera:
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
    bound_robot_geoms = 0
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running() and step_count < max(1, steps):
        if sim.is_stopped():
            break
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1
        if bound_robot_geoms == 0:
            bound_robot_geoms = force_bind_robot_visual_material(config)
            if bound_robot_geoms:
                print(
                    f"[openarm_scene] force-bound clean robot material to {bound_robot_geoms} geom prims",
                    file=sys.stderr,
                    flush=True,
                )

    print("[openarm_scene] reading camera", file=sys.stderr, flush=True)
    camera_shape = None
    camera_rgb_path = None
    if include_scene_camera:
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
        "scene_camera_enabled": include_scene_camera,
        "viewer_eye_m": [float(v) for v in viewer_eye],
        "viewer_target_m": [float(v) for v in viewer_target],
        "headless": headless,
        "device": device,
        "clean_robot_material_bound_geoms": int(bound_robot_geoms),
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

    include_scene_camera = not args.disable_scene_camera
    kit_args = args.kit_args
    viewer_eye_m, viewer_target_m = _viewer_camera_pose(config, args.viewer_camera)
    if args.viewer_renderer == "pxr":
        if args.save_camera_rgb:
            raise SystemExit("--viewer-renderer pxr is for GUI inspection and cannot save scene camera RGB.")
        if include_scene_camera:
            print(
                "[openarm_scene] PXR viewer selected; disabling scene camera sensor because Isaac Lab cameras require RTX.",
                file=sys.stderr,
                flush=True,
            )
            include_scene_camera = False
        kit_args = _merge_kit_args(PXR_VIEWER_KIT_ARGS, kit_args)

    result = run_isaac_scene(
        config,
        steps=args.steps,
        headless=args.headless,
        device=args.device,
        manifest_path=args.manifest,
        save_camera_rgb=args.save_camera_rgb,
        include_scene_camera=include_scene_camera,
        experience=args.experience,
        rendering_mode=args.rendering_mode,
        kit_args=kit_args,
        use_distant_light=args.viewer_renderer == "pxr",
        viewer_eye_m=viewer_eye_m,
        viewer_target_m=viewer_target_m,
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
