#!/usr/bin/env python3
"""Collect clean successful dense episodes across requested robot heights.

This is a thin orchestration layer over ``collect_dense_isaac_dataset.py``.  It
creates one generated scene config per robot height, shifts the robot root by
the requested height delta, then runs the dense collector until each height has
the requested clean success quota.

Simulation only. This script does not touch CAN, SSH, mirror, or real-replay
code. Run it with the Isaac Lab Python helper when actually collecting.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal shells
    yaml = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_contract import CONFIG_DIR, REPO_ROOT, load_yaml_config as _load_yaml_config  # noqa: E402


DEFAULT_HEIGHTS_CM = (125.0, 122.5, 120.0, 117.5, 115.0)
DEFAULT_TARGET_QUOTAS = "3,3,2,2"
DEFAULT_EPISODE_COMMANDS = 50


def height_tag(height_cm: float) -> str:
    text = f"{float(height_cm):g}".replace(".", "p").replace("-", "m")
    return f"h{text}cm"


def parse_heights(raw: str) -> list[float]:
    heights = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not heights:
        raise ValueError("at least one height is required")
    return heights


def parse_quota_counts(raw: str, object_names: list[str]) -> dict[str, int]:
    text = str(raw).strip()
    if not text:
        raise ValueError("--target-quotas cannot be empty")
    if "=" not in text:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
        if len(values) != len(object_names):
            raise ValueError(f"--target-quotas needs {len(object_names)} values aligned to {object_names}")
        return {name: max(0, value) for name, value in zip(object_names, values)}
    quotas = {name: 0 for name in object_names}
    valid = set(object_names)
    for part in text.split(","):
        if not part.strip():
            continue
        name, value = [piece.strip() for piece in part.split("=", 1)]
        if name not in valid:
            raise ValueError(f"unknown target quota {name!r}; expected one of {object_names}")
        quotas[name] = max(0, int(value))
    return quotas


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "None"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _simple_yaml_load(path: Path) -> dict[str, Any]:
    """Tiny fallback for the repo's simple config YAML when PyYAML is absent."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: tuple[int, dict[str, Any], str] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if pending_key is not None and pending_key[0] < indent:
            parent_indent, parent, key = pending_key
            value: list[Any] | dict[str, Any] = [] if text.startswith("- ") else {}
            parent[key] = value
            stack.append((parent_indent + 1, value))
            pending_key = None
        parent = stack[-1][1]
        if text.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"expected list parent while parsing {path}: {raw!r}")
            item_text = text[2:].strip()
            if ":" in item_text:
                key, value_text = [part.strip() for part in item_text.split(":", 1)]
                item: dict[str, Any] = {}
                parent.append(item)
                stack.append((indent, item))
                if value_text:
                    item[key] = _parse_scalar(value_text)
                else:
                    pending_key = (indent, item, key)
            else:
                parent.append(_parse_scalar(item_text))
            continue
        if ":" not in text:
            raise ValueError(f"cannot parse line in {path}: {raw!r}")
        key, value_text = [part.strip() for part in text.split(":", 1)]
        if not isinstance(parent, dict):
            raise ValueError(f"expected mapping parent while parsing {path}: {raw!r}")
        if value_text:
            parent[key] = _parse_scalar(value_text)
            pending_key = None
        else:
            pending_key = (indent, parent, key)
    return root


def load_scene_config(path: str | Path) -> dict[str, Any]:
    if yaml is not None:
        return _load_yaml_config(path)
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return _simple_yaml_load(resolved)


def _height_baseline_cm(config: dict[str, Any], requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    layout = config.get("scene", {}).get("layout_info_cm", {})
    if "robot_plus_table_height" not in layout:
        raise ValueError("base config needs scene.layout_info_cm.robot_plus_table_height or --baseline-height-cm")
    return float(layout["robot_plus_table_height"])


def apply_height_to_config(
    config: dict[str, Any],
    *,
    height_cm: float,
    baseline_height_cm: float | None = None,
    shift_camera_with_robot: bool = True,
) -> tuple[dict[str, Any], float]:
    """Return a config copy adjusted for ``height_cm``.

    The existing real-table reachable configs encode the user's 125 cm robot
    height as ``layout_info_cm.robot_plus_table_height`` and use
    ``robot.base_pose_m[2]`` as the simulated OpenArm root height.  A new height
    therefore applies the relative delta to the root z. The table and objects
    stay fixed.
    """
    out = copy.deepcopy(config)
    base_height = _height_baseline_cm(out, baseline_height_cm)
    delta_m = (float(height_cm) - base_height) / 100.0

    scene = out.setdefault("scene", {})
    layout = scene.setdefault("layout_info_cm", {})
    layout["robot_plus_table_height"] = float(height_cm)
    scene["name"] = f"{scene.get('name', 'openarm_scene')}_{height_tag(height_cm)}"
    scene["height_sweep"] = {
        "height_cm": float(height_cm),
        "baseline_height_cm": float(base_height),
        "applied_robot_root_z_delta_m": round(delta_m, 5),
        "camera_shifted_with_robot": bool(shift_camera_with_robot),
    }

    robot = out.setdefault("robot", {})
    base_pose = robot.get("base_pose_m")
    if not isinstance(base_pose, list) or len(base_pose) != 3:
        raise ValueError("base config needs robot.base_pose_m: [x, y, z] for height sweeps")
    base_pose = [float(v) for v in base_pose]
    base_pose[2] = round(base_pose[2] + delta_m, 5)
    robot["base_pose_m"] = base_pose

    if shift_camera_with_robot:
        camera = scene.get("camera", {})
        eye = camera.get("eye_m")
        if isinstance(eye, list) and len(eye) == 3:
            eye = [float(v) for v in eye]
            eye[2] = round(eye[2] + delta_m, 5)
            camera["eye_m"] = eye

    return out, delta_m


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-config",
        default=str(CONFIG_DIR / "scene_openarm_real_table_zero_train_reachable_left_v1.yaml"),
        help="approved scene config to mutate for each height",
    )
    p.add_argument(
        "--heights-cm",
        default=",".join(f"{h:g}" for h in DEFAULT_HEIGHTS_CM),
        help="comma-separated robot heights to collect, in cm",
    )
    p.add_argument("--baseline-height-cm", type=float, default=None)
    p.add_argument("--height-config-dir", default="synthetic_smolvla/configs/generated_height_sweep")
    p.add_argument("--dataset-prefix", default="openarm_height_sweep_lift5cm_10hz_50step")
    p.add_argument("--report-dir", default="synthetic_smolvla/reports/height_sweep_lift5cm_10hz_50step")
    p.add_argument("--collector-script", default="synthetic_smolvla/scripts/collect_dense_isaac_dataset.py")
    p.add_argument("--merge-script", default="synthetic_smolvla/scripts/merge_lerobot_datasets.py")
    p.add_argument("--combined-dataset-root", default="synthetic_smolvla/datasets/openarm_height_sweep_lift5cm_10hz_50eps_50step")
    p.add_argument("--combined-repo-id", default="local/openarm_height_sweep_lift5cm_10hz_50eps_50step")
    p.add_argument("--merge-report", default="synthetic_smolvla/reports/height_sweep_lift5cm_10hz_50step/merged_dataset.json")
    p.add_argument("--python", default=sys.executable, help="Python/Isaac helper used to run the collector")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rounds-per-height", type=int, default=80)
    p.add_argument("--seed", type=int, default=26000)
    p.add_argument("--target-quotas", default=DEFAULT_TARGET_QUOTAS, help="per-height retained successes aligned to config objects")
    p.add_argument("--target-weights", default="1,1,1,1")
    p.add_argument("--episode-commands", type=int, default=DEFAULT_EPISODE_COMMANDS)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--substeps", type=int, default=20, help="20 physics substeps at dt=0.005 gives 10 Hz commands")
    p.add_argument(
        "--camera-mode",
        choices=("isaac", "placeholder"),
        default="isaac",
        help="pass-through camera mode for the dense collector",
    )
    p.add_argument(
        "--placeholder-view",
        choices=("isaac_viewport", "robot_front"),
        default="isaac_viewport",
        help="pass-through placeholder visual style when --camera-mode placeholder",
    )
    p.add_argument("--experience", default="", help="optional Isaac/Kit experience file passed to the dense collector")
    p.add_argument("--rendering-mode", default=None, choices=("performance", "balanced", "quality"))
    p.add_argument("--kit-args", default="", help="raw Omniverse Kit args passed to the dense collector")
    p.add_argument("--max-motion-deg-per-s", type=float, default=10.0)
    p.add_argument("--approach-steps", type=int, default=14)
    p.add_argument("--descend-steps", type=int, default=12)
    p.add_argument("--close-steps", type=int, default=8)
    p.add_argument("--lift-steps", type=int, default=12)
    p.add_argument("--hold-steps", type=int, default=4)
    p.add_argument("--above-offset-m", type=float, default=0.12)
    p.add_argument("--lift-offset-m", type=float, default=0.05)
    p.add_argument("--lift-threshold-m", type=float, default=0.05)
    p.add_argument("--gripper-close-deg", type=float, default=-10.0)
    p.add_argument(
        "--open-gripper-deg",
        type=float,
        default=-34.0,
        help="episode initial/open gripper command; -34 can reach -10 within 50 commands at 1 deg/command",
    )
    p.add_argument("--jitter-x-m", type=float, default=0.005)
    p.add_argument("--jitter-y-m", type=float, default=0.003)
    p.add_argument("--prepose-warmup-steps", type=int, default=120)
    p.add_argument("--no-prepose-to-ready", action="store_true")
    p.add_argument("--no-camera-height-shift", action="store_true")
    p.add_argument("--no-merge", action="store_true", help="skip merging the five per-height datasets")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="write configs and command files without running Isaac")
    return p


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _dataset_root(args: argparse.Namespace, height_cm: float) -> Path:
    return REPO_ROOT / "synthetic_smolvla" / "datasets" / f"{args.dataset_prefix}_{height_tag(height_cm)}"


def _collector_command(args: argparse.Namespace, *, height_cm: float, height_config: Path, index: int) -> list[str]:
    tag = height_tag(height_cm)
    dataset_name = f"{args.dataset_prefix}_{tag}"
    report_dir = _resolve(args.report_dir)
    step_cap = float(args.max_motion_deg_per_s) / float(args.fps)
    cmd = [
        *shlex.split(args.python),
        str(_resolve(args.collector_script)),
        "--config", str(height_config),
        "--dataset-root", str(_dataset_root(args, height_cm)),
        "--repo-id", f"local/{dataset_name}",
        "--manifest", str(report_dir / f"{dataset_name}_manifest.jsonl"),
        "--report", str(report_dir / f"{dataset_name}_collect.md"),
        "--sample-frame-dir", str(report_dir / f"{dataset_name}_samples"),
        "--num-envs", str(args.num_envs),
        "--rounds", str(args.rounds_per_height),
        "--seed", str(args.seed + index * 1000),
        "--target-weights", str(args.target_weights),
        "--target-quotas", str(args.target_quotas),
        "--fps", str(args.fps),
        "--camera-mode", str(args.camera_mode),
        "--placeholder-view", str(args.placeholder_view),
        "--substeps", str(args.substeps),
        "--approach-steps", str(args.approach_steps),
        "--descend-steps", str(args.descend_steps),
        "--close-steps", str(args.close_steps),
        "--lift-steps", str(args.lift_steps),
        "--hold-steps", str(args.hold_steps),
        "--above-offset-m", str(args.above_offset_m),
        "--lift-offset-m", str(args.lift_offset_m),
        "--lift-threshold-m", str(args.lift_threshold_m),
        "--grasp-close-deg", str(args.gripper_close_deg),
        "--open-gripper-deg", str(args.open_gripper_deg),
        "--max-gripper-close-deg", str(args.gripper_close_deg),
        "--max-action-step-deg", f"{step_cap:.6f}",
        "--early-stop-on-lift",
        "--jitter-x-m", str(args.jitter_x_m),
        "--jitter-y-m", str(args.jitter_y_m),
    ]
    if args.experience:
        cmd.extend(["--experience", str(args.experience)])
    if args.rendering_mode:
        cmd.extend(["--rendering-mode", str(args.rendering_mode)])
    if args.kit_args:
        cmd.extend(["--kit-args", str(args.kit_args)])
    if not args.no_prepose_to_ready:
        cmd.extend(["--prepose-to-ready", "--prepose-warmup-steps", str(args.prepose_warmup_steps)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _merge_command(args: argparse.Namespace, heights: list[float], quota_total: int) -> list[str]:
    cmd = [
        *shlex.split(args.python),
        str(_resolve(args.merge_script)),
        "--output-root", str(_resolve(args.combined_dataset_root)),
        "--repo-id", str(args.combined_repo_id),
        "--fps", str(args.fps),
        "--max-total-episodes", str(quota_total * len(heights)),
        "--report", str(_resolve(args.merge_report)),
    ]
    for height_cm in heights:
        cmd.extend(["--input", str(_dataset_root(args, height_cm))])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def main() -> int:
    args = build_arg_parser().parse_args()
    heights = parse_heights(args.heights_cm)
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    phase_steps = {
        "approach": int(args.approach_steps),
        "descend": int(args.descend_steps),
        "close": int(args.close_steps),
        "lift": int(args.lift_steps),
        "hold": int(args.hold_steps),
    }
    if any(value < 0 for value in phase_steps.values()):
        raise SystemExit("phase steps must be non-negative.")
    phase_total = sum(phase_steps.values())
    if phase_total != int(args.episode_commands):
        raise SystemExit(f"phase steps must sum to exactly {args.episode_commands} commands per episode.")

    base_config = load_scene_config(args.base_config)
    obj_names = [str(obj["name"]) for obj in base_config["objects"]]
    try:
        quota_counts = parse_quota_counts(args.target_quotas, obj_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    quota_total = sum(quota_counts.values())
    if quota_total <= 0:
        raise SystemExit("--target-quotas must request at least one kept episode per height.")

    height_config_dir = _resolve(args.height_config_dir)
    report_dir = _resolve(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    generated: list[dict[str, Any]] = []
    for index, height_cm in enumerate(heights):
        adjusted, delta_m = apply_height_to_config(
            base_config,
            height_cm=height_cm,
            baseline_height_cm=args.baseline_height_cm,
            shift_camera_with_robot=not args.no_camera_height_shift,
        )
        cfg_path = height_config_dir / f"{Path(args.base_config).stem}_{height_tag(height_cm)}.yaml"
        _write_yaml(cfg_path, adjusted)
        cmd = _collector_command(args, height_cm=height_cm, height_config=cfg_path, index=index)
        commands.append(cmd)
        generated.append({
            "height_cm": height_cm,
            "height_tag": height_tag(height_cm),
            "config": str(cfg_path),
            "robot_root_z_delta_m": delta_m,
            "command": cmd,
        })

    command_file = report_dir / "run_height_sweep_collect.sh"
    merge_cmd = None if args.no_merge else _merge_command(args, heights, quota_total)
    command_blocks = [shlex.join(cmd) for cmd in commands]
    if merge_cmd is not None:
        command_blocks.append(shlex.join(merge_cmd))
    command_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO_ROOT)) + "\n\n"
        + "\n\n".join(command_blocks)
        + "\n",
        encoding="utf-8",
    )
    command_file.chmod(0o755)
    summary_path = report_dir / "height_sweep_plan.json"
    summary = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "heights_cm": heights,
        "successes_per_height": int(quota_total),
        "total_successes": int(quota_total * len(heights)),
        "target_quotas": quota_counts,
        "commands_per_episode": int(args.episode_commands),
        "phase_steps": phase_steps,
        "fps": int(args.fps),
        "camera_mode": str(args.camera_mode),
        "placeholder_view": str(args.placeholder_view),
        "experience": str(args.experience),
        "rendering_mode": args.rendering_mode,
        "kit_args": str(args.kit_args),
        "substeps": int(args.substeps),
        "max_motion_deg_per_s": float(args.max_motion_deg_per_s),
        "max_action_step_deg": float(args.max_motion_deg_per_s) / float(args.fps),
        "max_motion_deg_per_command": float(args.max_motion_deg_per_s) / float(args.fps),
        "gripper_close_cap_deg": float(args.gripper_close_deg),
        "open_gripper_deg": float(args.open_gripper_deg),
        "lift_offset_m": float(args.lift_offset_m),
        "lift_threshold_m": float(args.lift_threshold_m),
        "early_stop_on_lift": True,
        "combined_dataset_root": str(_resolve(args.combined_dataset_root)),
        "combined_repo_id": str(args.combined_repo_id),
        "merge_report": None if args.no_merge else str(_resolve(args.merge_report)),
        "generated": generated,
        "merge_command": merge_cmd,
        "command_file": str(command_file),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "summary": str(summary_path), "command_file": str(command_file)}, indent=2))
        return 0

    for item, cmd in zip(generated, commands):
        print(f"[height-sweep] collecting {item['height_tag']} ({item['height_cm']} cm)", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if merge_cmd is not None:
        print("[height-sweep] merging per-height datasets", flush=True)
        subprocess.run(merge_cmd, cwd=REPO_ROOT, check=True)

    print(json.dumps({"ok": True, "summary": str(summary_path), "command_file": str(command_file)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
