#!/usr/bin/env python3
"""Collect and validate the photo-clean RGB+depth staged-init height sweep.

Simulation only. This orchestration script never opens CAN, SSH, Jetson, or
real-replay paths. It writes new configs/datasets/reports under the
``openarm_photo_clean_v1_depth_stagedinit`` family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - minimal shells only
    yaml = None

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]
sys.path.insert(0, str(THIS.parent))

from collect_dense_isaac_dataset import (  # noqa: E402
    ACTION_KEY,
    DEPTH_KEY,
    JOINT_NAMES,
    load_staged_init_csv,
)
from collect_height_sweep_successes import (  # noqa: E402
    apply_height_to_config,
    height_tag,
    load_scene_config,
    parse_heights,
)
from sim_contract import SAFE_ARM_LIMITS_DEG, SAFE_GRIPPER_LIMIT_DEG  # noqa: E402


HEIGHTS_CM = (120.0, 117.5, 115.0, 112.5, 110.0, 107.5)
TASKS = ("orange_ball", "red_cube", "blue_cube")
INIT_TRAJECTORIES = {
    "initA": Path("/home/chayanin/Downloads/joint_positions_2.csv"),
    "initB": Path("/home/chayanin/Downloads/joint_positions_3.csv"),
}
TASK_WEIGHTS = {
    "orange_ball": "1,0,0,0",
    "red_cube": "0,1,0,0",
    "blue_cube": "0,0,0,1",
}
DEFAULT_BASE_CONFIG = "synthetic_smolvla/configs/scene_openarm_real_photo_left_centered_clean_v1.yaml"
DEFAULT_CONFIG_ROOT = "synthetic_smolvla/configs/generated_height_sweep_photo_clean_v1_depth_stagedinit"
DEFAULT_DATASET_ROOT = "synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit"
DEFAULT_REPORT_ROOT = "synthetic_smolvla/reports/photo_clean_v1_depth_stagedinit"

FPS = 20
COMMANDS = 400
MAX_STEP_DEG = 1.5
MAX_SPEED_DEG_S = FPS * MAX_STEP_DEG


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _stage1_reset_pose(stage1: list[float]) -> dict[str, float]:
    return {joint: float(stage1[index]) for index, joint in enumerate(JOINT_NAMES)} | {
        "gripper": float(stage1[-1])
    }


def _task_quota(task: str, count: int) -> str:
    return f"{task}={int(count)}"


def _dataset_dir(dataset_root: Path, height_cm: float, init_name: str, task: str) -> Path:
    return dataset_root / height_tag(height_cm) / init_name / task


def _report_dir(report_root: Path, height_cm: float, init_name: str, task: str) -> Path:
    return report_root / height_tag(height_cm) / init_name / task


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-root", default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--heights-cm", default=",".join(f"{height:g}" for height in HEIGHTS_CM))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--inits", default=",".join(INIT_TRAJECTORIES), help="comma-separated init names: initA,initB")
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--python", default="scripts/isaaclab_python.sh")
    parser.add_argument("--collector-script", default="synthetic_smolvla/scripts/collect_dense_isaac_dataset.py")
    parser.add_argument("--dry-run", action="store_true", help="write configs/plan only")
    parser.add_argument("--validate-only", action="store_true", help="validate existing outputs and write final_report.md")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip complete combinations and append missing episodes without deleting existing outputs",
    )
    parser.add_argument("--overwrite-output", action="store_true", help="allow deleting existing staged-depth output roots")
    return parser


def _build_config(
    base_config: dict[str, Any],
    *,
    height_cm: float,
    init_name: str,
    init_csv: Path,
    task: str,
    config_root: Path,
) -> tuple[Path, dict[str, Any]]:
    adjusted, delta_m = apply_height_to_config(base_config, height_cm=height_cm, shift_camera_with_robot=True)
    stages = load_staged_init_csv(init_csv, expected_side="left")
    scene = adjusted.setdefault("scene", {})
    scene["name"] = f"openarm_photo_clean_v1_depth_stagedinit_{height_tag(height_cm)}_{init_name}_{task}"
    scene["depth_stagedinit"] = {
        "height_cm": float(height_cm),
        "height_tag": height_tag(height_cm),
        "init_name": init_name,
        "init_csv": str(init_csv),
        "task": task,
        "stages_deg": stages,
        "fps": FPS,
        "commands": COMMANDS,
        "max_step_deg": MAX_STEP_DEG,
        "max_speed_deg_s": MAX_SPEED_DEG_S,
    }
    scene["height_sweep"]["applied_robot_root_z_delta_m"] = round(float(delta_m), 5)
    camera = scene.setdefault("camera", {})
    camera["resolution"] = [256, 256]
    camera["data_types"] = ["rgb", "distance_to_image_plane"]
    robot = adjusted.setdefault("robot", {})
    reset_pose = robot.setdefault("reset_pose_deg", {})
    reset_pose["left"] = _stage1_reset_pose(stages[0])
    cfg_path = config_root / height_tag(height_cm) / init_name / f"{task}.yaml"
    _write_yaml(cfg_path, adjusted)
    return cfg_path, adjusted


def _collector_command(
    args: argparse.Namespace,
    *,
    cfg_path: Path,
    height_cm: float,
    init_name: str,
    init_csv: Path,
    task: str,
    combo_index: int,
    dataset_root_override: Path | None = None,
    report_root_override: Path | None = None,
    target_count_override: int | None = None,
    seed_override: int | None = None,
    force_overwrite: bool | None = None,
) -> list[str]:
    dataset_root = dataset_root_override if dataset_root_override is not None else _resolve(args.dataset_root)
    report_root = report_root_override if report_root_override is not None else _resolve(args.report_root)
    ds = _dataset_dir(dataset_root, height_cm, init_name, task)
    rep = _report_dir(report_root, height_cm, init_name, task)
    target_count = int(target_count_override if target_count_override is not None else args.target_count)
    seed = int(seed_override if seed_override is not None else int(args.seed) + combo_index * 97)
    cmd = [
        *shlex.split(str(args.python)),
        str(_resolve(args.collector_script)),
        "--config",
        str(cfg_path),
        "--dataset-root",
        str(ds),
        "--repo-id",
        f"local/openarm_photo_clean_v1_depth_stagedinit_{height_tag(height_cm)}_{init_name}_{task}",
        "--manifest",
        str(rep / "manifest.jsonl"),
        "--report",
        str(rep / "collect.md"),
        "--sample-frame-dir",
        str(rep / "samples"),
        "--num-envs",
        str(args.num_envs),
        "--rounds",
        str(args.rounds),
        "--seed",
        str(seed),
        "--target-weights",
        TASK_WEIGHTS[task],
        "--target-quotas",
        _task_quota(task, target_count),
        "--max-keep",
        str(target_count),
        "--fps",
        str(FPS),
        "--substeps",
        "10",
        "--max-action-step-deg",
        str(MAX_STEP_DEG),
        "--staged-init-csv",
        str(init_csv),
        "--staged-init-name",
        init_name,
        "--target-episode-commands",
        str(COMMANDS),
        "--approach-steps",
        "40",
        "--descend-steps",
        "40",
        "--close-steps",
        "32",
        "--lift-steps",
        "48",
        "--hold-steps",
        "0",
        "--above-offset-m",
        "0.12",
        "--lift-offset-m",
        "0.08",
        "--lift-threshold-m",
        "0.05",
        "--grasp-z-offset-m",
        "0.01",
        "--grasp-close-deg",
        "-13.0",
        "--open-gripper-deg",
        "-50.0",
        "--max-gripper-close-deg",
        "-13.0",
        "--gripper-close-range-deg",
        "-17",
        "-13",
        "--action-clip-tol-deg",
        "180.0",
        "--early-stop-on-lift",
        "--jitter-x-m",
        "0.005",
        "--jitter-y-m",
        "0.003",
        "--dataset-backend",
        "local_npz",
        "--require-depth",
    ]
    overwrite = bool(args.overwrite_output) if force_overwrite is None else bool(force_overwrite)
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def _manifest_kept(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise ValueError(f"missing manifest: {manifest_path}")
    kept = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("kept"):
            kept.append(item)
    return kept


def _episode_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "episodes").glob("episode_*.npz"))


def _combo_progress(*, dataset_root: Path, manifest_path: Path, target_count: int) -> dict[str, Any]:
    episodes = len(_episode_files(dataset_root))
    kept = len(_manifest_kept(manifest_path)) if manifest_path.exists() else 0
    if episodes != kept:
        return {
            "ok": False,
            "status": "inconsistent",
            "episodes": episodes,
            "kept": kept,
            "missing": max(0, int(target_count) - min(episodes, kept)),
            "reason": f"episode count {episodes} != kept manifest count {kept}",
        }
    if episodes > int(target_count):
        return {
            "ok": False,
            "status": "inconsistent",
            "episodes": episodes,
            "kept": kept,
            "missing": 0,
            "reason": f"episode count {episodes} exceeds target {target_count}",
        }
    if episodes == int(target_count):
        status = "complete"
    elif episodes == 0:
        status = "empty"
    else:
        status = "partial"
    return {
        "ok": True,
        "status": status,
        "episodes": episodes,
        "kept": kept,
        "missing": max(0, int(target_count) - episodes),
        "reason": "",
    }


def _rewrite_local_npz_meta(dataset_root: Path) -> None:
    meta_path = dataset_root / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    episode_count = len(_episode_files(dataset_root))
    meta.update(
        {
            "ok": True,
            "backend": "local_npz",
            "num_episodes": episode_count,
            "num_frames": episode_count * COMMANDS,
            "depth_key": DEPTH_KEY,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_resume_output(
    *,
    source_dataset_root: Path,
    source_manifest_path: Path,
    destination_dataset_root: Path,
    destination_manifest_path: Path,
) -> dict[str, int]:
    source_eps = _episode_files(source_dataset_root)
    if not source_eps:
        raise ValueError(f"resume collection produced no episodes: {source_dataset_root}")
    if not source_manifest_path.exists():
        raise ValueError(f"resume collection produced no manifest: {source_manifest_path}")

    destination_episodes_dir = destination_dataset_root / "episodes"
    destination_episodes_dir.mkdir(parents=True, exist_ok=True)
    start_index = len(_episode_files(destination_dataset_root))
    for offset, episode_path in enumerate(source_eps):
        out = destination_episodes_dir / f"episode_{start_index + offset:06d}.npz"
        if out.exists():
            raise ValueError(f"refusing to overwrite existing episode while resuming: {out}")
        shutil.copy2(episode_path, out)

    destination_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_text = source_manifest_path.read_text(encoding="utf-8")
    with destination_manifest_path.open("a", encoding="utf-8") as handle:
        if destination_manifest_path.stat().st_size > 0 and source_text:
            existing_text = destination_manifest_path.read_text(encoding="utf-8")
            if not existing_text.endswith("\n"):
                handle.write("\n")
        handle.write(source_text)
        if source_text and not source_text.endswith("\n"):
            handle.write("\n")

    _rewrite_local_npz_meta(destination_dataset_root)
    return {"start_index": start_index, "appended": len(source_eps)}


def _inside_left_limits(action: "Any") -> bool:
    for index, joint in enumerate(JOINT_NAMES):
        low, high = SAFE_ARM_LIMITS_DEG["left"][joint]
        values = action[:, index]
        if bool(((values < low - 1.0e-5) | (values > high + 1.0e-5)).any()):
            return False
    grip_low, grip_high = SAFE_GRIPPER_LIMIT_DEG
    gripper = action[:, -1]
    return not bool(((gripper < grip_low - 1.0e-5) | (gripper > grip_high + 1.0e-5)).any())


def validate_one(
    *,
    dataset_root: Path,
    manifest_path: Path,
    init_csv: Path,
    init_name: str,
    task: str,
    target_count: int,
) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    stages = load_staged_init_csv(init_csv, expected_side="left")
    eps = sorted((dataset_root / "episodes").glob("episode_*.npz"))
    kept = _manifest_kept(manifest_path)
    failures: list[str] = []
    if len(eps) != int(target_count):
        failures.append(f"episode count {len(eps)} != {target_count}")
    if len(kept) != int(target_count):
        failures.append(f"kept manifest count {len(kept)} != {target_count}")
    max_step = 0.0
    rgb_shape = None
    depth_shape = None
    for item in kept:
        for flag in (
            "wrong_object_lifted",
            "object_collision",
            "gripper_table_collision",
            "object_swept_or_slid",
            "tabletop_penetration",
            "object_pushed_down",
            "refined_action_clip",
            "action_slew_violation",
            "gripper_cap_violation",
        ):
            if item.get(flag):
                failures.append(f"kept episode {item.get('episode_index')} has {flag}=true")
        if item.get("target_object") != task:
            failures.append(f"kept episode {item.get('episode_index')} target {item.get('target_object')} != {task}")
        if not item.get("success_label"):
            failures.append(f"kept episode {item.get('episode_index')} success_label=false")
    for ep in eps:
        data = np.load(ep, allow_pickle=True)
        if "action" not in data.files:
            failures.append(f"{ep.name}: missing action")
            continue
        if "camera" not in data.files:
            failures.append(f"{ep.name}: missing camera")
        if DEPTH_KEY not in data.files:
            failures.append(f"{ep.name}: missing {DEPTH_KEY}")
            continue
        action = np.asarray(data["action"], dtype=np.float32)
        if action.shape != (COMMANDS, 8):
            failures.append(f"{ep.name}: action shape {action.shape} != {(COMMANDS, 8)}")
        if not np.isfinite(action).all():
            failures.append(f"{ep.name}: action contains non-finite values")
        if action.shape[0] > 1:
            max_step = max(max_step, float(np.abs(np.diff(action, axis=0)).max()))
        if action.shape == (COMMANDS, 8) and float(np.max(np.abs(action[0] - np.asarray(stages[0])))) > 0.05:
            failures.append(f"{ep.name}: first action is not within 0.05 deg of {init_name} stage 1")
        for stage_index, stage in enumerate(stages[1:], start=2):
            if action.shape == (COMMANDS, 8):
                nearest = float(np.min(np.max(np.abs(action - np.asarray(stage, dtype=np.float32)), axis=1)))
                if nearest > 0.05:
                    failures.append(f"{ep.name}: path does not pass {init_name} stage {stage_index} within 0.05 deg")
        if not _inside_left_limits(action):
            failures.append(f"{ep.name}: action outside left-arm safe sim limits")
        rgb = np.asarray(data["camera"])
        depth = np.asarray(data[DEPTH_KEY], dtype=np.float32)
        rgb_shape = list(rgb.shape)
        depth_shape = list(depth.shape)
        if rgb.shape[0] != COMMANDS:
            failures.append(f"{ep.name}: RGB T {rgb.shape[0]} != {COMMANDS}")
        if depth.shape[0] != COMMANDS:
            failures.append(f"{ep.name}: depth T {depth.shape[0]} != {COMMANDS}")
        if depth.ndim not in (3, 4):
            failures.append(f"{ep.name}: depth shape must be [T,H,W] or [T,H,W,1], got {depth.shape}")
        if depth.size == 0:
            failures.append(f"{ep.name}: depth is empty")
        if not np.isfinite(depth).all():
            failures.append(f"{ep.name}: depth contains non-finite values")
        if depth.size and float(np.nanmax(depth) - np.nanmin(depth)) <= 1.0e-6:
            failures.append(f"{ep.name}: depth is constant")
        if rgb.shape[0] > 1 and not bool(np.any(rgb[1:] != rgb[0])):
            failures.append(f"{ep.name}: RGB is static")
    ok = not failures and max_step <= MAX_STEP_DEG + 1.0e-5
    if max_step > MAX_STEP_DEG + 1.0e-5:
        failures.append(f"max action step {max_step:.6f} > {MAX_STEP_DEG}")
        ok = False
    return {
        "ok": ok,
        "output_path": str(dataset_root),
        "episodes": len(eps),
        "fps": FPS,
        "command_length": COMMANDS,
        "target_object": task,
        "init_trajectory": init_name,
        "max_command_delta_deg": round(float(max_step), 6),
        "max_speed_deg_s": round(float(max_step) * FPS, 6),
        "rgb_shape": rgb_shape,
        "depth_shape": depth_shape,
        "depth_key": DEPTH_KEY,
        "expected_duration_sec": COMMANDS / FPS,
        "failures": failures,
    }


def write_final_report(report_root: Path, results: list[dict[str, Any]]) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    out = report_root / "final_report.md"
    lines = [
        "# Photo-Clean Depth Staged-Init Final Report",
        "",
        f"- Total combinations: {len(results)}",
        f"- Passing combinations: {sum(1 for item in results if item['ok'])}",
        f"- FPS: {FPS}",
        f"- Commands per episode: {COMMANDS}",
        f"- Expected duration: {COMMANDS / FPS:.1f} sec",
        f"- Max command delta: {MAX_STEP_DEG:.3f} deg",
        f"- Max speed: {MAX_SPEED_DEG_S:.3f} deg/sec",
        f"- Depth key: `{DEPTH_KEY}`",
        "",
        "| Height/Init/Task | Episodes | Max Step | Max Speed | RGB Shape | Depth Shape | Result |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for item in results:
        rel = Path(item["output_path"]).relative_to(REPO_ROOT)
        label = "/".join(rel.parts[-3:])
        result = "PASS" if item["ok"] else "FAIL: " + "; ".join(item["failures"][:3])
        lines.append(
            f"| `{label}` | {item['episodes']} | {item['max_command_delta_deg']:.6f} | "
            f"{item['max_speed_deg_s']:.6f} | `{item['rgb_shape']}` | "
            f"`{item['depth_shape']}` | {result} |"
        )
    lines.append("")
    lines.append("## Output Paths")
    lines.append("")
    for item in results:
        lines.append(
            f"- `{item['output_path']}`: "
            f"{'PASS' if item['ok'] else 'FAIL'}; target={item['target_object']}; "
            f"init={item['init_trajectory']}; episodes={item['episodes']}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_logged_command(cmd: list[str], *, cwd: Path, console_log: Path) -> None:
    """Run a collector command, saving full output while streaming compact progress."""
    with console_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            if (
                line.startswith("[dense] round")
                or line.startswith("[dense] target quotas")
                or line.startswith("[dense] reached --max-keep")
                or "episode_len=" in line
            ):
                print(line.rstrip(), flush=True)
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def main() -> int:
    args = build_arg_parser().parse_args()
    heights = parse_heights(args.heights_cm)
    tasks = [part.strip() for part in str(args.tasks).split(",") if part.strip()]
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_tasks:
        raise SystemExit(f"unknown tasks {unknown_tasks}; expected subset of {TASKS}")
    init_names = [part.strip() for part in str(args.inits).split(",") if part.strip()]
    unknown_inits = sorted(set(init_names) - set(INIT_TRAJECTORIES))
    if unknown_inits:
        raise SystemExit(f"unknown inits {unknown_inits}; expected subset of {tuple(INIT_TRAJECTORIES)}")
    if args.target_count <= 0:
        raise SystemExit("--target-count must be positive")
    if args.resume and args.overwrite_output:
        raise SystemExit("--resume cannot be combined with --overwrite-output")

    base_config = load_scene_config(args.base_config)
    config_root = _resolve(args.config_root)
    dataset_root = _resolve(args.dataset_root)
    report_root = _resolve(args.report_root)
    resume_temp_dataset_root = dataset_root.with_name(dataset_root.name + "_resume_tmp")
    resume_temp_report_root = report_root.with_name(report_root.name + "_resume_tmp")
    report_root.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    combo_index = 0
    selected_init_trajectories = {name: INIT_TRAJECTORIES[name] for name in init_names}
    for height_cm in heights:
        for init_name, init_csv in selected_init_trajectories.items():
            for task in tasks:
                cfg_path, _ = _build_config(
                    base_config,
                    height_cm=height_cm,
                    init_name=init_name,
                    init_csv=init_csv,
                    task=task,
                    config_root=config_root,
                )
                cmd = _collector_command(
                    args,
                    cfg_path=cfg_path,
                    height_cm=height_cm,
                    init_name=init_name,
                    init_csv=init_csv,
                    task=task,
                    combo_index=combo_index,
                )
                ds = _dataset_dir(dataset_root, height_cm, init_name, task)
                rep = _report_dir(report_root, height_cm, init_name, task)
                progress = _combo_progress(
                    dataset_root=ds,
                    manifest_path=rep / "manifest.jsonl",
                    target_count=args.target_count,
                )
                generated.append({
                    "combo_index": combo_index,
                    "height_cm": height_cm,
                    "height_tag": height_tag(height_cm),
                    "init": init_name,
                    "task": task,
                    "config": str(cfg_path),
                    "dataset_root": str(ds),
                    "report_dir": str(rep),
                    "manifest": str(rep / "manifest.jsonl"),
                    "progress": progress,
                    "command": cmd,
                    "command_shell": shlex.join(cmd),
                })
                commands.append(cmd)
                combo_index += 1

    plan = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "validate_only": bool(args.validate_only),
        "resume": bool(args.resume),
        "base_config": str(args.base_config),
        "config_root": str(config_root),
        "dataset_root": str(dataset_root),
        "report_root": str(report_root),
        "resume_temp_dataset_root": str(resume_temp_dataset_root),
        "resume_temp_report_root": str(resume_temp_report_root),
        "heights_cm": heights,
        "init_trajectories": {name: str(path) for name, path in selected_init_trajectories.items()},
        "tasks": tasks,
        "target_count": int(args.target_count),
        "contract": {
            "fps": FPS,
            "commands": COMMANDS,
            "expected_duration_sec": COMMANDS / FPS,
            "max_step_deg": MAX_STEP_DEG,
            "max_speed_deg_s": MAX_SPEED_DEG_S,
        },
        "generated": generated,
    }
    plan_path = report_root / "run_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_script = report_root / "run_depth_stagedinit_collect.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(REPO_ROOT))
        + "\n\n"
        + "\n\n".join(shlex.join(cmd) for cmd in commands)
        + "\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "plan": str(plan_path), "run_script": str(run_script)}, indent=2))
        return 0

    if not args.validate_only:
        for item, cmd in zip(generated, commands, strict=True):
            rep = Path(item["report_dir"])
            rep.mkdir(parents=True, exist_ok=True)
            console = rep / "collect_console.log"
            if args.resume:
                progress = _combo_progress(
                    dataset_root=Path(item["dataset_root"]),
                    manifest_path=Path(item["manifest"]),
                    target_count=args.target_count,
                )
                if not progress["ok"]:
                    raise SystemExit(
                        f"cannot resume {item['height_tag']} {item['init']} {item['task']}: "
                        f"{progress['reason']}"
                    )
                if progress["status"] == "complete":
                    print(
                        f"[depth-stagedinit] skip complete {item['height_tag']} "
                        f"{item['init']} {item['task']} ({progress['episodes']}/{args.target_count})",
                        flush=True,
                    )
                    continue
                if progress["status"] == "partial":
                    missing = int(progress["missing"])
                    temp_ds = _dataset_dir(
                        resume_temp_dataset_root,
                        float(item["height_cm"]),
                        str(item["init"]),
                        str(item["task"]),
                    )
                    temp_rep = _report_dir(
                        resume_temp_report_root,
                        float(item["height_cm"]),
                        str(item["init"]),
                        str(item["task"]),
                    )
                    resume_seed = int(args.seed) + int(item["combo_index"]) * 97 + 100_000 + int(progress["episodes"]) * 1009
                    temp_cmd = _collector_command(
                        args,
                        cfg_path=Path(item["config"]),
                        height_cm=float(item["height_cm"]),
                        init_name=str(item["init"]),
                        init_csv=INIT_TRAJECTORIES[str(item["init"])],
                        task=str(item["task"]),
                        combo_index=int(item["combo_index"]),
                        dataset_root_override=resume_temp_dataset_root,
                        report_root_override=resume_temp_report_root,
                        target_count_override=missing,
                        seed_override=resume_seed,
                        force_overwrite=True,
                    )
                    console = rep / f"resume_collect_console_{progress['episodes']:03d}_to_{args.target_count:03d}.log"
                    print(
                        f"[depth-stagedinit] resume {item['height_tag']} {item['init']} {item['task']}: "
                        f"{progress['episodes']}/{args.target_count}, collecting {missing} missing -> {item['dataset_root']}",
                        flush=True,
                    )
                    run_logged_command(temp_cmd, cwd=REPO_ROOT, console_log=console)
                    append_info = _append_resume_output(
                        source_dataset_root=temp_ds,
                        source_manifest_path=temp_rep / "manifest.jsonl",
                        destination_dataset_root=Path(item["dataset_root"]),
                        destination_manifest_path=Path(item["manifest"]),
                    )
                    append_info.update(
                        {
                            "missing_requested": missing,
                            "resume_seed": resume_seed,
                            "temp_dataset_root": str(temp_ds),
                            "temp_report_dir": str(temp_rep),
                        }
                    )
                    (rep / "resume_append.json").write_text(
                        json.dumps(append_info, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    shutil.rmtree(temp_ds, ignore_errors=True)
                    continue
                if Path(item["dataset_root"]).exists() and any(Path(item["dataset_root"]).iterdir()):
                    raise SystemExit(
                        f"cannot resume empty {item['height_tag']} {item['init']} {item['task']}: "
                        f"dataset directory is non-empty but has no kept episodes: {item['dataset_root']}"
                    )
            print(
                f"[depth-stagedinit] collecting {item['height_tag']} {item['init']} {item['task']} -> {item['dataset_root']}",
                flush=True,
            )
            run_logged_command(cmd, cwd=REPO_ROOT, console_log=console)

    results = []
    for item in generated:
        init_csv = INIT_TRAJECTORIES[item["init"]]
        results.append(
            validate_one(
                dataset_root=Path(item["dataset_root"]),
                manifest_path=Path(item["manifest"]),
                init_csv=init_csv,
                init_name=item["init"],
                task=item["task"],
                target_count=args.target_count,
            )
        )
    report = write_final_report(report_root, results)
    ok = all(item["ok"] for item in results)
    print(json.dumps({"ok": ok, "final_report": str(report), "results": results}, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
