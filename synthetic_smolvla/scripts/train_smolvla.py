#!/usr/bin/env python3
"""Prepare or run SmolVLA training for the synthetic OpenArm dataset."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from sim_contract import CONFIG_DIR, REPO_ROOT, load_yaml_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default=str(CONFIG_DIR / "train_v1.yaml"), help="training YAML to inspect")
    parser.add_argument("--output", default="synthetic_smolvla/reports/train_preflight.json")
    parser.add_argument("--command-output", default=None, help="optional shell script path for the training command")
    parser.add_argument("--run", action="store_true", help="actually launch LeRobot training")
    parser.add_argument("--steps", type=int, default=None, help="override training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size")
    parser.add_argument("--checkpoint-dir", default=None, help="override checkpoint output directory")
    parser.add_argument("--policy-path", default=None, help="optional pretrained policy path or HF id")
    parser.add_argument("--overwrite-output-dir", action="store_true", help="allow deleting .gitkeep-only checkpoint dir")
    return parser


def module_status(name: str) -> dict:
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    if spec is None:
        return {"available": False}
    return {"available": True, "origin": spec.origin}


def import_status(name: str) -> dict:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "origin": getattr(module, "__file__", None)}


def dataset_ready(dataset_root: Path) -> dict:
    required = [
        dataset_root / "meta" / "info.json",
        dataset_root / "meta" / "tasks.parquet",
        dataset_root / "data",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return {
        "root": str(dataset_root),
        "exists": dataset_root.exists(),
        "missing": missing,
        "ready": not missing,
    }


def repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def prepare_checkpoint_dir(path: Path, *, overwrite_gitkeep_only: bool) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "usable_for_new_run": True}
    entries = sorted(p.name for p in path.iterdir())
    if not entries and overwrite_gitkeep_only:
        path.rmdir()
        return {"path": str(path), "exists": True, "removed_empty_dir": True, "usable_for_new_run": True}
    if entries == [".gitkeep"] and overwrite_gitkeep_only:
        shutil.rmtree(path)
        return {"path": str(path), "exists": True, "removed_gitkeep_dir": True, "usable_for_new_run": True}
    return {"path": str(path), "exists": True, "entries": entries[:20], "usable_for_new_run": False}


def build_train_command(config: dict, *, steps: int | None, batch_size: int | None, policy_path: str | None) -> list[str]:
    training = config["training"]
    dataset_root = str(repo_path(training["dataset"]).resolve())
    output_dir = str(repo_path(training["checkpoint_dir"]).resolve())
    policy_repo_id = f"local/{Path(training['checkpoint_dir']).name}"
    eff_steps = str(steps or training.get("steps", 3000))
    eff_save_freq = str(training.get("save_freq") or eff_steps)
    # Either finetune from a pretrained SmolVLA (recommended: loads the pretrained
    # VLM + action expert) or initialise a fresh smolvla policy. The legacy path
    # (no policy_path, no load_vlm_weights) trains the VLM from scratch, which is
    # almost certainly why the 14000 run reached low loss but 0 Isaac lifts.
    eff_policy_path = policy_path or training.get("policy_path")
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        "--dataset.repo_id",
        f"local/{Path(training['dataset']).name}",
        "--dataset.root",
        dataset_root,
    ]
    if eff_policy_path:
        # LeRobot 0.4.x finetunes from a pretrained policy via the special draccus
        # path directive `--policy.path=<dir-or-hf-id>` (train.py get_path_arg),
        # which loads the pretrained VLM + action expert. It MUST use the `=` form
        # as a single token; the space-separated form is rejected by the parser.
        # Remaining `--policy.*` flags become cli-overrides on the loaded config.
        cmd.append(f"--policy.path={eff_policy_path}")
        if training.get("infer_input_features_from_dataset", True):
            # SmolVLA base/checkpoint configs carry stale input feature metadata
            # (notably observation.state shape [6] and extra cameras). Clearing it
            # lets lerobot infer the correct camera/state contract from ds_meta.
            cmd.append("--policy.input_features=null")
    else:
        cmd.extend(["--policy.type", "smolvla"])
        if training.get("load_vlm_weights"):
            cmd.extend(["--policy.load_vlm_weights", "true"])
    # `--policy.*` flags must use the `=` form: when `--policy.path=` loads a
    # pretrained config, draccus treats the rest as cli-overrides and a
    # space-separated value falls through as an unrecognized positional.
    cmd.extend([
        f"--policy.device={training.get('device', 'cuda')}",
        "--policy.push_to_hub=false",
        f"--policy.repo_id={policy_repo_id}",
        "--output_dir",
        output_dir,
        "--batch_size",
        str(batch_size or training.get("batch_size", 4)),
        "--steps",
        eff_steps,
        "--save_freq",
        eff_save_freq,
        "--log_freq",
        "10",
        "--num_workers",
        "0",
        "--wandb.enable",
        "false",
    ])
    # Match the policy horizon to the corrected dense episode/control window. Only
    # override chunk_size when training from scratch; when finetuning from a base
    # the chunk_size must stay at the pretrained value to keep the action-expert
    # weight shapes. n_action_steps is inference-time and is always safe to set.
    if not eff_policy_path and training.get("chunk_size"):
        cmd.append(f"--policy.chunk_size={training['chunk_size']}")
    if training.get("n_action_steps"):
        cmd.append(f"--policy.n_action_steps={training['n_action_steps']}")
    for key in (
        "optimizer_lr",
        "optimizer_weight_decay",
        "optimizer_grad_clip_norm",
        "scheduler_warmup_steps",
        "scheduler_decay_steps",
        "scheduler_decay_lr",
        "freeze_vision_encoder",
        "train_expert_only",
        "train_state_proj",
        "num_steps",
    ):
        if key in training:
            value = training[key]
            if isinstance(value, bool):
                value = str(value).lower()
            cmd.append(f"--policy.{key}={value}")
    return cmd


def shell_quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_yaml_config(args.train_config)
    training = config["training"]
    if args.checkpoint_dir:
        training["checkpoint_dir"] = args.checkpoint_dir
    dataset_root = repo_path(training["dataset"]).resolve()
    checkpoint_dir = repo_path(training["checkpoint_dir"]).resolve()
    command = build_train_command(
        config,
        steps=args.steps,
        batch_size=args.batch_size,
        policy_path=args.policy_path,
    )

    report = {
        "training_config": training,
        "modules": {
            "lerobot": module_status("lerobot"),
            "lerobot_dataset": import_status("lerobot.datasets.lerobot_dataset"),
            "lerobot_train": import_status("lerobot.scripts.lerobot_train"),
            "smolvla_policy": import_status("lerobot.policies.smolvla.modeling_smolvla"),
            "torch": module_status("torch"),
            "transformers": module_status("transformers"),
        },
        "dataset": dataset_ready(dataset_root),
        "checkpoint_dir": prepare_checkpoint_dir(
            checkpoint_dir,
            overwrite_gitkeep_only=args.overwrite_output_dir,
        ),
        "command": command,
        "shell_command": shell_quote(command),
        "status": "prepared",
    }

    if args.command_output:
        command_output = Path(args.command_output)
        if not command_output.is_absolute():
            command_output = REPO_ROOT / command_output
        command_output.parent.mkdir(parents=True, exist_ok=True)
        command_output.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shell_quote(command) + "\n", encoding="utf-8")
        command_output.chmod(0o755)
        report["command_output"] = str(command_output)

    if args.run:
        if not report["dataset"]["ready"]:
            report["status"] = "blocked_dataset_missing"
        elif not report["checkpoint_dir"]["usable_for_new_run"]:
            report["status"] = "blocked_checkpoint_dir_not_empty"
        else:
            start = time.time()
            proc = subprocess.run(command, text=True, capture_output=True, check=False)
            report["run"] = {
                "returncode": proc.returncode,
                "duration_sec": time.time() - start,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
            report["status"] = "completed" if proc.returncode == 0 else "failed"

    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["status"] in {"prepared", "completed"}, "status": report["status"], "report": str(output)}, indent=2))
    return 0 if report["status"] in {"prepared", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
