#!/usr/bin/env python3
"""Run the synthetic-only OpenArm SmolVLA pipeline phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from sim_contract import PROJECT_ROOT


REPO_ROOT = PROJECT_ROOT.parent
PYTHON = sys.executable


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--use-conda", action="store_true", help="run LeRobot-dependent phases in env_isaaclab")
    parser.add_argument("--skip-v2", action="store_true", help="skip the large V2 dataset export")
    parser.add_argument("--run-training", action="store_true", help="launch LeRobot training commands")
    parser.add_argument("--report", default="synthetic_smolvla/reports/final_pipeline_status.md")
    return parser


def command_prefix(use_conda: bool) -> list[str]:
    if use_conda:
        return ["conda", "run", "-n", "env_isaaclab", "python"]
    return [PYTHON]


def run_step(name: str, cmd: list[str], *, timeout: int | None = None) -> dict:
    start = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_sec": time.time() - start,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-3000:],
        "ok": proc.returncode == 0,
    }


def write_report(path: Path, mode: str, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synthetic SmolVLA Pipeline Status",
        "",
        f"Mode: `{mode}`",
        "",
        "| Step | Status | Seconds |",
        "|---|---:|---:|",
    ]
    for step in steps:
        status = "ok" if step["ok"] else f"failed ({step['returncode']})"
        lines.append(f"| {step['name']} | {status} | {step['duration_sec']:.1f} |")
    lines.extend(["", "## Commands", ""])
    for step in steps:
        lines.append(f"### {step['name']}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(step["cmd"]))
        lines.append("```")
        if not step["ok"]:
            lines.append("")
            lines.append("stderr tail:")
            lines.append("")
            lines.append("```text")
            lines.append(step["stderr_tail"] or step["stdout_tail"])
            lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    prefix = command_prefix(args.use_conda)
    smoke = args.mode == "smoke"
    v1_episodes = 16 if smoke else 1000
    v2_episodes = 32 if smoke else 5000
    oracle_trials = 100 if not smoke else 16
    stress_trials = 64 if smoke else 1000

    steps: list[dict] = []
    steps.append(
        run_step(
            "scene dry-run",
            [PYTHON, "synthetic_smolvla/scripts/make_scene.py", "--dry-run", "--manifest", "synthetic_smolvla/reports/scene_manifest.json"],
        )
    )
    steps.append(
        run_step(
            "oracle acceptance manifest",
            [
                *prefix,
                "synthetic_smolvla/scripts/collect_oracle_demos.py",
                "--episodes",
                str(oracle_trials),
                "--output",
                "synthetic_smolvla/reports/oracle_acceptance_manifest.jsonl",
            ],
        )
    )
    steps.append(
        run_step(
            "oracle acceptance report",
            [
                PYTHON,
                "synthetic_smolvla/scripts/eval_smolvla.py",
                "--manifest",
                "synthetic_smolvla/reports/oracle_acceptance_manifest.jsonl",
                "--output",
                "synthetic_smolvla/reports/oracle_acceptance.md",
            ],
        )
    )
    steps.append(
        run_step(
            "dataset v1 export",
            [
                *prefix,
                "synthetic_smolvla/scripts/collect_oracle_demos.py",
                "--dataset-config",
                "synthetic_smolvla/configs/dataset_v1.yaml",
                "--episodes",
                str(v1_episodes),
                "--export-lerobot",
                "--overwrite",
            ],
            timeout=None,
        )
    )
    steps.append(
        run_step(
            "train v1 prepare",
            [
                *prefix,
                "synthetic_smolvla/scripts/train_smolvla.py",
                "--train-config",
                "synthetic_smolvla/configs/train_v1.yaml",
                "--output",
                "synthetic_smolvla/reports/train_v1_preflight.json",
                "--command-output",
                "synthetic_smolvla/reports/train_v1.sh",
                "--overwrite-output-dir",
                *(["--run"] if args.run_training else []),
            ],
        )
    )
    steps.append(
        run_step(
            "eval v1 manifest",
            [
                PYTHON,
                "synthetic_smolvla/scripts/eval_smolvla.py",
                "--manifest",
                "synthetic_smolvla/datasets/openarm_synth_v1/oracle_manifest.jsonl",
                "--output",
                "synthetic_smolvla/reports/eval_v1.md",
            ],
        )
    )
    if not args.skip_v2:
        steps.append(
            run_step(
                "dataset v2 export",
                [
                    *prefix,
                    "synthetic_smolvla/scripts/collect_oracle_demos.py",
                    "--dataset-config",
                    "synthetic_smolvla/configs/dataset_v2.yaml",
                    "--episodes",
                    str(v2_episodes),
                    "--export-lerobot",
                    "--overwrite",
                ],
            )
        )
        steps.append(
            run_step(
                "train v2 prepare",
                [
                    *prefix,
                    "synthetic_smolvla/scripts/train_smolvla.py",
                    "--train-config",
                    "synthetic_smolvla/configs/train_v2.yaml",
                    "--output",
                    "synthetic_smolvla/reports/train_v2_preflight.json",
                    "--command-output",
                    "synthetic_smolvla/reports/train_v2.sh",
                    "--overwrite-output-dir",
                    *(["--run"] if args.run_training else []),
                ],
            )
        )
    steps.append(
        run_step(
            "stress test report",
            [
                PYTHON,
                "synthetic_smolvla/scripts/stress_test.py",
                "--episodes",
                str(stress_trials),
                "--output",
                "synthetic_smolvla/reports/stress_test_v2.md",
            ],
        )
    )

    report = Path(args.report)
    if not report.is_absolute():
        report = REPO_ROOT / report
    write_report(report, args.mode, steps)

    ok = all(step["ok"] for step in steps)
    print(json.dumps({"ok": ok, "report": str(report), "steps": steps}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

