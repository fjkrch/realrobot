#!/usr/bin/env python3
"""Run a SmolVLA Isaac eval sweep with optional process-level parallelism.

Simulation only. This launches ``eval_vla_isaac.py`` and never touches a real
robot, SSH, CAN, replay, or mirror scripts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    "synthetic_smolvla/checkpoints/"
    "smolvla_openarm_real_table_zero_lift5cm_v2_extra_right_plus_start20_1500_direct_from010_state8_lr3e5/"
    "checkpoints"
)
DEFAULT_CONFIG = "synthetic_smolvla/configs/scene_openarm_real_table_zero_train_right_fallback_v1.yaml"
DEFAULT_PREFIX = "openarm_real_table_zero_lift5cm_1500_direct_eval"
DEFAULT_ISAAC_PYTHON = "/home/chyanin/IsaacLab/isaaclab_python.sh"


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _int_csv(value: str) -> list[int]:
    return [int(item) for item in _csv(value)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_complete(jsonl: Path, expected_trials: int) -> bool:
    if not jsonl.exists():
        return False
    try:
        records = _read_jsonl(jsonl)
    except (OSError, json.JSONDecodeError):
        return False
    return len(records) == expected_trials


def _nvidia_total_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    first = out.strip().splitlines()[0].strip()
    return int(first) if first else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="checkpoint root containing step/checkpoint dirs")
    parser.add_argument("--checkpoints", default="003000,006000,009000,012000,015000")
    parser.add_argument("--steps", default="100,150", help="comma-separated steps-per-trial values")
    parser.add_argument("--jobs", type=int, default=1, help="parallel eval processes")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--target-filter", default="orange_ball,red_cube")
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--jitter-x-m", type=float, default=0.0)
    parser.add_argument("--jitter-y-m", type=float, default=0.0)
    parser.add_argument("--max-gripper-close-deg", type=float, default=-3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isaac-python", default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--reports-dir", default="synthetic_smolvla/reports")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--skip-complete", action="store_true", default=True)
    parser.add_argument("--rerun-complete", dest="skip_complete", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--estimated-gpu-mb-per-job",
        type=int,
        default=5500,
        help="safety estimate for one Isaac+SmolVLA eval process",
    )
    parser.add_argument(
        "--force-unsafe-parallel",
        action="store_true",
        help="allow CUDA jobs even when estimated VRAM is too small",
    )
    return parser


def _job_paths(reports_dir: Path, prefix: str, checkpoint: str, steps: int) -> tuple[Path, Path, Path]:
    base = f"{prefix}_ckpt{checkpoint}_m3_steps{steps}"
    return (
        reports_dir / f"{base}.jsonl",
        reports_dir / f"{base}.md",
        reports_dir / f"{base}.log",
    )


def _run_job(cmd: list[str], log_path: Path) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {"returncode": proc.returncode, "duration_sec": round(time.time() - started, 3)}


def main() -> int:
    args = _build_parser().parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    if args.jobs > 1 and args.device.startswith("cuda") and not args.force_unsafe_parallel:
        total_mb = _nvidia_total_mb()
        need_mb = args.jobs * args.estimated_gpu_mb_per_job
        if total_mb is not None and need_mb > total_mb:
            raise SystemExit(
                "Refusing unsafe CUDA parallel eval: "
                f"jobs={args.jobs} needs about {need_mb} MB, GPU has {total_mb} MB. "
                "Use --jobs 1, lower --estimated-gpu-mb-per-job after measuring, or pass "
                "--force-unsafe-parallel if you accept OOM risk."
            )

    root = _abs(args.root)
    reports_dir = _abs(args.reports_dir)
    checkpoints = _csv(args.checkpoints)
    steps_values = _int_csv(args.steps)
    jobs = []
    skipped = []
    for checkpoint in checkpoints:
        ckpt_dir = root / checkpoint / "pretrained_model"
        for steps in steps_values:
            jsonl, md, log = _job_paths(reports_dir, args.output_prefix, checkpoint, steps)
            if args.skip_complete and _is_complete(jsonl, args.trials):
                skipped.append({"checkpoint": checkpoint, "steps": steps, "jsonl": str(jsonl)})
                continue
            cmd = [
                args.isaac_python,
                "synthetic_smolvla/scripts/eval_vla_isaac.py",
                "--config",
                args.config,
                "--checkpoint",
                str(ckpt_dir),
                "--trials",
                str(args.trials),
                "--target-filter",
                args.target_filter,
                "--seed",
                str(args.seed),
                "--jitter-x-m",
                str(args.jitter_x_m),
                "--jitter-y-m",
                str(args.jitter_y_m),
                "--max-gripper-close-deg",
                str(args.max_gripper_close_deg),
                "--steps-per-trial",
                str(steps),
                "--headless",
                "--device",
                args.device,
                "--output-jsonl",
                str(jsonl),
                "--output-md",
                str(md),
            ]
            jobs.append(
                {
                    "checkpoint": checkpoint,
                    "steps": steps,
                    "cmd": cmd,
                    "jsonl": str(jsonl),
                    "report": str(md),
                    "log": str(log),
                }
            )

    print(json.dumps({"queued": len(jobs), "skipped": skipped, "jobs": args.jobs}, indent=2), flush=True)
    if args.dry_run or not jobs:
        return 0

    summary: list[dict[str, Any]] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(_run_job, job["cmd"], Path(job["log"])): job for job in jobs}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                result = future.result()
                record = {k: job[k] for k in ("checkpoint", "steps", "jsonl", "report", "log")}
                record.update(result)
                summary.append(record)
                if result["returncode"] != 0:
                    failures += 1
                print(json.dumps(record, sort_keys=True), flush=True)

    summary_path = reports_dir / f"{args.output_prefix}_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
