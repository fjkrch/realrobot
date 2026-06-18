#!/usr/bin/env python3
"""Check whether IsaacLab still works with a fresh empty logs directory.

This script is deliberately conservative. By default it does NOT delete logs:

1. Rename ``~/IsaacLab/logs`` to a timestamped backup.
2. Create a fresh empty ``~/IsaacLab/logs``.
3. Run lightweight smoke checks.
4. Restore the original logs directory.

If the checks pass, deleting the original logs directory should be safe from a
setup perspective. The old logs may still contain training history/checkpoints,
so review them before removing the backup manually.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = Path.home() / "IsaacLab" / "logs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOGS_DIR),
        help="IsaacLab logs directory to test",
    )
    parser.add_argument(
        "--keep-fresh-logs",
        action="store_true",
        help=(
            "leave the fresh empty logs directory in place and keep the old logs "
            "as a backup instead of restoring automatically"
        ),
    )
    parser.add_argument(
        "--skip-conda-smoke",
        action="store_true",
        help="skip the env_isaaclab Python import smoke check",
    )
    return parser


def run_check(name: str, command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print(f"\n[check] {name}")
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def backup_path_for(logs_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return logs_dir.with_name(f"{logs_dir.name}.cleanup-test-backup-{stamp}")


def main() -> int:
    args = build_arg_parser().parse_args()
    logs_dir = Path(args.logs_dir).expanduser().resolve()
    isaaclab_root = logs_dir.parent
    backup_dir = backup_path_for(logs_dir)
    moved_old_logs = False

    if not isaaclab_root.exists():
        raise SystemExit(f"IsaacLab root does not exist: {isaaclab_root}")
    if backup_dir.exists():
        raise SystemExit(f"Backup path already exists: {backup_dir}")

    print(f"[info] testing logs cleanup for: {logs_dir}")
    print("[info] no real robot commands will run")

    try:
        if logs_dir.exists():
            print(f"[step] moving old logs to: {backup_dir}")
            logs_dir.rename(backup_dir)
            moved_old_logs = True
        else:
            print("[step] logs directory did not exist; creating a fresh one")

        logs_dir.mkdir(parents=True, exist_ok=True)

        run_check(
            "repo safety tests",
            [sys.executable, "-m", "pytest", "tests/test_real_robot_safety.py", "-q"],
        )
        run_check(
            "synthetic scene dry run",
            [
                sys.executable,
                "synthetic_smolvla/scripts/make_scene.py",
                "--dry-run",
                "--manifest",
                "/tmp/realrobot_isaaclab_logs_cleanup_scene.json",
            ],
        )
        if not args.skip_conda_smoke:
            run_check(
                "env_isaaclab Python smoke",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "env_isaaclab",
                    "python",
                    "-c",
                    "import pathlib, torch; pathlib.Path('logs').mkdir(exist_ok=True); "
                    "print('env_isaaclab ok', torch.__version__)",
                ],
                cwd=isaaclab_root,
            )

    except Exception:
        if moved_old_logs and backup_dir.exists():
            print("\n[restore] check failed; restoring original logs")
            if logs_dir.exists():
                shutil.rmtree(logs_dir)
            backup_dir.rename(logs_dir)
        raise

    if args.keep_fresh_logs:
        print("\n[ok] checks passed")
        if moved_old_logs:
            print(f"[result] fresh logs kept at: {logs_dir}")
            print(f"[result] old logs backup kept at: {backup_dir}")
            print("[next] after you are sure, delete the backup with:")
            print(f"rm -rf {backup_dir}")
        else:
            print(f"[result] fresh logs directory exists at: {logs_dir}")
        return 0

    if moved_old_logs:
        print("\n[restore] checks passed; restoring original logs")
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        backup_dir.rename(logs_dir)
    print("\n[ok] checks passed. A fresh empty logs directory did not break the smoke checks.")
    print("[next] to actually clean space safely, run:")
    print(f"mv {logs_dir} {backup_dir}")
    print(f"mkdir -p {logs_dir}")
    print(f"# later, if everything still works: rm -rf {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
