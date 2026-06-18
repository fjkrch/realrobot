#!/usr/bin/env python3
"""Run a manifest-level all-objects-visible synthetic stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from collect_oracle_demos import main as collect_main
from eval_smolvla import load_records, write_report
from sim_contract import CONFIG_DIR


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9090)
    parser.add_argument(
        "--dataset-config",
        default=str(CONFIG_DIR / "dataset_v2.yaml"),
        help="dataset YAML to reuse for randomized settings",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="optional JSONL output; defaults to a temporary file",
    )
    parser.add_argument(
        "--output",
        default="synthetic_smolvla/reports/stress_test_manifest.md",
        help="markdown report path",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest) if args.manifest else Path(tempfile.gettempdir()) / "openarm_stress_manifest.jsonl"
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path

    import sys

    previous_argv = sys.argv[:]
    sys.argv = [
        "collect_oracle_demos.py",
        "--dataset-config",
        args.dataset_config,
        "--episodes",
        str(args.episodes),
        "--output",
        str(manifest_path),
        "--seed",
        str(args.seed),
        "--randomized",
        "--all-objects-visible-ratio",
        "1.0",
    ]
    try:
        collect_main()
    finally:
        sys.argv = previous_argv

    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    records = load_records(manifest_path)
    write_report(records, output)
    print(json.dumps({"ok": True, "manifest": str(manifest_path), "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

