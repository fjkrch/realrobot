#!/usr/bin/env python3
"""Evaluate an oracle or policy manifest and write a compact markdown report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL manifest to evaluate")
    parser.add_argument(
        "--output",
        default="synthetic_smolvla/reports/eval_manifest.md",
        help="markdown report path",
    )
    return parser


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def metric_rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def write_report(records: list[dict], output: Path) -> None:
    total = len(records)
    success = sum(1 for record in records if record.get("success_label"))
    wrong_object = sum(1 for record in records if record.get("wrong_object_lifted"))
    limit_exceeded = sum(1 for record in records if record.get("limit_exceeded"))
    by_object = Counter(str(record.get("target_object", "unknown")) for record in records)

    lines = [
        "# Synthetic SmolVLA Manifest Evaluation",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Episodes | {total} | 1.000 |",
        f"| Success | {success} | {metric_rate(success, total):.3f} |",
        f"| Wrong object | {wrong_object} | {metric_rate(wrong_object, total):.3f} |",
        f"| Limit exceeded | {limit_exceeded} | {metric_rate(limit_exceeded, total):.3f} |",
        "",
        "## Episodes By Target",
        "",
        "| Target | Episodes |",
        "|---|---:|",
    ]
    for target, count in sorted(by_object.items()):
        lines.append(f"| {target} | {count} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This report evaluates manifest labels. Isaac physics and RGB-policy evaluation are pending.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    records = load_records(manifest)
    write_report(records, output)
    print(json.dumps({"ok": True, "manifest": str(manifest), "report": str(output), "episodes": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

