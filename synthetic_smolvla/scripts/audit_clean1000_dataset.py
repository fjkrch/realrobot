#!/usr/bin/env python3
"""Audit the clean balanced 1000-episode real-table zero-pose dataset.

Reads the per-object kept-manifest JSONL files produced by
``collect_dense_isaac_dataset.py`` and (optionally) the merged LeRobot dataset, and
emits a markdown report plus a machine-readable JSON audit next to it. Every required
proof from the recollection contract is checked as a pass/fail gate:

  * source attempts, retained count, per-target balance (250 each)
  * target rise min/max/mean (>= 0.04 m lift)
  * max surface sweep before lift, object-object min distance
  * finger/table clearance + tabletop-penetration proof (authoritative finger-body check)
  * object-pushed-down proof
  * gripper command min/max (never closes past -3 deg)
  * refined action-clip proof (0 genuine unsafe clips; old limit_exceeded rate shown for contrast)
  * 100 VLA-step proof, 5-substep proof
  * state/action/image shape proof (via the merged LeRobot dataset)
  * sample frame paths

Data/audit only. Never touches the robot. The manifest gates run without Isaac/GPU; the
LeRobot dataset gates run only if ``--dataset-root``/``--repo-id`` are given and importable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _parse_target_counts(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        name, _, count = piece.partition("=")
        out[name.strip()] = int(count)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", action="append", required=True,
                   help="per-object collection manifest JSONL (repeat for each object)")
    p.add_argument("--dataset-root", default="", help="merged LeRobot dataset root (optional)")
    p.add_argument("--repo-id", default="", help="merged LeRobot repo id (optional)")
    p.add_argument("--image-key", default="observation.images.camera1")
    p.add_argument("--target-counts",
                   default="orange_ball=250,red_cube=250,green_cube=250,blue_cube=250",
                   help="expected retained count per target")
    p.add_argument("--expected-episode-len", type=int, default=100)
    p.add_argument("--expected-substeps", type=int, default=5)
    p.add_argument("--expected-state-dim", type=int, default=8)
    p.add_argument("--expected-image", type=int, default=256)
    p.add_argument("--lift-threshold-m", type=float, default=0.04)
    p.add_argument("--max-gripper-close-deg", type=float, default=-3.0,
                   help="gripper command must never exceed this (close past it)")
    p.add_argument("--max-arm-action-delta-deg", type=float, default=0.0,
                   help="optional gate: max recorded arm command delta per control step (0 disables)")
    p.add_argument("--out-md", default="synthetic_smolvla/reports/openarm_real_table_zero_v2_clean1000_audit.md")
    p.add_argument("--out-json", default="synthetic_smolvla/reports/openarm_real_table_zero_v2_clean1000_audit.json")
    return p


def audit_manifests(manifests: list[Path], args) -> dict:
    """Compute all manifest-derived stats and gates."""
    source_total = 0
    kept_rows: list[dict] = []
    source_by_target: dict[str, int] = {}
    kept_by_target: dict[str, int] = {}
    per_manifest = []

    for mpath in manifests:
        rows = _load_jsonl(mpath)
        kept = [r for r in rows if r.get("kept")]
        source_total += len(rows)
        for r in rows:
            source_by_target[r.get("target_object", "?")] = source_by_target.get(r.get("target_object", "?"), 0) + 1
        for r in kept:
            kept_by_target[r.get("target_object", "?")] = kept_by_target.get(r.get("target_object", "?"), 0) + 1
        kept_rows.extend(kept)
        per_manifest.append({
            "manifest": str(mpath),
            "source": len(rows),
            "kept": len(kept),
        })

    retained = len(kept_rows)
    gates: list[dict] = []

    def gate(name: str, passed: bool, detail: str) -> None:
        gates.append({"gate": name, "pass": bool(passed), "detail": detail})

    # --- balance ---
    expected = _parse_target_counts(args.target_counts)
    balance_ok = all(kept_by_target.get(t, 0) == c for t, c in expected.items())
    total_ok = retained == sum(expected.values())
    gate("per_target_balance", balance_ok,
         f"expected={expected} got={ {t: kept_by_target.get(t, 0) for t in expected} }")
    gate("retained_total", total_ok, f"retained={retained} expected_total={sum(expected.values())}")

    def fcol(key: str) -> list[float]:
        return [float(r[key]) for r in kept_rows if key in r and r[key] is not None]

    # --- target rise ---
    rises = fcol("target_rise_m")
    rise_min = min(rises) if rises else float("nan")
    rise_ok = bool(rises) and rise_min >= args.lift_threshold_m
    gate("target_rise_min", rise_ok,
         f"min={rise_min:.5f} max={max(rises) if rises else float('nan'):.5f} "
         f"mean={statistics.fmean(rises) if rises else float('nan'):.5f} threshold={args.lift_threshold_m}")

    # --- sweep / object distance ---
    sweeps = fcol("max_surface_sweep_m")
    swept_flags = [bool(r.get("object_swept_or_slid")) for r in kept_rows]
    gate("no_object_sweep", not any(swept_flags),
         f"max_surface_sweep_m_over_kept={max(sweeps) if sweeps else float('nan'):.5f} swept_count={sum(swept_flags)}")
    obj_dists = fcol("min_object_distance_m")
    obj_coll = [bool(r.get("object_collision")) for r in kept_rows]
    gate("no_object_collision", not any(obj_coll),
         f"min_object_distance_m={min(obj_dists) if obj_dists else float('nan'):.5f} collision_count={sum(obj_coll)}")

    # --- tabletop penetration (authoritative finger-body check) ---
    pen_flags = [bool(r.get("tabletop_penetration")) for r in kept_rows]
    finger_clear = fcol("min_finger_table_clearance_m")
    has_pen_field = all("tabletop_penetration" in r for r in kept_rows)
    finite_clear = [c for c in finger_clear if math.isfinite(c)]
    gate("no_tabletop_penetration", has_pen_field and not any(pen_flags),
         f"tabletop_penetration_count={sum(pen_flags)} has_field={has_pen_field} "
         f"min_finger_table_clearance_over_footprint_m={min(finite_clear) if finite_clear else float('inf')}")

    # --- object pushed down ---
    push_flags = [bool(r.get("object_pushed_down")) for r in kept_rows]
    has_push_field = all("object_pushed_down" in r for r in kept_rows)
    gate("no_object_pushed_down", has_push_field and not any(push_flags),
         f"object_pushed_down_count={sum(push_flags)} has_field={has_push_field}")

    # --- gripper command cap ---
    gmax = fcol("gripper_cmd_max_deg")
    gmin = fcol("gripper_cmd_min_deg")
    has_grip_field = bool(gmax)
    worst_close = max(gmax) if gmax else float("nan")  # closest-to-0 command across kept
    grip_ok = has_grip_field and worst_close <= args.max_gripper_close_deg + 1e-6
    gate("gripper_never_past_cap", grip_ok,
         f"gripper_cmd_max_deg(worst close)={worst_close:.3f} cap={args.max_gripper_close_deg} "
         f"gripper_cmd_min_deg(most open)={min(gmin) if gmin else float('nan'):.3f}")

    # --- refined action clip (authoritative) + old limit_exceeded for contrast ---
    refined_flags = [bool(r.get("refined_action_clip")) for r in kept_rows]
    has_refined_field = all("refined_action_clip" in r for r in kept_rows)
    refined_max = fcol("max_refined_action_clip_deg")
    old_limit = [bool(r.get("limit_exceeded")) for r in kept_rows]
    gate("no_refined_action_clip", has_refined_field and not any(refined_flags),
         f"refined_action_clip_count={sum(refined_flags)} has_field={has_refined_field} "
         f"max_refined_action_clip_deg={max(refined_max) if refined_max else float('nan'):.4f} "
         f"(legacy limit_exceeded rate over kept={sum(old_limit)}/{retained} — known misleading proxy)")

    # --- recorded arm command slew limit ---
    arm_delta = fcol("max_arm_action_delta_deg")
    has_arm_delta_field = all("max_arm_action_delta_deg" in r for r in kept_rows)
    worst_arm_delta = max(arm_delta) if arm_delta else float("nan")
    if args.max_arm_action_delta_deg > 0.0:
        arm_delta_ok = has_arm_delta_field and bool(arm_delta) and worst_arm_delta <= args.max_arm_action_delta_deg + 1e-3
        gate("arm_action_delta_within_limit", arm_delta_ok,
             f"max_arm_action_delta_deg={worst_arm_delta:.4f} "
             f"limit={args.max_arm_action_delta_deg:.4f} has_field={has_arm_delta_field}")

    # --- 100-step proof ---
    ep_lens = [int(r.get("episode_len", -1)) for r in kept_rows]
    gate("episode_len_100", bool(ep_lens) and all(n == args.expected_episode_len for n in ep_lens),
         f"distinct_episode_len={sorted(set(ep_lens))} expected={args.expected_episode_len}")

    # --- 5-substep proof ---
    substeps = [int(r.get("substeps", -1)) for r in kept_rows]
    gate("substeps_5", bool(substeps) and all(s == args.expected_substeps for s in substeps),
         f"distinct_substeps={sorted(set(substeps))} expected={args.expected_substeps}")

    # --- sample frames ---
    sample_frames = []
    for r in kept_rows:
        if r.get("sample_frames"):
            sample_frames.extend(r["sample_frames"])

    return {
        "source_total": source_total,
        "retained": retained,
        "source_by_target": source_by_target,
        "kept_by_target": kept_by_target,
        "per_manifest": per_manifest,
        "target_rise": {"min": min(rises) if rises else None, "max": max(rises) if rises else None,
                        "mean": statistics.fmean(rises) if rises else None},
        "max_surface_sweep_m": max(sweeps) if sweeps else None,
        "min_object_distance_m": min(obj_dists) if obj_dists else None,
        "min_finger_table_clearance_over_footprint_m": min(finite_clear) if finite_clear else None,
        "gripper_cmd_worst_close_deg": worst_close,
        "gripper_cmd_most_open_deg": min(gmin) if gmin else None,
        "legacy_limit_exceeded_kept": sum(old_limit),
        "max_arm_action_delta_deg": worst_arm_delta if arm_delta else None,
        "sample_frames": sample_frames[:16],
        "gates": gates,
    }


def audit_dataset(args, gates: list[dict]) -> dict:
    """LeRobot dataset gates (shape, per-episode 100 frames, motion). Optional."""
    out: dict = {"checked": False}
    if not args.dataset_root or not args.repo_id:
        gates.append({"gate": "lerobot_dataset", "pass": True,
                      "detail": "skipped (no --dataset-root/--repo-id)"})
        return out
    try:
        import numpy as np  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from verify_dense_dataset import to_hw3_uint8  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        gates.append({"gate": "lerobot_dataset", "pass": False,
                      "detail": f"import failed: {type(exc).__name__}: {exc}"})
        return out

    ds = LeRobotDataset(repo_id=args.repo_id, root=str(_abs(args.dataset_root)))
    out["checked"] = True
    out["num_episodes"] = int(ds.num_episodes)
    out["num_frames"] = int(len(ds))
    out["fps"] = float(ds.fps)

    s0 = ds[0]
    state_shape = tuple(s0["observation.state"].shape)
    action_shape = tuple(s0["action"].shape)
    img = to_hw3_uint8(s0[args.image_key])
    img_shape = tuple(img.shape)
    out["state_shape"] = list(state_shape)
    out["action_shape"] = list(action_shape)
    out["image_shape"] = list(img_shape)

    shape_ok = (
        state_shape == (args.expected_state_dim,)
        and action_shape == (args.expected_state_dim,)
        and img_shape == (args.expected_image, args.expected_image, 3)
    )
    gates.append({"gate": "lerobot_shapes", "pass": shape_ok,
                  "detail": f"state={state_shape} action={action_shape} image={img_shape} "
                            f"expected state/action=({args.expected_state_dim},) image=({args.expected_image},{args.expected_image},3)"})

    # Per-episode frame count == expected_episode_len.
    bad_lens = []
    for ep in range(ds.num_episodes):
        n = int(ds.meta.episodes["dataset_to_index"][ep]) - int(ds.meta.episodes["dataset_from_index"][ep])
        if n != args.expected_episode_len:
            bad_lens.append((ep, n))
    gate_len_ok = not bad_lens
    out["bad_episode_lengths"] = bad_lens[:10]
    gates.append({"gate": "lerobot_episode_len", "pass": gate_len_ok,
                  "detail": f"episodes={ds.num_episodes} all=={args.expected_episode_len}? "
                            f"{'yes' if gate_len_ok else f'no, {len(bad_lens)} bad e.g. {bad_lens[:3]}'}"})

    # Motion check on episode 0.
    ep_from = int(ds.meta.episodes["dataset_from_index"][0])
    ep_to = int(ds.meta.episodes["dataset_to_index"][0])
    n = ep_to - ep_from
    idxs = sorted({ep_from, ep_from + n // 4, ep_from + n // 2, ep_from + (3 * n) // 4, ep_to - 1})
    base = to_hw3_uint8(ds[idxs[0]][args.image_key]).astype("int16")
    max_diff = 0.0
    for i in idxs:
        diff = float(abs(to_hw3_uint8(ds[i][args.image_key]).astype("int16") - base).mean())
        max_diff = max(max_diff, diff)
    out["episode0_max_frame_diff"] = max_diff
    gates.append({"gate": "lerobot_frames_move", "pass": max_diff >= 1.0,
                  "detail": f"episode0 max mean|Δ|={max_diff:.3f} (>=1.0 means frames move)"})
    return out


def write_report(manifest_audit: dict, dataset_audit: dict, args, out_md: Path) -> None:
    gates = manifest_audit["gates"]
    overall = all(g["pass"] for g in gates)
    tr = manifest_audit["target_rise"]
    lines = [
        "# Clean 1000-Episode Real-Table Zero-Pose Dataset — Audit",
        "",
        f"Overall: **{'PASS' if overall else 'FAIL'}**",
        "",
        "Simulation/data only. No real robot, SSH, CAN, replay, or mirror.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    for g in gates:
        lines.append(f"| {g['gate']} | {'PASS' if g['pass'] else 'FAIL'} | {g['detail']} |")
    lines += [
        "",
        "## Counts",
        "",
        f"- Source attempts: `{manifest_audit['source_total']}`",
        f"- Retained: `{manifest_audit['retained']}`",
        f"- Retained by target: `{manifest_audit['kept_by_target']}`",
        f"- Source by target: `{manifest_audit['source_by_target']}`",
        "",
        "### Per-manifest",
        "",
        "| Manifest | Source | Kept |",
        "|---|---:|---:|",
    ]
    for m in manifest_audit["per_manifest"]:
        lines.append(f"| `{m['manifest']}` | {m['source']} | {m['kept']} |")
    lines += [
        "",
        "## Proof stats (over retained episodes)",
        "",
        f"- Target rise m: min `{tr['min']}` / max `{tr['max']}` / mean `{tr['mean']}`",
        f"- Max surface sweep m: `{manifest_audit['max_surface_sweep_m']}`",
        f"- Object-object min distance m: `{manifest_audit['min_object_distance_m']}`",
        f"- Min finger/table clearance over footprint m: `{manifest_audit['min_finger_table_clearance_over_footprint_m']}`",
        f"- Gripper cmd worst close deg (<= {args.max_gripper_close_deg}): `{manifest_audit['gripper_cmd_worst_close_deg']}`",
        f"- Gripper cmd most open deg: `{manifest_audit['gripper_cmd_most_open_deg']}`",
        f"- Max arm action delta deg: `{manifest_audit['max_arm_action_delta_deg']}`",
        f"- Legacy `limit_exceeded` over retained (known misleading proxy): `{manifest_audit['legacy_limit_exceeded_kept']}`",
        "",
        "## LeRobot dataset gates",
        "",
    ]
    if dataset_audit.get("checked"):
        lines += [
            f"- Episodes: `{dataset_audit['num_episodes']}`  Frames: `{dataset_audit['num_frames']}`  fps: `{dataset_audit['fps']}`",
            f"- state shape: `{dataset_audit['state_shape']}`  action shape: `{dataset_audit['action_shape']}`  image shape: `{dataset_audit['image_shape']}`",
            f"- Episode 0 max frame diff: `{dataset_audit.get('episode0_max_frame_diff')}`",
        ]
    else:
        lines.append("- (skipped — no `--dataset-root`/`--repo-id`)")
    lines += [
        "",
        "## Sample frames",
        "",
    ]
    for sf in manifest_audit["sample_frames"]:
        lines.append(f"- `{sf}`")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    manifests = [_abs(m) for m in args.manifest]
    missing = [str(m) for m in manifests if not m.exists()]
    if missing:
        raise SystemExit(f"Missing manifest(s): {missing}")

    manifest_audit = audit_manifests(manifests, args)
    dataset_audit = audit_dataset(args, manifest_audit["gates"])

    out_md = _abs(args.out_md)
    out_json = _abs(args.out_json)
    write_report(manifest_audit, dataset_audit, args, out_md)

    overall = all(g["pass"] for g in manifest_audit["gates"])
    payload = {
        "ok": overall,
        "source_total": manifest_audit["source_total"],
        "retained": manifest_audit["retained"],
        "kept_by_target": manifest_audit["kept_by_target"],
        "source_by_target": manifest_audit["source_by_target"],
        "target_rise": manifest_audit["target_rise"],
        "max_surface_sweep_m": manifest_audit["max_surface_sweep_m"],
        "min_object_distance_m": manifest_audit["min_object_distance_m"],
        "min_finger_table_clearance_over_footprint_m": manifest_audit["min_finger_table_clearance_over_footprint_m"],
        "gripper_cmd_worst_close_deg": manifest_audit["gripper_cmd_worst_close_deg"],
        "gripper_cmd_most_open_deg": manifest_audit["gripper_cmd_most_open_deg"],
        "legacy_limit_exceeded_kept": manifest_audit["legacy_limit_exceeded_kept"],
        "sample_frames": manifest_audit["sample_frames"],
        "dataset": dataset_audit,
        "gates": manifest_audit["gates"],
        "report_md": str(out_md),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"ok": overall, "report_md": str(out_md), "report_json": str(out_json),
                      "retained": manifest_audit["retained"],
                      "failed_gates": [g["gate"] for g in manifest_audit["gates"] if not g["pass"]]}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
