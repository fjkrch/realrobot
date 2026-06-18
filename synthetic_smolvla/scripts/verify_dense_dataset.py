#!/usr/bin/env python3
"""Verify a dense Isaac-camera LeRobot dataset before training.

Checks the pre-training gates from docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md:
  * dataset loads with LeRobotDataset
  * task / state / action / image shapes are present and correct
  * frames inside a single episode are NOT static (the core fix vs the old
    5-keyframe static-placeholder dataset)

Simulation/data only. Never touches the robot.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_hw3_uint8(img) -> np.ndarray:
    """Normalize a LeRobot image sample to HxWx3 uint8 for hashing/diffing."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0 if arr.max() <= 1.0 + 1e-6 else arr
        arr = arr.astype(np.uint8)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--image-key", default="observation.images.camera1")
    args = ap.parse_args()

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    print(f"[verify] root={args.root}")
    print(f"[verify] total frames (len)={len(ds)}  episodes={ds.num_episodes}  fps={ds.fps}")

    s0 = ds[0]
    print(f"[verify] sample0 task={s0.get('task')!r}")
    print(f"[verify] state shape={tuple(s0['observation.state'].shape)} dtype={s0['observation.state'].dtype}")
    print(f"[verify] action shape={tuple(s0['action'].shape)} dtype={s0['action'].dtype}")
    img0 = s0[args.image_key]
    print(f"[verify] image key={args.image_key} raw shape={tuple(img0.shape) if hasattr(img0,'shape') else None} dtype={getattr(img0,'dtype',None)}")

    # Episode 0 frame range from metadata.
    ep_from = int(ds.meta.episodes["dataset_from_index"][0])
    ep_to = int(ds.meta.episodes["dataset_to_index"][0])
    n = ep_to - ep_from
    print(f"[verify] episode 0 spans frames [{ep_from}, {ep_to}) => {n} frames")

    # Sample several frames across episode 0 and confirm they differ.
    idxs = sorted({ep_from, ep_from + n // 4, ep_from + n // 2, ep_from + (3 * n) // 4, ep_to - 1})
    frames = {i: to_hw3_uint8(ds[i][args.image_key]) for i in idxs}
    base_i = idxs[0]
    base = frames[base_i].astype(np.int16)
    print("[verify] per-frame vs first-frame mean-abs-diff (0 => identical/static):")
    max_diff = 0.0
    for i in idxs:
        diff = float(np.abs(frames[i].astype(np.int16) - base).mean())
        max_diff = max(max_diff, diff)
        print(f"    frame {i:5d}: mean|Δ|={diff:7.3f}  mean_px={frames[i].mean():7.3f}")

    # Also confirm state actually changes across the episode.
    st_first = ds[ep_from]["observation.state"].detach().cpu().numpy()
    st_last = ds[ep_to - 1]["observation.state"].detach().cpu().numpy()
    state_delta = np.abs(st_last - st_first)
    print(f"[verify] |state_last - state_first| per dim = {np.round(state_delta, 3).tolist()}")

    static = max_diff < 1.0
    print(f"\n[verify] RESULT: images {'STATIC (FAIL)' if static else 'MOVE across episode (OK)'} "
          f"(max mean|Δ|={max_diff:.3f})")
    return 1 if static else 0


if __name__ == "__main__":
    raise SystemExit(main())
