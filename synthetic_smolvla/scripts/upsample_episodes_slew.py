#!/usr/bin/env python3
"""Upsample saved NPZ episodes so no 8D command exceeds a per-step slew cap.

The grasp is planned/executed freely (any per-control-step joint delta). To honour
the "never move faster than 20 deg/s" rule, this post-processor resamples each
episode at 10 Hz so that between consecutive *saved* commands no joint (or the
gripper) moves more than ``--max-step-deg`` degrees. Where the raw motion between
two control steps exceeds the cap, intermediate commands are linearly inserted
(upsampling), rather than clamped — so the trajectory is preserved, just retimed.

state/action are interpolated; the camera frame is held from the source step
nearest each inserted command (a sub-2-degree move barely changes the image).

This keeps the dataset contract: 10 Hz, <= max-step-deg per command on all 8 dims.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]


def upsample_trajectory(state: np.ndarray, action: np.ndarray, camera: np.ndarray, max_step_deg: float):
    """Return (state, action, camera) resampled so consecutive action deltas <= cap.

    For each original segment [i, i+1], compute how many sub-steps are needed so the
    largest per-dim delta is <= max_step_deg, then linearly interpolate that many
    commands. The camera frame is held from the source step nearest each command.
    """
    T = action.shape[0]
    if T <= 1:
        return state, action, camera
    out_state = [state[0]]
    out_action = [action[0]]
    out_camera = [camera[0]]
    for i in range(T - 1):
        a0, a1 = action[i], action[i + 1]
        s0, s1 = state[i], state[i + 1]
        delta = np.abs(a1 - a0)
        max_delta = float(delta.max())
        n_sub = max(1, int(math.ceil(max_delta / max_step_deg - 1e-9)))
        for k in range(1, n_sub + 1):
            frac = k / n_sub
            out_action.append(a0 + (a1 - a0) * frac)
            out_state.append(s0 + (s1 - s0) * frac)
            # hold the camera frame from whichever source step is nearest in time
            out_camera.append(camera[i + 1] if frac >= 0.5 else camera[i])
    return (
        np.stack(out_state, axis=0).astype(np.float32),
        np.stack(out_action, axis=0).astype(np.float32),
        np.stack(out_camera, axis=0),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True, help="dataset root containing episodes/*.npz and meta.json")
    p.add_argument("--output-root", required=True)
    p.add_argument("--max-step-deg", type=float, default=2.0, help="per-command slew cap on all 8 dims (deg)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    in_root = Path(args.input_root)
    in_root = in_root if in_root.is_absolute() else REPO_ROOT / in_root
    out_root = Path(args.output_root)
    out_root = out_root if out_root.is_absolute() else REPO_ROOT / out_root
    in_eps = sorted((in_root / "episodes").glob("episode_*.npz"))
    if not in_eps:
        raise SystemExit(f"no episodes under {in_root}/episodes")
    if out_root.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out_root)
    (out_root / "episodes").mkdir(parents=True, exist_ok=True)

    summary = []
    total_in = total_out = 0
    for idx, ep in enumerate(in_eps):
        d = np.load(ep, allow_pickle=True)
        state, action, camera = d["state"], d["action"], d["camera"]
        us_state, us_action, us_camera = upsample_trajectory(state, action, camera, args.max_step_deg)
        # verify cap
        max_after = float(np.abs(np.diff(us_action, axis=0)).max()) if us_action.shape[0] > 1 else 0.0
        out = out_root / "episodes" / f"episode_{idx:06d}.npz"
        np.savez_compressed(out, camera=us_camera, state=us_state, action=us_action,
                            task=d["task"], fps=np.asarray(args.fps, dtype=np.int32))
        total_in += int(action.shape[0])
        total_out += int(us_action.shape[0])
        summary.append({"episode": ep.name, "in_len": int(action.shape[0]),
                        "out_len": int(us_action.shape[0]), "max_step_deg_after": round(max_after, 4)})
        print(f"[upsample] {ep.name}: {action.shape[0]} -> {us_action.shape[0]} commands, max_step={max_after:.3f} deg", flush=True)

    meta = {"ok": True, "backend": "local_npz", "fps": args.fps,
            "num_episodes": len(in_eps), "num_frames": total_out,
            "max_step_deg": args.max_step_deg,
            "format": "episodes/episode_000000.npz with arrays camera,state,action,task",
            "source_root": str(in_root), "episodes": summary}
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"ok": True, "episodes": len(in_eps), "frames_in": total_in,
                      "frames_out": total_out, "output_root": str(out_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
