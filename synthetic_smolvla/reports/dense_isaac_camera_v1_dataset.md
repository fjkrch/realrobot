# Dense Isaac-Camera SmolVLA Dataset (v1)

Date: 2026-06-17

This is the **corrected** dataset called for by the
[14k dataset audit](openarm_success_filtered_14000_dataset_audit.md) and the
[SmolVLA training handoff](../../docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md).
It replaces the weak `openarm_success_filtered_14000` baseline (5 sparse
keyframes, static placeholder images). The old baseline dataset is **kept** as
evidence; it is not deleted.

- LeRobot dataset root: `synthetic_smolvla/datasets/openarm_dense_isaac_camera_v1`
- LeRobot repo id: `local/openarm_dense_isaac_camera_v1`
- Producer: `synthetic_smolvla/scripts/collect_dense_isaac_dataset.py`
- Scene config: `synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml`
- Dataset spec: `synthetic_smolvla/configs/dataset_dense_isaac_camera_v1.yaml`
- Episode metadata JSONL: `synthetic_smolvla/reports/dense_isaac_camera_v1_manifest.jsonl`
- Sample frames: `synthetic_smolvla/reports/dense_isaac_camera_v1_samples/`

> Note: this file is regenerated (basic form) if the collector is re-run. The
> verification and audit-resolution sections below are maintained by hand.

## Collection result

Source episodes: 2080 across 16 envs x 130 rounds. Kept (measured target lift,
no wrong-object): **940 (45.2%)**. Each kept episode is a **dense** rollout with
**real Isaac camera frames at every control step** (50 steps/episode, 256x256 RGB).

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 2080 | 1.000 |
| Kept successes | 940 | 0.452 |
| Wrong-object lifts (source) | 0 | 0.000 |
| Limit-clamp episodes (source) | 2079 | 0.9995 |
| Frames (kept) | 47,000 | — |

### Targets

| Target | Source | Success | Fail | Success rate | Kept |
|---|---:|---:|---:|---:|---:|
| orange_ball | 709 | 273 | 436 | 39% | 273 |
| red_cube | 433 | 214 | 219 | 49% | 214 |
| green_cube | 470 | 278 | 192 | 59% | 278 |
| blue_cube | 468 | 175 | 293 | 37% | 175 |
| **Total** | **2080** | **940** | **1140** | **45.2%** | **940** |

The orange ball is over-sampled (target weight 1.6) because it is the hardest
class (round, rolls out of a parallel-jaw grip). The kept distribution is now
reasonably balanced (orange 273 vs the old 610-vs-2222 imbalance).

## Schema (LeRobot `meta/info.json`)

| Feature | dtype | Shape | Meaning |
|---|---|---|---|
| `observation.images.camera1` | image | 256 x 256 x 3 | **real Isaac scene camera**, captured per control step |
| `observation.state` | float32 | 8 | **measured** 7 arm joints + gripper, degrees |
| `action` | float32 | 8 | **commanded** clamped IK target, degrees |
| `task` (language) | str | — | e.g. `"pick up the blue cube"` |

fps = 10, episode length = 50 control steps (approach 14 / descend 12 / close 8
/ lift 12 / hold 4). `use_videos=false` (images stored as frames).

## Pre-training verification (passed)

Verified with `synthetic_smolvla/scripts/verify_dense_dataset.py`:

- `LeRobotDataset` loads: **940 episodes / 47,000 frames**, fps 10.
- `ds[0].task = "pick up the blue cube"`, state shape `(8,)`, action shape `(8,)`,
  image `(3, 256, 256)` float32.
- **Frames move within an episode** (max mean abs-diff ~21 across episode 0,
  vs ~0 for the old static-placeholder dataset). State changes by tens of
  degrees across an episode.

## How this fixes the audited gaps

| Old gap (14k audit) | Status in v1 |
|---|---|
| 5 keyframes per episode | Fixed — 50 dense control steps per episode |
| One static image repeated per episode | Fixed — real Isaac camera captured every step (verified non-static) |
| Placeholder renderer, not Isaac camera | Fixed — frames are the actual scene camera tensor |
| Default horizon (chunk_size=50 on 5 frames = padding) | Fixed — 50-step episodes; finetune keeps chunk_size=50, n_action_steps=10 |
| Orange-ball imbalance (610 vs 2222) | Improved — orange 273 of 940, over-sampled to balance |
| No per-frame contact/rise signal | Added — manifest stores object poses, per-object rises, contact steps, limit flag |
| Limit-clamp 99% | Unchanged (~100%). Established as a safety flag in this pocket, not the grasp bottleneck (see 14k audit + README "Optimization attempts"). Revisit via base-pose re-centering only if eval precision needs it. |

## Training

Finetuned from `lerobot/smolvla_base` (loads the pretrained VLM + action
expert), batch 1, 20000 steps, save_freq 5000, `n_action_steps=10`.

- Train config: `synthetic_smolvla/configs/train_dense_isaac_camera_v1.yaml`
- Command: `synthetic_smolvla/reports/train_dense_isaac_camera_v1.sh`
- Checkpoints: `synthetic_smolvla/checkpoints/smolvla_openarm_dense_isaac_camera_v1/checkpoints/`

LeRobot 0.4.4 note: finetuning requires the draccus path directive
`--policy.path=<id>` (the `=` form, as a single token) and `=`-form
`--policy.*` overrides; the space-separated form is rejected. This is handled in
`scripts/train_smolvla.py`.

## Evaluation

Evaluated in Isaac physics with `synthetic_smolvla/scripts/eval_vla_isaac.py`
(real camera, closed loop, 50 steps/trial, contract-clamped actions).

The planned `020000` checkpoint was not available: the accessible training log
stops around step 18,602/20,000 and no
`checkpoints/020000/pretrained_model/model.safetensors` exists. The latest
complete checkpoint, `015000`, was evaluated instead. Accessible logs showed no
Python traceback, CUDA OOM, or system-journal OOM/kill marker; the exact early
stop cause is therefore inconclusive from saved logs.

| Run | Checkpoint | Trials | Success | Success rate | Wrong object | Wrong-object rate |
|---|---|---:|---:|---:|---:|---:|
| Smoke eval with action trace | `015000` | 20 | 10 | 0.500 | 0 | 0.000 |
| Main eval | `015000` | 100 | 66 | 0.660 | 0 | 0.000 |

Artifacts:

- 20-trial JSONL trace: `synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval20_trace.jsonl`
- 20-trial report: `synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval20.md`
- 100-trial JSONL: `synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval100.jsonl`
- 100-trial report: `synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval100.md`

### 100-trial per-target result

| Target | Success | Trials | Success rate | Wrong object lifts |
|---|---:|---:|---:|---:|
| orange_ball | 8 | 25 | 0.320 | 0 |
| red_cube | 20 | 25 | 0.800 | 0 |
| green_cube | 19 | 25 | 0.760 | 0 |
| blue_cube | 19 | 25 | 0.760 | 0 |
| **Total** | **66** | **100** | **0.660** | **0** |

### Eval audit notes

- This is measured closed-loop Isaac lift success, not training loss.
- `015000` produces real target lifts and zero wrong-object lifts in 100 trials.
- Failures are mostly no-lift outcomes. Orange ball remains the weakest target
  (8/25), with repeated zero-rise misses and two negative-rise failures.
- The 20-trial action trace confirms the policy commands and observed joint
  states change over all 50 control steps; this is not a static-output or
  action-unit-dead eval.
- Contract clamp events remain common (`100/100` trials in the 100-run, average
  35.47 events/trial). Eval still applies the same degree-space contract clamp
  as the dataset.

## Notes

- `observation.state` is the measured joint state; `action` is the clamped IK command.
- Frames are the real Isaac scene camera (not the placeholder renderer); they move across the episode.
- Only successful, correct-object lifts are kept; wrong-object lifts are rejected.
- Cubes are 1 inch (0.0254 m); orange ball radius 0.020 m. Simulation only.
