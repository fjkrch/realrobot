# SmolVLA Training Handoff

Last updated: 2026-06-17

## Safety

This handoff is for simulation data and SmolVLA training only. It does not
authorize real robot motion.

Real robot credentials and hardware access notes stay in
`docs/real/OPENARM_ROBOT_HANDOFF.md`; do not copy those credentials into this
file, chat, commits, or training reports. Real motion still requires a human at
the robot, e-stop ready, and explicit `--i-am-at-robot` style confirmation.

## Task

The current learned task is target-conditioned pick-and-lift:

- Reach the requested object.
- Grasp the requested object.
- Lift the requested object.
- Avoid lifting the wrong object.

This is not yet a place/drop task. The generated success labels measure target
object lift success and wrong-object lift rejection.

## Current Dataset (CORRECTED) — Dense Isaac-Camera v1

This is now the main training dataset. It is the corrected data this handoff
asked for, built by `synthetic_smolvla/scripts/collect_dense_isaac_dataset.py`.

- LeRobot root: `synthetic_smolvla/datasets/openarm_dense_isaac_camera_v1`
- LeRobot repo id: `local/openarm_dense_isaac_camera_v1`
- Dataset report: `synthetic_smolvla/reports/dense_isaac_camera_v1_dataset.md`
- Episode metadata JSONL: `synthetic_smolvla/reports/dense_isaac_camera_v1_manifest.jsonl`

| Metric | Value |
|---|---:|
| Source episodes | 2,080 |
| Kept successes | 940 (45.2%) |
| Frames | 47,000 (50/episode) |
| Wrong-object lifts | 0 |
| Image | 256x256x3 **real Isaac camera** |
| Kept by target | orange 273, red 214, green 278, blue 175 |

What is different from the old dataset (and verified):

- Dense 50-step rollouts (approach/descend/close/lift/hold), not 5 keyframes.
- `observation.state` is the **measured** joint state; `action` is the
  **commanded** clamped IK target, recorded every control step.
- RGB is the **real Isaac scene camera** captured per step. Verified non-static
  with `synthetic_smolvla/scripts/verify_dense_dataset.py` (frames move across
  the episode; the old dataset's frames did not).
- Better target balance (orange over-sampled). Manifest carries per-episode
  object poses, per-object rises, contact steps, and the limit-clamp flag.
- Limit-clamp still ~100% in this pocket (a safety flag, not the grasp
  bottleneck — see the 14k audit and README "Optimization attempts").

Training (this dataset):

- Finetune from `lerobot/smolvla_base`, batch 1, 20000 steps, save_freq 5000,
  `n_action_steps=10`, `chunk_size=50` (kept at the pretrained value).
- Config `synthetic_smolvla/configs/train_dense_isaac_camera_v1.yaml`;
  command `synthetic_smolvla/reports/train_dense_isaac_camera_v1.sh`;
  checkpoints `synthetic_smolvla/checkpoints/smolvla_openarm_dense_isaac_camera_v1/`.
- **LeRobot 0.4.4 fix:** finetuning needs the draccus path directive
  `--policy.path=<id>` (the `=` form, as one token) and `=`-form `--policy.*`
  overrides. The space-separated form is rejected. `scripts/train_smolvla.py`
  now emits the `=` form. A 2-step smoke confirmed it loads the base and trains.

Status: dataset built + verified. The 20,000-step finetune did not produce the
planned `020000` checkpoint; the accessible train log stops around
18,602/20,000 with no Python traceback, CUDA OOM, or system-journal OOM/kill
marker. The latest complete checkpoint, `015000`, has measured Isaac eval
results in the dataset report and the "Isaac Eval Result" section below.

## Old Baseline Dataset (kept as evidence, do not delete)

The previous main dataset, retained for comparison only:

- LeRobot root: `synthetic_smolvla/datasets/openarm_success_filtered_14000`
- LeRobot repo id: `local/openarm_success_filtered_14000`
- Source manifest: `synthetic_smolvla/reports/oracle_parallel_combined14000_all.jsonl`
- Success manifest: `synthetic_smolvla/reports/oracle_parallel_combined14000_success.jsonl`
- Dataset report: `synthetic_smolvla/reports/success_filtered_14000_dataset.md`
- Dataset audit: `synthetic_smolvla/reports/openarm_success_filtered_14000_dataset_audit.md`

Measured contents:

| Metric | Value |
|---|---:|
| Source episodes | 14,000 |
| Kept successes | 6,280 |
| Frames | 31,400 |
| Wrong-object lifts | 0 |
| Success rate | 44.9% |

Kept successes by target:

| Target | Kept |
|---|---:|
| red_cube | 2,222 |
| blue_cube | 1,769 |
| green_cube | 1,679 |
| orange_ball | 610 |

Important caveat: this dataset is structurally valid but not strong enough for
reliable closed-loop physics success. The RGB frames are deterministic top-down
placeholder frames from `synthetic_smolvla/scripts/dataset_export.py`, not
Isaac camera tensors yet, and all five frames in each episode use the same
image. The state and action tensors are the stronger part of this dataset, but
the episodes are still sparse five-keyframe trajectories.

## Verification Already Done

- LeRobot load passed for `local/openarm_success_filtered_14000`.
- Loaded episode count: 6,280.
- Loaded frame count: 31,400.
- State shape: `(8,)`.
- Action shape: `(8,)`.
- Image shape: `(96, 96, 3)`.
- Image key: `observation.images.camera1`.
- One-step SmolVLA CUDA smoke test passed.

Smoke test artifacts:

- Report: `synthetic_smolvla/reports/train_success_filtered_14000_smoke.json`
- Command: `synthetic_smolvla/reports/train_success_filtered_14000_smoke.sh`

## Full Training Run

The full 10,000-step SmolVLA run on the combined success-filtered dataset is
complete.

Completed run:

| Field | Value |
|---|---|
| Status | `completed` |
| Return code | `0` |
| Duration | `1251.41` seconds |
| Batch size | `1` |
| Steps | `10000` |
| Dataset root | `synthetic_smolvla/datasets/openarm_success_filtered_14000` |
| Checkpoint step | `010000` |

The command used was:

```bash
conda run --no-capture-output -n env_isaaclab python synthetic_smolvla/scripts/train_smolvla.py \
  --train-config synthetic_smolvla/configs/train_success_filtered_14000.yaml \
  --batch-size 1 \
  --steps 10000 \
  --run \
  --overwrite-output-dir \
  --output synthetic_smolvla/reports/train_success_filtered_14000_full.json \
  --command-output synthetic_smolvla/reports/train_success_filtered_14000_full.sh
```

Artifacts:

- Run report: `synthetic_smolvla/reports/train_success_filtered_14000_full.json`
- Exact command: `synthetic_smolvla/reports/train_success_filtered_14000_full.sh`
- Checkpoints: `synthetic_smolvla/checkpoints/smolvla_openarm_success_filtered_14000/checkpoints/010000`
- Model weights: `synthetic_smolvla/checkpoints/smolvla_openarm_success_filtered_14000/checkpoints/010000/pretrained_model/model.safetensors`

## If Training Stops

Read the run report first. It includes the return code, duration, stdout tail,
and stderr tail:

```bash
python -m json.tool synthetic_smolvla/reports/train_success_filtered_14000_full.json
```

If the failure is CUDA memory related, keep batch size at 1 and reduce other
memory pressure before trying again. If the checkpoint directory is non-empty
from a partial run, save or rename it before using `--overwrite-output-dir`
again.

## Next Data Upgrade

Do not spend more long training runs on the same five-keyframe placeholder-image
dataset and expect a large Isaac success jump.

The next quality jump is a corrected dataset:

- Dense physics rollouts instead of only five keyframes.
- Real Isaac camera tensors showing robot, gripper, contact, and lift.
- Lower limit-clamp rate in successful oracle demos.
- Better target balance, especially more orange-ball successes.
- Policy horizon matched to the corrected episode structure.

After that, rerun the LeRobot load check and a one-step SmolVLA smoke before
long training.

## Next Agent Checklist

Start here if a new agent is continuing the work.

1. Read the dataset audit:
   `synthetic_smolvla/reports/openarm_success_filtered_14000_dataset_audit.md`.

2. Confirm no old train/eval process is running before using the GPU:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

3. Do not use the current dataset as the main retrain target unless the user
   explicitly wants a quick baseline. It is the wrong quality level for the
   desired closed-loop policy.

4. Implement corrected data collection/export:

- Use dense physics rollouts, not only five oracle keyframes.
- Save actual observed joint state and commanded action at every control step.
- Save Isaac camera RGB frames, not the placeholder renderer.
- Keep language instructions and target IDs.
- Save per-frame or per-episode object pose/rise/contact metadata if the
  LeRobot schema can accept it cleanly.
- Preserve wrong-object-lift rejection.

5. Re-export a small smoke dataset first, then verify:

```bash
conda run --no-capture-output -n env_isaaclab python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = "PATH_TO_NEW_DATASET"
ds = LeRobotDataset(repo_id="local/NEW_REPO_ID", root=root)
print(len(ds), ds[0]["task"], ds[0]["observation.state"].shape, ds[0]["action"].shape)
PY
```

6. Train only after the corrected dataset passes the loader check. Match
   `chunk_size` and `n_action_steps` to the corrected episode/control horizon.

7. Evaluate with Isaac physics using
   `synthetic_smolvla/scripts/eval_vla_isaac.py`. A useful first target is 20
   trials, then 100 trials only after nonzero success appears.

## Isaac Eval Result (Dense v1)

Measured in Isaac physics with `synthetic_smolvla/scripts/eval_vla_isaac.py`
(real camera, closed loop, 50 steps/trial, contract-clamped degree actions).
The planned `020000` checkpoint is missing, so these are results for the latest
complete checkpoint: `checkpoints/015000/pretrained_model`.

| Run | Checkpoint | Trials | Success | Success rate | Wrong object | Wrong-object rate |
|---|---|---:|---:|---:|---:|---:|
| Smoke eval with action trace | `015000` | 20 | 10 | 0.500 | 0 | 0.000 |
| Main eval | `015000` | 100 | 66 | 0.660 | 0 | 0.000 |

100-trial per-target result:

| Target | Success | Trials | Success rate | Wrong object lifts |
|---|---:|---:|---:|---:|
| orange_ball | 8 | 25 | 0.320 | 0 |
| red_cube | 20 | 25 | 0.800 | 0 |
| green_cube | 19 | 25 | 0.760 | 0 |
| blue_cube | 19 | 25 | 0.760 | 0 |
| **Total** | **66** | **100** | **0.660** | **0** |

Artifacts:

- 20-trial JSONL trace: `synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval20_trace.jsonl`
- 20-trial report: `synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval20.md`
- 100-trial JSONL: `synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval100.jsonl`
- 100-trial report: `synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval100.md`

Audit notes:

- This is measured closed-loop Isaac lift success, not training loss.
- `015000` is not broken: it produces real target lifts and zero wrong-object
  lifts in 100 trials.
- Reliability is still incomplete. Most failures are no-lift outcomes; orange
  ball is the main weak target (8/25), with repeated zero-rise misses and two
  negative-rise failures in the 100-run.
- The 20-trial action trace confirms non-static policy commands and changing
  measured joint state through the rollout, so eval is not failing from dead
  actions or a degree/radian mismatch.
- Contract clamp events remain common (`100/100` trials in the 100-run, average
  35.47 events/trial). The same degree-space contract clamp is applied at eval.
- Early training stop cause is inconclusive from saved logs. Accessible evidence:
  the GPU became idle, the train process exited before `020000`, the train log
  ends mid-progress near step 18,602/20,000, `journalctl` shows no OOM/kill
  marker in the relevant window, and `dmesg` was not readable without elevated
  permission.

## Do Not Repeat

- Do not treat low training loss as proof of pickup success.
- Do not run another 10,000-step training job on the unchanged 5-frame static
  placeholder-image dataset as the main fix.
- Do not evaluate against commanded state only; use actual Isaac joint state.
- Do not touch the real robot for this VLA work.
