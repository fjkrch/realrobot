# Success-Filtered SmolVLA Dataset Plan

This plan continues from the physics-measured IK oracle work. The reinforcement
learning route stalled in a reach-only local optimum, so the next path is
success-filtered behavioral cloning:

1. Run the parallel IK oracle in the four-object Isaac scene.
2. Measure real target lift in physics for every episode.
3. Keep only successful target-object lifts.
4. Export the kept trajectories to LeRobot format.
5. Train SmolVLA on the success-only dataset.

No command in this plan moves the physical robot.

## Current Diagnosis

- The open-loop scaffold oracle is not a real pick oracle; it writes
  `success_label: True` without measuring physics.
- The single-env measured IK oracle produced 157 successful target lifts from
  400 randomized four-object episodes, with 0 wrong-object lifts.
- RL reached the object reliably but never learned to grasp/lift, even after
  grasp shaping. Stop using RL for the dataset path.
- Success-filtered behavioral cloning is appropriate because SmolVLA only needs
  successful demonstrations to imitate.

## New Files

- `synthetic_smolvla/scripts/oracle_pick_ik_parallel.py`
  Runs many Isaac environments at once and records measured success labels.
- `synthetic_smolvla/scripts/export_success_filtered_dataset.py`
  Filters successful episodes and exports a LeRobot dataset.
- `synthetic_smolvla/configs/dataset_success_filtered.yaml`
  Dataset output config for the filtered 4000-episode run.
- `synthetic_smolvla/configs/train_success_filtered.yaml`
  SmolVLA training config for the filtered dataset.
- `synthetic_smolvla/configs/dataset_success_filtered_extra10000.yaml`
  Dataset output config for the additional 10000-episode run.
- `synthetic_smolvla/configs/dataset_success_filtered_14000.yaml`
  Dataset output config for the combined 4000+10000 run.
- `synthetic_smolvla/configs/train_success_filtered_14000.yaml`
  SmolVLA training config for the combined success-filtered dataset.

## Smoke Test

Run a tiny parallel oracle job first:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/oracle_pick_ik_parallel.py \
  --num-envs 8 --rounds 1 --device cuda:0 \
  --manifest synthetic_smolvla/reports/oracle_parallel_smoke_all.jsonl \
  --success-manifest synthetic_smolvla/reports/oracle_parallel_smoke_success.jsonl \
  --output synthetic_smolvla/reports/oracle_parallel_smoke_eval.md
```

Export any smoke successes:

```bash
conda run --no-capture-output -n env_isaaclab \
  python synthetic_smolvla/scripts/export_success_filtered_dataset.py \
  --source-manifest synthetic_smolvla/reports/oracle_parallel_smoke_all.jsonl \
  --success-manifest synthetic_smolvla/reports/oracle_parallel_smoke_success.jsonl \
  --dataset-root /tmp/openarm_parallel_success_smoke \
  --repo-id local/openarm_parallel_success_smoke \
  --overwrite \
  --report synthetic_smolvla/reports/oracle_parallel_smoke_dataset.md
```

## Requested 4000-Episode Run

This produces 4000 measured physics episodes as `500 envs x 8 rounds`.

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/oracle_pick_ik_parallel.py \
  --num-envs 500 --rounds 8 --device cuda:0 \
  --manifest synthetic_smolvla/reports/oracle_parallel_all.jsonl \
  --success-manifest synthetic_smolvla/reports/oracle_parallel_success.jsonl \
  --output synthetic_smolvla/reports/oracle_parallel_eval.md
```

Then export the success-filtered VLA dataset:

```bash
conda run --no-capture-output -n env_isaaclab \
  python synthetic_smolvla/scripts/export_success_filtered_dataset.py \
  --dataset-config synthetic_smolvla/configs/dataset_success_filtered.yaml \
  --overwrite \
  --prepare-train \
  --report synthetic_smolvla/reports/success_filtered_dataset.md
```

The filtered dataset root will be:

```text
synthetic_smolvla/datasets/openarm_success_filtered_4000
```

The generated SmolVLA launch script will be:

```text
synthetic_smolvla/reports/train_success_filtered.sh
```

Completed result:

```text
4000 source episodes -> 1799 kept successes (44.975%), 0 wrong-object lifts
```

## Additional 10000-Episode Run

The follow-up run produced another 10000 measured physics episodes as
`500 envs x 20 rounds`.

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/oracle_pick_ik_parallel.py \
  --num-envs 500 --rounds 20 --device cuda:0 \
  --manifest synthetic_smolvla/reports/oracle_parallel_extra10000_all.jsonl \
  --success-manifest synthetic_smolvla/reports/oracle_parallel_extra10000_success.jsonl \
  --output synthetic_smolvla/reports/oracle_parallel_extra10000_eval.md
```

Completed result:

```text
10000 source episodes -> 4481 kept successes (44.81%), 0 wrong-object lifts
```

Exported dataset:

```text
synthetic_smolvla/datasets/openarm_success_filtered_extra10000
local/openarm_success_filtered_extra10000
```

Report:

```text
synthetic_smolvla/reports/success_filtered_extra10000_dataset.md
```

## Combined 14000-Source Dataset

The 4000-source and 10000-source manifests were concatenated into:

```text
synthetic_smolvla/reports/oracle_parallel_combined14000_all.jsonl
synthetic_smolvla/reports/oracle_parallel_combined14000_success.jsonl
```

Then the combined success-filtered dataset was exported with:

```bash
conda run --no-capture-output -n env_isaaclab \
  python synthetic_smolvla/scripts/export_success_filtered_dataset.py \
  --dataset-config synthetic_smolvla/configs/dataset_success_filtered_14000.yaml \
  --overwrite \
  --report synthetic_smolvla/reports/success_filtered_14000_dataset.md
```

Completed result:

```text
14000 source episodes -> 6280 kept successes (44.86%), 0 wrong-object lifts
31400 LeRobot frames at 5 frames/episode
```

Kept successes by target:

| Target | Kept | Source Episodes |
|---|---:|---:|
| blue_cube | 1769 | 3500 |
| green_cube | 1679 | 3500 |
| orange_ball | 610 | 3500 |
| red_cube | 2222 | 3500 |

Combined dataset:

```text
synthetic_smolvla/datasets/openarm_success_filtered_14000
local/openarm_success_filtered_14000
```

Combined training launch script:

```text
synthetic_smolvla/reports/train_success_filtered_14000.sh
```

Validation completed:

```text
LeRobot loads 6280 episodes / 31400 frames.
Feature keys include observation.state, observation.images.camera1, and action.
One-step SmolVLA CUDA smoke training completed successfully.
```

## Training

Run:

```bash
synthetic_smolvla/reports/train_success_filtered_14000.sh
```

This trains local-only SmolVLA on the success-filtered LeRobot dataset and
writes checkpoints under:

```text
synthetic_smolvla/checkpoints/smolvla_openarm_success_filtered_14000
```

## Important Caveat

The current LeRobot exporter uses deterministic top-down RGB renderings derived
from object poses. It is format-ready for SmolVLA and useful for pipeline
testing, but it is not yet real Isaac camera imagery. The Isaac scene itself can
render camera tensors. For a stronger visual policy, the next upgrade is to
record Isaac camera frames at each oracle keyframe and have
`dataset_export.py` use those frames instead of the placeholder renderer.

## Acceptance Targets

- Parallel oracle completes 4000 episodes without real robot access.
- `oracle_parallel_eval.md` reports nonzero measured target lifts and 0
  wrong-object lifts.
- `success_filtered_dataset.md` reports the kept success count by target.
- LeRobot can load `local/openarm_success_filtered_4000`.
- `train_success_filtered.sh` starts SmolVLA CUDA training.
- Additional 10000-episode run completes without real robot access.
- Combined 14000-source dataset exports to
  `local/openarm_success_filtered_14000`.
- `train_success_filtered_14000.sh` starts SmolVLA CUDA training.
