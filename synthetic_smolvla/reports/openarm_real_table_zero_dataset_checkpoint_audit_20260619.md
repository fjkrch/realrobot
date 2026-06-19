# OpenArm Real-Table Zero Dataset/Checkpoint Audit

Date: 2026-06-19

Simulation only. No real robot, SSH, CAN, mirror sink, or replay action was used.

## Active Process Check

No active `slice_lerobot_prefix_dataset.py`, `lerobot_train`, `eval_vla_isaac.py`,
or `collect_dense_isaac_dataset.py` process was found during this audit.

## Main Correction

The preferred checkpoint name includes `routed_v2`, but its embedded
`train_config.json` says it trained on:

```text
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v1
```

That dataset has:

| Dataset | Episodes | Frames |
|---|---:|---:|
| `openarm_real_table_zero_v1_lift5cm_routed_v1` | 469 | 46,900 |

The larger dataset:

```text
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right
```

exists and is clean/merged, but it is not the dataset embedded in the current
preferred checkpoint's training config.

| Dataset | Episodes | Frames | Status |
|---|---:|---:|---|
| `openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right` | 974 | 97,400 | Current larger candidate dataset; not embedded in preferred checkpoint |
| `openarm_real_table_zero_v1_lift5cm_right_start20_v1` | 760 | 15,200 | New start-focused derived dataset; not trained yet |

## Current Preferred Checkpoint

```text
synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v2_from020_lr3e5/checkpoints/010000/pretrained_model_typed
```

Embedded training config:

| Field | Value |
|---|---|
| Dataset repo | `local/openarm_real_table_zero_v1_lift5cm_routed_v1` |
| Dataset root | `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v1` |
| Pretrained path | `.../smolvla_openarm_real_table_zero_lift5cm_routed_v1/checkpoints/020000/pretrained_model_typed` |
| Steps | 30,000 |
| Batch size | 1 |
| `n_action_steps` | 10 |
| Config state shape | `[6]` |
| Dataset state/action shape | `[8]` / `[8]` |

Note: the normalizer tensors are 8-D, and previous diagnostics showed LeRobot
pads the actual state tensor rather than truncating it based on the stale
`[6]` config metadata.

## Half-Correct Eval Reports

These are the real-table zero VLA reports that reached about half correct:

| Report | Score | Target Split | Reproducible Now |
|---|---:|---|---|
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000.jsonl` | 2/4 | orange 1/2, red 1/2 | No, current rechecks are 0/4 |
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_steps150.jsonl` | 2/4 | orange 0/2, red 2/2 | No, current rechecks are 0/4 |
| `openarm_real_table_zero_lift5cm_right_fallback_m2_v1_eval_right_fixed_002500_m2_steps150.jsonl` | 2/4 | orange 1/2, red 1/2 | No, exact recheck is 0/4 |
| `openarm_real_table_zero_lift5cm_routed_v3_extra_right_eval_right_fixed_005000.jsonl` | 2/4 | orange 1/2, red 1/2 | Checkpoint branch was deleted after later failures |

Current reproducible preferred-checkpoint rechecks are 0/4:

| Report | Score |
|---|---:|
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_seed9910_steps100.jsonl` | 0/4 |
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_defaultseed_steps100_recheck.jsonl` | 0/4 |
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_exact_recheck_steps100.jsonl` | 0/4 |
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_n1_exact_steps100.jsonl` | 0/4 |
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_state8_steps150.jsonl` | 0/4 |

Fallback `-2 deg` exact recheck:

| Report | Score |
|---|---:|
| `openarm_real_table_zero_lift5cm_right_fallback_m2_v1_eval_right_fixed_002500_m2_exact_recheck_steps150.jsonl` | 0/4 |

## Offline Policy-vs-Teacher Diagnostic

Report:

```text
synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_policy_teacher_diag_orange_red_v2_010000.md
```

Main result:

| Metric | Value |
|---|---:|
| Records | 968 |
| Mean MAE | 2.9782 deg |
| P90 MAE | 7.8948 deg |
| Arm-limit violation rows | 185 |
| Gripper-too-closed rows | 151 |

Failure shape:

- Start/frame-0 and approach predictions are weak.
- Later lift/hold predictions are much closer to the teacher.
- The closed-loop policy likely fails before it reaches a useful grasp/lift state.

## Keep Set

Do not delete these unless the current plan changes:

| Path | Reason |
|---|---|
| `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v1` | Actual dataset embedded in preferred checkpoint train config |
| `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right` | Current larger clean candidate dataset |
| `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_start20_v1` | New start-focused derived dataset for next training |
| `synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v2_from020_lr3e5` | Current preferred checkpoint family |

Optional keep:

| Path | Reason |
|---|---|
| `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_fallback_m2_v1` | Old fallback `-2 deg` data; old half-correct report, current recheck failed |
| `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_fallback_m2_150step_v1` | Old fallback `-2 deg` 150-step data; later training failed |
| `synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_right_fallback_m2_v1_from_v2_010_lr1e5` | Old fallback checkpoint; current exact recheck failed |

## Dataset Delete Candidates

These are stale, duplicate source, smoke, old placeholder, or non-current-version
datasets. They can be deleted after explicit confirmation:

```text
synthetic_smolvla/datasets/openarm_dense_isaac_camera_smoke
synthetic_smolvla/datasets/openarm_dense_isaac_camera_v1
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_blue_left_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_green_left_zp01_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_orange_right_extra_m3_256
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_orange_right_fallback_m2_128
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_orange_right_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_red_right_extra_m3_256
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_red_right_fallback_m2_128
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_red_right_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_smoke_green_left_zp01_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_blue_left_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_green_left_center_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_green_left_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_green_left_zp01_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_orange_right_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_red_right_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_blue_left_m2
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_left_green_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_left_green_zp01_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_left_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_left_orange_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_left_red_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_right_green_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_right_orange_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_right_red_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_right_m3
synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_routed_merge
synthetic_smolvla/datasets/openarm_success_filtered_14000
```

Optional delete if abandoning the `-2 deg` fallback:

```text
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_fallback_m2_v1
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_fallback_m2_150step_v1
synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_right_fallback_m2_v1_from_v2_010_lr1e5
```

## Checkpoint Delete Candidates

These are old non-current checkpoint families:

```text
synthetic_smolvla/checkpoints/smolvla_openarm_dense_isaac_camera_v1
synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v1
synthetic_smolvla/checkpoints/smolvla_openarm_success_filtered_14000
synthetic_smolvla/checkpoints/smolvla_openarm_success_filtered_extra10000
synthetic_smolvla/checkpoints/smolvla_openarm_synth_v1
synthetic_smolvla/checkpoints/smolvla_openarm_synth_v1_smoke
synthetic_smolvla/checkpoints/smolvla_openarm_synth_v2_smoke
```

Note: deleting `smolvla_openarm_real_table_zero_lift5cm_routed_v1` removes the
source checkpoint family used to initialize the current preferred checkpoint, but
the current preferred checkpoint itself remains loadable.

## Storage

Current free space at audit time:

```text
6.8G free on /home/chyanin/Desktop/realrobot
```

Largest old checkpoint candidates:

| Path | Size |
|---|---:|
| `synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v1` | 6.2G |
| `synthetic_smolvla/checkpoints/smolvla_openarm_dense_isaac_camera_v1` | 3.7G |
| `synthetic_smolvla/checkpoints/smolvla_openarm_success_filtered_14000` | 1.5G |
| `synthetic_smolvla/checkpoints/smolvla_openarm_synth_v1` | 1.5G |
| `synthetic_smolvla/checkpoints/smolvla_openarm_synth_v1_smoke` | 1.5G |
| `synthetic_smolvla/checkpoints/smolvla_openarm_synth_v2_smoke` | 1.5G |

