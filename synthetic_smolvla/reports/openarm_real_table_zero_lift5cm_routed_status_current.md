# OpenArm Real-Table Zero-Pose 5 cm Status

Date: 2026-06-18

Simulation only. No real robot motion, SSH, CAN, real replay, or mirror sink was
used.

## Active User Constraints

- Render and approve the real-table/floor robot-view scene before data work.
- Use the real-table robot-view camera as close to the real camera as practical.
- Train only from clean episodes:
  - no object-object collision/contact
  - no object sweep, drag, or slide before a valid lift
  - no gripper/finger/table contact or tabletop penetration
  - no wrong-object lift
- Prefer one active arm with the other arm locked/passive. The left arm was
  tested first; it could not lift the right-side targets under the gripper cap.
  Orange/red therefore use the documented simulation-only right-arm fallback.
- Gripper close must not pass `-3 deg`. The user allowed `-2 deg` only if
  `-3 deg` cannot grip. `-2 deg` was tested as a fallback, but it is not better.
- Lift waypoint is only `5 cm`; collection uses `--lift-offset-m 0.05`.
- Use the existing robot joint-limit contract for collection and eval clamps.
- If storage is insufficient, old generated datasets/checkpoints may be deleted.

## Rendered Scene PNGs

- `synthetic_smolvla/reports/real_table_robot_view.png`
- `synthetic_smolvla/reports/real_table_zero_train_camera.png`
- `synthetic_smolvla/reports/real_table_zero_train_reachable_left_camera.png`

## Clean 5 cm Data

The routed 5 cm dataset was collected with object-collision, gripper/table,
sweep/slide, wrong-object, gripper-cap, and joint-limit checks enabled.

- Merged v1 root:
  `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v1`
- Merged v2 extra-right root:
  `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right`
- Start-focused orange/red prefix root:
  `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_start20_v1`

Audit correction from 2026-06-19: the preferred `routed_v2_from020_lr3e5`
checkpoint name is misleading. Its embedded `train_config.json` says it trained
on `openarm_real_table_zero_v1_lift5cm_routed_v1` (469 episodes / 46,900 frames),
not the larger `openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right`
dataset (974 episodes / 97,400 frames). The larger dataset still exists as the
complete clean candidate for future training.

After cleanup, only these dataset roots remain:

- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v1`
- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right`
- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_start20_v1`

## Current Best Checkpoint

Preferred `-3 deg` checkpoint:

```text
synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v2_from020_lr3e5/checkpoints/010000/pretrained_model_typed
```

The old `-2 deg` fallback checkpoint and fallback datasets were deleted in the
2026-06-19 cleanup because exact current-code rechecks failed 0/4. Their reports
remain for audit history.

## Latest Reproducible Evaluation Results

Fresh recreation on 2026-06-19 could not reproduce the archived half-success
reports. The surviving preferred checkpoint now evaluates as 0/4 under both
100-step and 150-step fixed orange/red runs.

Preferred `-3 deg`, 150-step fixed right-side eval recreation:

| Target | Success | Trials | Wrong Object Lifts |
|---|---:|---:|---:|
| orange_ball | 0 | 2 | 0 |
| red_cube | 0 | 2 | 0 |

Report:
`synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_steps150_recreate_20260619.md`

Preferred `-3 deg`, 100-step fixed right-side eval recreation:

| Target | Success | Trials | Wrong Object Lifts |
|---|---:|---:|---:|
| orange_ball | 0 | 2 | 0 |
| red_cube | 0 | 2 | 0 |

Report:
`synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_steps100_recreate_20260619.md`

Half-success recreation audit:
`synthetic_smolvla/reports/openarm_real_table_zero_half_success_recreation_20260619.md`

Archived 2/4 reports remain useful history, but they are not current
reproducible results. The old `-2 deg` fallback checkpoint and fallback datasets
were deleted after exact rechecks failed 0/4.

## Deleted Generated Artifacts

To recover disk space, these failed/generated checkpoint branches were removed:

- `synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_right_fallback_m2_150step_v1_from_m2_002500_lr5e6_002000`
- `synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v3_extra_right_from_v2_010_lr3e5_longsched`
- `synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_right_fallback_m2_150step_v1_from_m2_002500_lr2e6_000500`
- `synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v2_extra_right_from_v2_010_lr1e6_000500`

Their evaluation reports were kept.

The empty zero-success orange-only preferred `-3 deg` 150-step dataset root was
also removed after its report and manifest were written:

- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_orange_right_m3_150step_128`

## Additional Attempts After This Audit Started

Preferred `-3 deg` orange-only 150-step collection:

| Metric | Value |
|---|---:|
| Source episodes | 128 |
| Kept clean lifts | 0 |
| Object collisions | 0 |
| Gripper/table collisions | 0 |
| Object sweep/slide rejects | 0 |

Report:
`synthetic_smolvla/reports/openarm_real_table_zero_v1_lift5cm_orange_right_m3_150step_128_collect.md`

Conclusion: orange does not produce clean 150-step lifts at the preferred
`-3 deg` close cap, even though there are no collision/filter rejects.

Low-LR `-2 deg` fallback fine-tune from the existing fallback base:

```text
source checkpoint: synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_right_fallback_m2_v1_from_v2_010_lr1e5/checkpoints/002500/pretrained_model_typed
dataset: synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_fallback_m2_150step_v1
steps: 500
save_freq: 250
optimizer_lr: 2e-6
scheduler_decay_lr: 1e-6
```

Both eval checkpoints failed fixed orange/red 150-step eval:

| Checkpoint | Success | Trials | Wrong Object Lifts |
|---|---:|---:|---:|
| 000250 | 0 | 4 | 0 |
| 000500 | 0 | 4 | 0 |

Reports:

- `synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_right_fallback_m2_150step_lr2e6_eval_right_fixed_000250_m2_steps150.md`
- `synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_right_fallback_m2_150step_lr2e6_eval_right_fixed_000500_m2_steps150.md`

Conclusion: the shorter lower-LR 150-step fallback fine-tune still regressed to
zero lifts, so it was deleted.

Feature-shape diagnostic:

- Dataset metadata reports `observation.state` shape `[8]`.
- Checkpoint `config.json` reports `observation.state` shape `[6]`.
- The checkpoint normalizer tensors for `observation.state` are shape `(8,)`.
- LeRobot `SmolVLAPolicy.prepare_state` pads the actual received state tensor to
  `max_state_dim`; it does not truncate based on the config shape.
- A temporary state-8 typed wrapper for the preferred `-3 deg` checkpoint was
  evaluated and regressed to 0/4 fixed orange/red 150-step eval.

Report:
`synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_state8_steps150.md`

Conclusion: the stale `[6]` state metadata is an audit issue, but changing the
eval wrapper alone is not a fix and should not be used as the current best model.

Very-low-LR preferred `-3 deg` refresh:

```text
source checkpoint: synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v2_from020_lr3e5/checkpoints/010000/pretrained_model_typed
dataset: synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right
steps: 500
save_freq: 250
optimizer_lr: 1e-6
scheduler_decay_lr: 5e-7
```

Both eval checkpoints failed fixed orange/red 100-step eval:

| Checkpoint | Success | Trials | Wrong Object Lifts |
|---|---:|---:|---:|
| 000250 | 0 | 4 | 0 |
| 000500 | 0 | 4 | 0 |

Reports:

- `synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_extra_right_lr1e6_eval_right_fixed_000250_m3_steps100.md`
- `synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_extra_right_lr1e6_eval_right_fixed_000500_m3_steps100.md`

Conclusion: even a tiny refresh on the extra-right `-3 deg` data regressed to
zero lifts under the matched recheck conditions, so it was deleted.

Reproducibility checks for the archived preferred `-3 deg` checkpoint:

The older report
`synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000.md`
records 2/4 success. With the current eval code/runtime, that result did not
reproduce:

| Recheck | Success | Trials | Wrong Object Lifts |
|---|---:|---:|---:|
| seed 9910, exact poses, 100 steps, 10 substeps | 0 | 4 | 0 |
| default seed/jitter, 100 steps | 0 | 4 | 0 |
| seed 9100, exact poses, 100 steps, 12 substeps | 0 | 4 | 0 |

Reports:

- `synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_seed9910_steps100.md`
- `synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_defaultseed_steps100_recheck.md`
- `synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_exact_recheck_steps100.md`

Inference-frequency diagnostic:

A temporary wrapper with `n_action_steps = 1` was evaluated to force re-observe
and replan every control step. It also failed 0/4.

Report:
`synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_n1_exact_steps100.md`

Conclusion: queued action drift is not the main blocker.

Fallback `-2 deg` reproducibility check:

The archived fallback report
`synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_right_fallback_m2_v1_eval_right_fixed_002500_m2_steps150.md`
records 2/4 success, but the exact current-code recheck failed 0/4.

| Recheck | Success | Trials | Wrong Object Lifts |
|---|---:|---:|---:|
| seed 9100, exact poses, 150 steps, 12 substeps | 0 | 4 | 0 |

Report:
`synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_right_fallback_m2_v1_eval_right_fixed_002500_m2_exact_recheck_steps150.md`

## Current Conclusion

The task is not good yet. The archived preferred `-3 deg` checkpoint has an old
2/4 report, but that result is not reproducible under the current eval code and
runtime. Current reproducible evals are 0/4 for the preferred `-3 deg` checkpoint
on orange/red. The `-2 deg` fallback also has an old 2/4 report, but it now
rechecks at 0/4.

Next safest improvement path: inspect why the policy fails to reproduce the
successful teacher trajectories before spending more training. In particular,
compare the closed-loop action trace against the clean data action/state traces
for orange and verify the SmolVLA state/action feature contract, gripper cap, and
normalizer dimensions. Do not train from the zero-success `-3 deg` 150-step
orange attempt.
