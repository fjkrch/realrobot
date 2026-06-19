# OpenArm Real-Table Zero Half-Success Recreation Audit

Date: 2026-06-19

Simulation only. No real robot, SSH, CAN, mirror sink, or replay action was used.

## Goal

Audit the surviving SmolVLA checkpoint and try to recreate the archived
half-success fixed-scene evaluation result.

## Checkpoint Audited

```text
synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v2_from020_lr3e5/checkpoints/010000/pretrained_model_typed
```

Important correction: this checkpoint name says `routed_v2`, but its embedded
training config says it trained on:

```text
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v1
```

That dataset contains 469 episodes / 46,900 frames. The larger
`openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right` dataset still exists,
but the current audited checkpoint did not train from it.

## Fixed Eval Conditions

- Scene config:
  `synthetic_smolvla/configs/scene_openarm_real_table_zero_train_right_fallback_v1.yaml`
- Active arm: right-arm simulation fallback for orange/red
- Target filter: `orange_ball,red_cube`
- Object poses: fixed real-table row, no jitter
- Seed: `9100`
- Substeps: `12`
- Settle steps: `40`
- Gripper close cap: `-3 deg`
- Camera: real Isaac robot-view camera
- Wrong-object lifts: counted and rejected as failures

## Archived Half-Success Reports

| Archived report | Steps/trial | Result | Target split | Wrong lifts |
|---|---:|---:|---|---:|
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000.jsonl` | 100 | 2/4 | orange 1/2, red 1/2 | 0 |
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_steps150.jsonl` | 150 | 2/4 | orange 0/2, red 2/2 | 0 |

These archived files are real reports, but the score is not currently
reproducible with the same surviving checkpoint and current Isaac/eval runtime.

## Fresh Recreation Results

| Fresh report | Steps/trial | Result | Target split | Wrong lifts |
|---|---:|---:|---|---:|
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_steps150_recreate_20260619.jsonl` | 150 | 0/4 | orange 0/2, red 0/2 | 0 |
| `openarm_real_table_zero_lift5cm_routed_v2_eval_right_fixed_010000_m3_steps100_recreate_20260619.jsonl` | 100 | 0/4 | orange 0/2, red 0/2 | 0 |

Trial details:

| Fresh report | Trial | Target | Success | Rise (m) | Action steps | Clamp events | Wrong lift |
|---|---:|---|---|---:|---:|---:|---|
| steps150 recreation | 0 | orange_ball | false | 0.0000 | 150 | 107 | false |
| steps150 recreation | 1 | red_cube | false | 0.0000 | 150 | 65 | false |
| steps150 recreation | 2 | orange_ball | false | 0.0000 | 150 | 131 | false |
| steps150 recreation | 3 | red_cube | false | 0.0000 | 150 | 57 | false |
| steps100 recreation | 0 | orange_ball | false | 0.0000 | 100 | 79 | false |
| steps100 recreation | 1 | red_cube | false | 0.0000 | 100 | 51 | false |
| steps100 recreation | 2 | orange_ball | false | 0.0000 | 100 | 68 | false |
| steps100 recreation | 3 | red_cube | false | 0.0000 | 100 | 63 | false |

## Conclusion

The archived 2/4 result could not be recreated. The current reproducible score
for the surviving preferred checkpoint is 0/4 under both 100-step and 150-step
fixed orange/red evaluations.

The failure is a no-lift failure, not a wrong-object-lift failure. The model
does not currently produce enough reliable closed-loop grasp/lift behavior from
the zero-pose robot-view start under the `-3 deg` gripper cap.

## Next Useful Step

Do not treat this checkpoint as good. The better next training candidate is a
new run from the retained clean data, especially:

```text
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right
synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_right_start20_v1
```

The start-focused dataset exists because offline diagnostics showed the policy
is weakest at frame 0 / approach, while later lift/hold predictions are closer
to the teacher.
