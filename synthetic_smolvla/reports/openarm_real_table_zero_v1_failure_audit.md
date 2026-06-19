# Real-Table Zero V1 Smoke Failure Audit

Date: 2026-06-18

Simulation only. No real robot motion, SSH, CAN, or real replay was used.

## Attempt

Left arm active, right arm locked/passive, real-table camera scene, 100-frame
episodes, `-3 deg` gripper close cap, object-collision rejection,
gripper/table-collision rejection, and object sweep/slide rejection.

Command:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh /home/chyanin/Desktop/realrobot/synthetic_smolvla/scripts/collect_dense_isaac_dataset.py \
  --config /home/chyanin/Desktop/realrobot/synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --dataset-root /home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke \
  --repo-id local/openarm_real_table_zero_v1_smoke \
  --num-envs 4 \
  --rounds 2 \
  --seed 23000 \
  --target-weights 1.6,1,1,1 \
  --approach-steps 28 \
  --descend-steps 24 \
  --close-steps 16 \
  --lift-steps 24 \
  --hold-steps 8 \
  --substeps 10 \
  --fps 10 \
  --grasp-close-deg -3.0 \
  --max-gripper-close-deg -3.0 \
  --manifest /home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_manifest.jsonl \
  --report /home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_dataset.md \
  --sample-frame-dir /home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_samples \
  --device cuda:0 \
  --overwrite
```

## Terminal Summary

```text
[dense] round 1/2: source=4 kept=0 (round success=0.000)
[dense] round 2/2: source=8 kept=0 (round success=0.000)
source_episodes=8
kept=0
wrong_object=0
limit_clamp=8
object_collision=0
gripper_table_collision=0
object_sweep=0
grasp_close_deg=-3.0
episode_len=100
```

## Manifest Stats

| Metric | Value |
|---|---:|
| Source episodes | 8 |
| Kept episodes | 0 |
| Success labels | 0 |
| Wrong-object lifts | 0 |
| Object collisions | 0 |
| Gripper/table collisions | 0 |
| Object sweep/slide rejects | 0 |
| Max target rise | 0.0 m |
| Episode length | 100 |

Target sampling in this smoke: orange 5, red 1, green 2, blue 0.

## Sample Frames

No sample frames were dumped because the collector only dumps frames for the
first kept episode, and this smoke had zero kept episodes.

## Safest Next Fix

The failure is no-lift with the left arm and `-3 deg` gripper close cap, not a
collision-filter rejection. The user explicitly allowed trying `-2 deg` if
`-3 deg` cannot grip anything. Next action: run the same smoke collection with
`--grasp-close-deg -2.0 --max-gripper-close-deg -2.0` while keeping the left arm,
right-arm lock, joint-limit contract, and all clean-data filters enabled.

If `-2 deg` also has zero kept successes, stop and audit the left-arm reach/IK
layout before considering any fallback.

## Follow-up Attempt: User-Allowed `-2 deg`

After the `-3 deg` no-lift smoke, the same smoke command was rerun with:

```text
--grasp-close-deg -2.0 --max-gripper-close-deg -2.0
```

All other constraints stayed enabled: left arm active, right arm locked/passive,
joint-limit contract clamps, no object collisions, no gripper/table collisions,
and no object sweep/slide episodes.

Terminal summary:

```text
[dense] round 1/2: source=4 kept=0 (round success=0.000)
[dense] round 2/2: source=8 kept=0 (round success=0.000)
source_episodes=8
kept=0
wrong_object=0
limit_clamp=8
object_collision=0
gripper_table_collision=0
object_sweep=0
grasp_close_deg=-2.0
episode_len=100
```

Manifest stats:

| Metric | Value |
|---|---:|
| Source episodes | 8 |
| Kept episodes | 0 |
| Success labels | 0 |
| Wrong-object lifts | 0 |
| Object collisions | 0 |
| Gripper/table collisions | 0 |
| Object sweep/slide rejects | 0 |
| Max target rise | 0.0 m |
| Episode length | 100 |

Target sampling in this smoke: orange 5, red 1, green 2, blue 0.

Representative final action traces show the left arm hitting the left-side safe
joint clamps while still producing zero object rise:

```text
orange_ball final action: [-73.0, 7.0, 41.176, 14.828, -79.801, -32.849, -37.766, -2.0]
red_cube final action:    [-73.0, 7.0, 27.168, 5.611, -6.669, 3.622, -12.783, -2.0]
```

No sample frames were dumped because there were still zero kept episodes.

## Current Blocker

The left-arm real-table zero-pose IK layout is not producing any target lift at
`-3 deg` or at the user-approved `-2 deg` gripper close cap. This is a no-lift
and left-arm reach/IK issue, not a data-filter rejection.

## Audit Gaps Found

- The `-3 deg` raw JSONL manifest was overwritten by the later `-2 deg` fallback
  run because both smoke commands wrote to the same manifest path. The terminal
  summary and stats are preserved above, but future fallback attempts should use
  distinct manifest/report paths.
- The collector previously saved sample frames only after a kept episode. That
  meant zero-success smoke failures had no sample frame paths. The collector has
  been patched so future smoke runs dump the first source episode frames even if
  the episode is rejected.

Do not full-collect, train, or evaluate a model from this zero-success smoke.

Safest next options:

1. Diagnose left-arm reach/IK for the real-table object row and adjust only if it
   remains faithful to the real setup and robot-safe joint limits.
2. If left arm cannot produce clean successful lifts, document that outcome and
   ask/confirm before falling back to the previously working right-arm pipeline.

## New Constraint: 5 cm Lift Only

The user later added: lift only `5 cm`. Collection must use:

```text
--lift-offset-m 0.05
```

Do not train on class-routed datasets that were collected before this constraint
with the older 15 cm lift waypoint. Treat these existing roots as audit-only:

- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_orange_right_m3`
- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_red_right_m3`
- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_green_left_m3`
- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_green_left_zp01_m3`
- `synthetic_smolvla/datasets/openarm_real_table_zero_v1_routed_blue_left_m3`

Keep all other data constraints unchanged: no object-object collision/contact,
no sweep/drag/large table slide before lift, no gripper/table contact, no
wrong-object lift, active arm only with the other arm locked/passive, robot-safe
joint clamps, and gripper close no tighter than `-3 deg` unless a documented
`-2 deg` fallback is explicitly needed.
