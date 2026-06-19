# Dense Isaac-Camera SmolVLA Dataset (v1) — collection

Source episodes: 2 across 2 envs x 1 rounds.
Kept (measured success, no wrong-object): 0 (0.000).

Each kept episode is a DENSE rollout with real Isaac camera frames at every
control step (100 steps/episode, 256x256 RGB).

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 2 | 1.000 |
| Kept successes | 0 | 0.000 |
| Wrong-object lifts (source) | 0 | 0.000 |
| Limit-clamp episodes (source) | 2 | 1.000 |
| Object-collision episodes (source) | 0 | 0.000 |
| Gripper/table collision episodes (source) | 1 | 0.500 |
| Object sweep/slide episodes (source) | 0 | 0.000 |

## Targets

| Target | Kept | Source |
|---|---:|---:|
| orange_ball | 0 | 0 |
| red_cube | 0 | 0 |
| green_cube | 0 | 0 |
| blue_cube | 0 | 2 |

## Files

- LeRobot dataset root: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_blue_left_m2`
- LeRobot repo id: `local/openarm_real_table_zero_v1_smoke_blue_left_m2`
- Episode metadata JSONL (dense state/action + poses/rises/contact): `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_blue_left_m2_manifest.jsonl`
- Sample frames: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_blue_left_m2_samples`

## Notes

- `observation.state` is the measured joint state; `action` is the clamped IK command.
- Frames are the real Isaac scene camera (not the placeholder renderer); they move across the episode.
- Only successful, correct-object lifts are kept; wrong-object lifts, object collisions, gripper/table collisions, and object sweep/slide episodes are rejected.
- Gripper close command is capped at `-2.000` deg.
