# Dense Isaac-Camera SmolVLA Dataset (v1) — collection

Source episodes: 4 across 4 envs x 1 rounds.
Kept (measured success, no wrong-object): 1 (0.250).

Each kept episode is a DENSE rollout with real Isaac camera frames at every
control step (100 steps/episode, 256x256 RGB).

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 4 | 1.000 |
| Kept successes | 1 | 0.250 |
| Wrong-object lifts (source) | 0 | 0.000 |
| Limit-clamp episodes (source) | 4 | 1.000 |
| Object-collision episodes (source) | 0 | 0.000 |
| Gripper/table collision episodes (source) | 0 | 0.000 |
| Object sweep/slide episodes (source) | 3 | 0.750 |

## Targets

| Target | Kept | Source |
|---|---:|---:|
| orange_ball | 0 | 0 |
| red_cube | 0 | 0 |
| green_cube | 0 | 1 |
| blue_cube | 1 | 3 |

## Files

- LeRobot dataset root: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke_reachable_left_m3`
- LeRobot repo id: `local/openarm_real_table_zero_v1_smoke_reachable_left_m3`
- Episode metadata JSONL (dense state/action + poses/rises/contact): `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_reachable_left_m3_manifest.jsonl`
- Sample frames: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_reachable_left_m3_samples`

## Notes

- `observation.state` is the measured joint state; `action` is the clamped IK command.
- Frames are the real Isaac scene camera (not the placeholder renderer); they move across the episode.
- Only successful, correct-object lifts are kept; wrong-object lifts, object collisions, gripper/table collisions, and object sweep/slide episodes are rejected.
- Gripper close command is capped at `-3.000` deg.
