# Dense Isaac-Camera SmolVLA Dataset (v1) — collection

Source episodes: 288 across 8 envs x 36 rounds.
Kept (measured success, no wrong-object): 250 (0.868).

Each kept episode is a DENSE rollout with real Isaac camera frames at every
control step (100 steps/episode, 256x256 RGB).

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 288 | 1.000 |
| Kept successes | 250 | 0.868 |
| Wrong-object lifts (source) | 0 | 0.000 |
| Limit-clamp episodes (source) | 0 | 0.000 |
| Object-collision episodes (source) | 10 | 0.035 |
| Gripper/table collision episodes (source) | 0 | 0.000 |
| Object sweep/slide episodes (source) | 1 | 0.003 |
| Tabletop-penetration episodes (source, finger body) | 0 | 0.000 |
| Object-pushed-down episodes (source) | 0 | 0.000 |
| Refined-action-clip episodes (source) | 0 | 0.000 |

## Targets

| Target | Kept | Source |
|---|---:|---:|
| orange_ball | 0 | 0 |
| red_cube | 0 | 0 |
| green_cube | 250 | 288 |
| blue_cube | 0 | 0 |

## Files

- LeRobot dataset root: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/clean1000_green_left_v1`
- LeRobot repo id: `local/clean1000_green_left_v1`
- Episode metadata JSONL (dense state/action + poses/rises/contact): `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/clean1000_green_left_v1_manifest.jsonl`
- Sample frames: `synthetic_smolvla/reports/clean1000_green_left_v1_samples`

## Notes

- `observation.state` is the measured joint state; `action` is the clamped IK command.
- Frames are the real Isaac scene camera (not the placeholder renderer); they move across the episode.
- Only successful, correct-object lifts are kept; wrong-object lifts, object collisions, gripper/table collisions, and object sweep/slide episodes are rejected.
- Prepose-to-ready is `True` with `120` non-recorded warmup steps.
- Gripper close command is capped at `-3.000` deg.
- Lift waypoint is `0.050` m above the grasp waypoint.
- Grasp z offset is `0.000` m.
