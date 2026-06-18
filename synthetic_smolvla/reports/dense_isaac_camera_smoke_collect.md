# Dense Isaac-Camera SmolVLA Dataset (v1) — collection

Source episodes: 8 across 8 envs x 1 rounds.
Kept (measured success, no wrong-object): 3 (0.375).

Each kept episode is a DENSE rollout with real Isaac camera frames at every
control step (50 steps/episode, 256x256 RGB).

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 8 | 1.000 |
| Kept successes | 3 | 0.375 |
| Wrong-object lifts (source) | 0 | 0.000 |
| Limit-clamp episodes (source) | 8 | 1.000 |

## Targets

| Target | Kept | Source |
|---|---:|---:|
| orange_ball | 2 | 4 |
| red_cube | 0 | 0 |
| green_cube | 0 | 2 |
| blue_cube | 1 | 2 |

## Files

- LeRobot dataset root: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_dense_isaac_camera_smoke`
- LeRobot repo id: `local/openarm_dense_isaac_camera_smoke`
- Episode metadata JSONL (dense state/action + poses/rises/contact): `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/dense_isaac_camera_smoke_manifest.jsonl`
- Sample frames: `synthetic_smolvla/reports/dense_isaac_camera_smoke_samples`

## Notes

- `observation.state` is the measured joint state; `action` is the clamped IK command.
- Frames are the real Isaac scene camera (not the placeholder renderer); they move across the episode.
- Only successful, correct-object lifts are kept; wrong-object lifts are rejected.
