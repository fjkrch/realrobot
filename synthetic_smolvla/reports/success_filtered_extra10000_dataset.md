# Success-Filtered SmolVLA Dataset

This dataset keeps only measured successful target-object lifts from the parallel IK oracle.

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 10000 | 1.000 |
| Kept successes | 4481 | 0.448 |
| Source wrong-object lifts | 0 | 0.000 |
| Source limit-clamp flags | 9912 | 0.991 |

## Kept Successes By Target

| Target | Kept | Source Episodes |
|---|---:|---:|
| blue_cube | 1279 | 2500 |
| green_cube | 1190 | 2500 |
| orange_ball | 436 | 2500 |
| red_cube | 1576 | 2500 |

## Files

- Source manifest: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/oracle_parallel_extra10000_all.jsonl`
- Success manifest: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/oracle_parallel_extra10000_success.jsonl`
- LeRobot root: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_success_filtered_extra10000`
- LeRobot repo id: `local/openarm_success_filtered_extra10000`

## Notes

- Drop limit-exceeded records: `False`.
- RGB frames currently use the deterministic top-down renderer in `dataset_export.py`.
- The parallel oracle measures physics success; this export does not keep failed attempts.
