# Success-Filtered SmolVLA Dataset

This dataset keeps only measured successful target-object lifts from the parallel IK oracle.

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 4000 | 1.000 |
| Kept successes | 1799 | 0.450 |
| Source wrong-object lifts | 0 | 0.000 |
| Source limit-clamp flags | 3956 | 0.989 |

## Kept Successes By Target

| Target | Kept | Source Episodes |
|---|---:|---:|
| blue_cube | 490 | 1000 |
| green_cube | 489 | 1000 |
| orange_ball | 174 | 1000 |
| red_cube | 646 | 1000 |

## Files

- Source manifest: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/oracle_parallel_all.jsonl`
- Success manifest: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/oracle_parallel_success.jsonl`
- LeRobot root: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_success_filtered_4000`
- LeRobot repo id: `local/openarm_success_filtered_4000`

## Notes

- Drop limit-exceeded records: `False`.
- RGB frames currently use the deterministic top-down renderer in `dataset_export.py`.
- The parallel oracle measures physics success; this export does not keep failed attempts.
