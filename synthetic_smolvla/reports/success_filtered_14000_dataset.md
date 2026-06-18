# Success-Filtered SmolVLA Dataset

This dataset keeps only measured successful target-object lifts from the parallel IK oracle.

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 14000 | 1.000 |
| Kept successes | 6280 | 0.449 |
| Source wrong-object lifts | 0 | 0.000 |
| Source limit-clamp flags | 13868 | 0.991 |

## Kept Successes By Target

| Target | Kept | Source Episodes |
|---|---:|---:|
| blue_cube | 1769 | 3500 |
| green_cube | 1679 | 3500 |
| orange_ball | 610 | 3500 |
| red_cube | 2222 | 3500 |

## Files

- Source manifest: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/oracle_parallel_combined14000_all.jsonl`
- Success manifest: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/oracle_parallel_combined14000_success.jsonl`
- LeRobot root: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_success_filtered_14000`
- LeRobot repo id: `local/openarm_success_filtered_14000`

## Notes

- Drop limit-exceeded records: `False`.
- RGB frames currently use the deterministic top-down renderer in `dataset_export.py`.
- The parallel oracle measures physics success; this export does not keep failed attempts.
