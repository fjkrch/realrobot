# Success-Filtered SmolVLA Dataset

This dataset keeps only measured successful target-object lifts from the parallel IK oracle.

| Metric | Count | Rate |
|---|---:|---:|
| Source episodes | 8 | 1.000 |
| Kept successes | 1 | 0.125 |
| Source wrong-object lifts | 0 | 0.000 |
| Source limit-clamp flags | 7 | 0.875 |

## Kept Successes By Target

| Target | Kept | Source Episodes |
|---|---:|---:|
| blue_cube | 0 | 2 |
| green_cube | 0 | 2 |
| orange_ball | 0 | 2 |
| red_cube | 1 | 2 |

## Files

- Source manifest: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/oracle_parallel_smoke_all.jsonl`
- Success manifest: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/oracle_parallel_smoke_success.jsonl`
- LeRobot root: `/tmp/openarm_parallel_success_smoke`
- LeRobot repo id: `local/openarm_parallel_success_smoke`

## Notes

- Drop limit-exceeded records: `False`.
- RGB frames currently use the deterministic top-down renderer in `dataset_export.py`.
- The parallel oracle measures physics success; this export does not keep failed attempts.
