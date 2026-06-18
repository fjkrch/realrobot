# Parallel IK Oracle (4-object, target-conditioned) — measured in physics

Ran 8 episodes across 8 parallel envs x 1 rounds.
Success = measured target lift; dataset keeps only successful episodes.

| Metric | Count | Rate |
|---|---:|---:|
| Episodes | 8 | 1.000 |
| Success (kept for VLA) | 1 | 0.125 |
| Wrong object lifted | 0 | 0.000 |

## Success by target

| Target | Success | Episodes |
|---|---:|---:|
| orange_ball | 0 | 2 |
| red_cube | 1 | 2 |
| green_cube | 0 | 2 |
| blue_cube | 0 | 2 |

- All manifest: `synthetic_smolvla/reports/oracle_parallel_smoke_all.jsonl`
- Success-filtered manifest: `synthetic_smolvla/reports/oracle_parallel_smoke_success.jsonl`
