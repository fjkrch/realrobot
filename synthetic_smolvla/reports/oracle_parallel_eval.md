# Parallel IK Oracle (4-object, target-conditioned) — measured in physics

Ran 4000 episodes across 500 parallel envs x 8 rounds.
Success = measured target lift; dataset keeps only successful episodes.

| Metric | Count | Rate |
|---|---:|---:|
| Episodes | 4000 | 1.000 |
| Success (kept for VLA) | 1799 | 0.450 |
| Wrong object lifted | 0 | 0.000 |

## Success by target

| Target | Success | Episodes |
|---|---:|---:|
| orange_ball | 174 | 1000 |
| red_cube | 646 | 1000 |
| green_cube | 489 | 1000 |
| blue_cube | 490 | 1000 |

- All manifest: `synthetic_smolvla/reports/oracle_parallel_all.jsonl`
- Success-filtered manifest: `synthetic_smolvla/reports/oracle_parallel_success.jsonl`
