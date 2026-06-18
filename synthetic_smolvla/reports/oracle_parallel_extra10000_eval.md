# Parallel IK Oracle (4-object, target-conditioned) — measured in physics

Ran 10000 episodes across 500 parallel envs x 20 rounds.
Success = measured target lift; dataset keeps only successful episodes.

| Metric | Count | Rate |
|---|---:|---:|
| Episodes | 10000 | 1.000 |
| Success (kept for VLA) | 4481 | 0.448 |
| Wrong object lifted | 0 | 0.000 |

## Success by target

| Target | Success | Episodes |
|---|---:|---:|
| orange_ball | 436 | 2500 |
| red_cube | 1576 | 2500 |
| green_cube | 1190 | 2500 |
| blue_cube | 1279 | 2500 |

- All manifest: `synthetic_smolvla/reports/oracle_parallel_extra10000_all.jsonl`
- Success-filtered manifest: `synthetic_smolvla/reports/oracle_parallel_extra10000_success.jsonl`
