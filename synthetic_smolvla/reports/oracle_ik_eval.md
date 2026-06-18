# Synthetic SmolVLA IK Oracle Evaluation

Closed-loop differential-IK oracle (approach-above -> descend -> close -> lift).
Success is the MEASURED target rise above 0.040 m in Isaac
physics. Every IK joint solution is clamped to the OpenArm limit contract.

| Metric | Count | Rate |
|---|---:|---:|
| Episodes | 8 | 1.000 |
| Success (real lift) | 5 | 0.625 |
| Wrong object lifted | 0 | 0.000 |
| Limit-clamp engaged | 8 | 1.000 |

## Success By Target

| Target | Success | Episodes |
|---|---:|---:|
| orange_ball | 0 | 2 |
| red_cube | 1 | 2 |
| green_cube | 2 | 2 |
| blue_cube | 2 | 2 |

## Notes

- Camera RGB shape during rollout: `[1, 480, 640, 3]`.
- EE body: `openarm_right_ee_tcp`; position-only DLS IK, 70 iters x 2 substeps per phase.
- Labels are measured from physics, not the hard-coded scaffold label.
