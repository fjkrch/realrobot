# Synthetic SmolVLA IK Oracle Evaluation

Closed-loop differential-IK oracle (approach-above -> descend -> close -> lift).
Success is the MEASURED target rise above 0.040 m in Isaac
physics. Every IK joint solution is clamped to the OpenArm limit contract.

| Metric | Count | Rate |
|---|---:|---:|
| Episodes | 400 | 1.000 |
| Success (real lift) | 157 | 0.393 |
| Wrong object lifted | 0 | 0.000 |
| Limit-clamp engaged | 387 | 0.968 |

## Success By Target

| Target | Success | Episodes |
|---|---:|---:|
| orange_ball | 20 | 100 |
| red_cube | 48 | 100 |
| green_cube | 42 | 100 |
| blue_cube | 47 | 100 |

## Notes

- Camera RGB shape during rollout: `[1, 480, 640, 3]`.
- EE body: `openarm_right_ee_tcp`; position-only DLS IK, 70 iters x 2 substeps per phase.
- Labels are measured from physics, not the hard-coded scaffold label.
