# SmolVLA Isaac Policy Evaluation

Closed-loop policy rollout in Isaac physics. Simulation only; no real robot motion.

| Metric | Count | Rate |
|---|---:|---:|
| Trials | 4 | 1.000 |
| Success | 0 | 0.000 |
| Wrong object lifted | 0 | 0.000 |

- Average target rise: `0.00000 m`
- Checkpoint: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_right_fallback_m2_150step_v1_from_m2_002500_lr2e6_000500/checkpoints/000250/pretrained_model_typed`
- JSONL trials: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_right_fallback_m2_150step_lr2e6_eval_right_fixed_000250_m2_steps150.jsonl`

## By Target

| Target | Success | Trials | Success Rate | Wrong Object Lifts |
|---|---:|---:|---:|---:|
| orange_ball | 0 | 2 | 0.000 | 0 |
| red_cube | 0 | 2 | 0.000 | 0 |

## Notes

- Success means the requested target object rose above the lift threshold.
- Wrong-object lift is measured from non-target object rises.
- RGB is the real Isaac scene camera captured per step (default), matching the dense training dataset.
- Policy gripper commands are capped by `--max-gripper-close-deg` before simulation.
