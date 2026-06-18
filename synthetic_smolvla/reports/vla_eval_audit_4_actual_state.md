# SmolVLA Isaac Policy Evaluation

Closed-loop policy rollout in Isaac physics. Simulation only; no real robot motion.

| Metric | Count | Rate |
|---|---:|---:|
| Trials | 4 | 1.000 |
| Success | 0 | 0.000 |
| Wrong object lifted | 0 | 0.000 |

- Average target rise: `0.00000 m`
- Checkpoint: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/checkpoints/smolvla_openarm_success_filtered_14000/checkpoints/010000/pretrained_model_typed`
- JSONL trials: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/vla_eval_audit_4_actual_state.jsonl`

## By Target

| Target | Success | Trials | Success Rate | Wrong Object Lifts |
|---|---:|---:|---:|---:|
| blue_cube | 0 | 1 | 0.000 | 0 |
| green_cube | 0 | 1 | 0.000 | 0 |
| orange_ball | 0 | 1 | 0.000 | 0 |
| red_cube | 0 | 1 | 0.000 | 0 |

## Notes

- Success means the requested target object rose above the lift threshold.
- Wrong-object lift is measured from non-target object rises.
- RGB is still the deterministic top-down renderer used by the training dataset, not Isaac camera tensors.
