# SmolVLA Isaac Policy Evaluation

Closed-loop policy rollout in Isaac physics. Simulation only; no real robot motion.

| Metric | Count | Rate |
|---|---:|---:|
| Trials | 4 | 1.000 |
| Success | 0 | 0.000 |
| Wrong object lifted | 0 | 0.000 |

- Average target rise: `0.00000 m`
- Checkpoint: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_v2_extra_right_plus_start20_1500_direct_from010_state8_lr3e5/checkpoints/006000/pretrained_model_typed`
- JSONL trials: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_1500_direct_eval_ckpt006000_m3_steps150.jsonl`

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
