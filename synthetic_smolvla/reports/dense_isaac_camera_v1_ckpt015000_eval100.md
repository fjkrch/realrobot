# SmolVLA Isaac Policy Evaluation

Closed-loop policy rollout in Isaac physics. Simulation only; no real robot motion.

| Metric | Count | Rate |
|---|---:|---:|
| Trials | 100 | 1.000 |
| Success | 66 | 0.660 |
| Wrong object lifted | 0 | 0.000 |

- Average target rise: `0.09376 m`
- Checkpoint: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/checkpoints/smolvla_openarm_dense_isaac_camera_v1/checkpoints/015000/pretrained_model_typed`
- JSONL trials: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/reports/dense_isaac_camera_v1_ckpt015000_eval100.jsonl`

## By Target

| Target | Success | Trials | Success Rate | Wrong Object Lifts |
|---|---:|---:|---:|---:|
| blue_cube | 19 | 25 | 0.760 | 0 |
| green_cube | 19 | 25 | 0.760 | 0 |
| orange_ball | 8 | 25 | 0.320 | 0 |
| red_cube | 20 | 25 | 0.800 | 0 |

## Notes

- Success means the requested target object rose above the lift threshold.
- Wrong-object lift is measured from non-target object rises.
- RGB is the real Isaac scene camera captured per step (default), matching the dense training dataset.
