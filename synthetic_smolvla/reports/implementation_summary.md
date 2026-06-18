# Synthetic SmolVLA Implementation Summary

Date: 2026-06-16

## Completed

| Area | Result |
|---|---|
| Isaac scene | Headless OpenArm scene launched and stepped successfully. |
| Camera | Isaac RGB tensor observed with shape `1 x 480 x 640 x 3`. |
| Oracle acceptance | 100 / 100 manifest successes, 0 wrong-object, 0 limit-exceeded. |
| Dataset V1 | 1000 episodes, 5000 frames, 250 episodes per target. |
| Dataset V2 | 5000 episodes, 25000 frames, 1250 episodes per target. |
| V2 all-visible ratio | 1732 / 5000 episodes, 34.6%. |
| Stress test | 1000 / 1000 manifest successes, 0 wrong-object, 0 limit-exceeded. |
| SmolVLA training smoke | V1 and V2 one-step CUDA training completed and saved smoke checkpoints. |

## Key Outputs

- `synthetic_smolvla/reports/isaac_scene_manifest.json`
- `synthetic_smolvla/reports/full_pipeline_status.md`
- `synthetic_smolvla/reports/oracle_acceptance.md`
- `synthetic_smolvla/reports/eval_v1.md`
- `synthetic_smolvla/reports/stress_test_v2.md`
- `synthetic_smolvla/reports/train_v1.sh`
- `synthetic_smolvla/reports/train_v2.sh`

## Remaining Long Jobs

The generated V1 and V2 training commands are verified with one-step smoke runs.
The full 3000-step V1 and 10000-step V2 GPU jobs were not run to completion in
this pass because they are long training jobs. Run:

```bash
synthetic_smolvla/reports/train_v1.sh
synthetic_smolvla/reports/train_v2.sh
```

After those complete, evaluate the learned checkpoints with Isaac policy
rollouts. Current evaluation reports are oracle/manifest-label reports.

## Notes

The exported LeRobot images are deterministic synthetic top-down placeholders
used to prove the dataset and SmolVLA training path. The Isaac scene itself is
live and renders RGB; replacing placeholder export with Isaac camera capture is
the next quality improvement for learned-policy performance.
