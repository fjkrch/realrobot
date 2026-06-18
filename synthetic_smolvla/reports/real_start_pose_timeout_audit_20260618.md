# Real Start-Pose Timeout Audit - 2026-06-18

## Observed Failure

Command path: `replay_openarm_trajectory.py --mirror-real --prepare-real-start-pose --read-real-state ...`

The real helper connected and read the right arm state, then aborted during
normal start-pose preparation:

```text
HelperError: Start pose not reached within 25.0s; torque was released by disconnect.
```

No trajectory replay began after this failure.

## Start Pose Target

The configured right-arm start pose is read from
`synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml`:

```text
[0.0, 20.0, 0.0, 55.0, 0.0, 15.0, 0.0, -65.0]
```

## Readback Before Start Move

The reported current state was:

```text
[-10.28369, 0.75406, -0.09836, 22.32686, -0.01093, 13.9338, -15.5075, -64.75118]
```

Initial absolute errors from the configured start pose:

```text
joint_1: 10.28369 deg
joint_2: 19.24594 deg
joint_3: 0.09836 deg
joint_4: 32.67314 deg
joint_5: 0.01093 deg
joint_6: 1.06620 deg
joint_7: 15.50750 deg
gripper: 0.24882 deg
```

The largest starting error was `joint_4` at `32.67314 deg`.

## Likely Cause

The normal helper `prepare_start` path uses LeRobot `max_relative_target` for
safe creep motion. When the real replay command uses `--max-joint-delta-deg 1.0`,
the helper also receives a 1 degree relative target cap. The robot may need more
than the default 25 seconds to settle to the configured start pose under that
cap, especially from a starting state with 30+ degrees of error.

## Fix Applied

- Added laptop CLI flag: `--real-start-pose-timeout-sec`.
- The laptop now passes that timeout to the Jetson helper as
  `--prepare-timeout-sec`.
- The Jetson helper timeout error now includes final `state_deg`, `target_deg`,
  per-joint `errors_deg`, worst joint, max error, and last command sent.
- Deployed the updated helper to the Jetson path used by the mirror.

## Next Safest Test

Run start-pose preflight only first. Keep tolerance strict, but allow more time:

```bash
python3 synthetic_smolvla/scripts/replay_openarm_trajectory.py \
  --trajectory "$UPSAMPLED_TRAJ" \
  --mirror-real \
  --prepare-real-start-pose \
  --read-real-state \
  --real-preflight-only \
  --real-confirm "I am at the robot with e-stop ready" \
  --enable-gripper-real \
  --replay-rate-hz 25.0 \
  --max-joint-delta-deg 1.0 \
  --real-start-pose-timeout-sec 60.0 \
  --watchdog-timeout-sec 2.0 \
  --hold-interval-sec 0.03
```

If this still times out, do not replay the trajectory. Inspect the new error
details to see which joint is not moving or not settling.
