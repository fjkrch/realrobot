# SmolVLA Isaac To Real OpenArm Mirror

This path lets the typed interactive SmolVLA Isaac demo optionally mirror
clamped arm joint targets to the real OpenArm. Default behavior is still
simulation only.

## Safety Contract

- No real robot access happens unless a real flag is passed.
- Any real motion requires both flags:
  - `--mirror-real`
  - `--real-confirm "I am at the robot with e-stop ready"`
- Preparing the real start pose also requires `--prepare-real-start-pose`.
- Real gripper motion is disabled by default through `--disable-gripper-real`.
- The real helper uses `connect(calibrate=False)` and a fresh non-calibration id.
- The mirror clamps to the sim contract, validates live robot limits, rate-limits
  targets, enforces `--max-joint-delta-deg`, and has a watchdog timeout.

Stop/abort:

```bash
# On the laptop, Ctrl-C the interactive process.
# On the Jetson, release torque if needed:
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python scripts/disable_torque.py --port can0 --side right
```

If the helper is still running, stop it first:

```bash
pkill -f openarm_safe_real_mirror.py
python scripts/disable_torque.py --port can0 --side right
```

## Deploy Helper

Copy the guarded helper to the Jetson once:

```bash
cd /home/chyanin/Desktop/realrobot
scp scripts/openarm_safe_real_mirror.py arms@10.10.10.2:/home/arms/hsi-pre-grasp/scripts/
```

The watchdog self-test does not touch CAN or motors:

```bash
python3 scripts/openarm_safe_real_mirror.py --self-test-watchdog
```

## Sim Only

Headless is recommended on the 8 GB GPU:

```bash
cd /home/chyanin/Desktop/realrobot
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/interactive_vla_isaac.py \
  --headless
```

Type a task such as `pick up the red cube`, then `q` to quit.
The default camera resolution comes from the scene config.

## Dry-Run Mirror

This logs exactly what would be mirrored, but does not contact the real robot:

```bash
cd /home/chyanin/Desktop/realrobot
TRACE=synthetic_smolvla/reports/interactive_vla_mirror_dry_run_red_trace.jsonl
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/interactive_vla_isaac.py \
  --headless \
  --mirror-dry-run "$TRACE"
```

Audit the trace before any real test:

```bash
python3 synthetic_smolvla/scripts/audit_mirror_trace.py \
  --trace "$TRACE" \
  --side right \
  --max-joint-delta-deg 3.0 \
  --expected-steps 50 \
  --start-pose-config synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml \
  --first-target-tolerance-deg 15.0 \
  --output-md synthetic_smolvla/reports/interactive_vla_mirror_dry_run_audit.md
```

## Read Real State

Only do this while physically at the robot with e-stop ready and motor power in
the intended state:

```bash
cd /home/chyanin/Desktop/realrobot
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/interactive_vla_isaac.py \
  --read-real-state \
  --real-confirm "I am at the robot with e-stop ready"
```

This reads arm joints plus gripper and exits. It does not send motion targets.

## Prepare Real Start Pose

This moves the real arm to the same/symmetric reset pose in
`synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml`:

```bash
cd /home/chyanin/Desktop/realrobot
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/interactive_vla_isaac.py \
  --headless \
  --mirror-real \
  --prepare-real-start-pose \
  --read-real-state \
  --real-preflight-only \
  --real-confirm "I am at the robot with e-stop ready" \
  --mirror-rate-hz 2.0 \
  --max-joint-delta-deg 3.0 \
  --watchdog-timeout-sec 2.0
```

The script reads the current real state, moves the arm to the reset pose, reads
back the state again, audits the error, then exits before typed tasks. Gripper
commands are not sent unless `--enable-gripper-real` is explicitly added.

Because this mode exits, it releases torque after the audit. Use the full
supervised mirror command below when you want the arm to hold the start pose
while waiting for a typed task.

## Later Human-Supervised Mirror Test

Only after dry-run audit and start-pose preflight pass:

```bash
cd /home/chyanin/Desktop/realrobot
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/interactive_vla_isaac.py \
  --headless \
  --mirror-dry-run synthetic_smolvla/reports/interactive_vla_real_shadow_trace.jsonl \
  --mirror-real \
  --prepare-real-start-pose \
  --read-real-state \
  --real-confirm "I am at the robot with e-stop ready" \
  --mirror-rate-hz 2.0 \
  --max-joint-delta-deg 3.0 \
  --watchdog-timeout-sec 2.0
```

At the prompt, type one simple task, watch the robot continuously, and stop
immediately if anything looks wrong.

After the start-pose preflight passes, the real helper sends hold heartbeats so
the arm keeps holding the start pose while the prompt waits for your typed task.
After a task finishes, the script holds the final pose and refuses new tasks
until you type `reset`. `reset` moves the real arm back to the configured start
pose and resets the sim scene. Type `hold` to keep holding the current pose, or
`q` only when you are ready to release/exit.

## Sim-First Then Real Replay

Use this safer mode when you want Isaac to run the pick first, save the command
trajectory, and only then replay it on the real arm if the sim outcome is a clean
success:

```bash
cd /home/chyanin/Desktop/realrobot
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/interactive_vla_isaac.py \
  --viewer \
  --policy-device cpu \
  --mirror-dry-run synthetic_smolvla/reports/interactive_vla_real_shadow_trace.jsonl \
  --mirror-real \
  --real-replay-after-sim-success \
  --prepare-real-start-pose \
  --read-real-state \
  --real-confirm "I am at the robot with e-stop ready" \
  --mirror-rate-hz 2.0 \
  --max-joint-delta-deg 3.0 \
  --watchdog-timeout-sec 2.0 \
  --hold-interval-sec 0.2
```

In this mode:

- The real arm moves to the configured start pose and holds.
- The typed task runs in Isaac only.
- The command trajectory is saved under
  `synthetic_smolvla/reports/interactive_vla_saved_trajectories/`.
- If sim succeeds and no wrong object is lifted, the script asks for one more
  operator confirmation. Type exactly `RUN` to re-prepare the real arm to start,
  replay the saved trajectory, then hold the final pose.
- If sim fails, real replay is skipped and the arm keeps holding its current
  commanded pose.
- After the sim task or real replay finishes, the script refuses a new task until
  you type `reset`.

## Real Mirror Preflight Checklist

- Robot connection reachable through the guarded helper.
- Current real state read back as 8 values.
- Start pose loaded from the scene config and clamped.
- Real arm moved to start pose only after the exact confirmation phrase.
- Readback error is within tolerance before typed tasks begin.
- First policy target is checked against current real state.
- Larger target changes are split into `--max-joint-delta-deg` intermediate
  targets at `--mirror-rate-hz`.
- Watchdog timeout releases torque by disconnecting the helper.
- Logs contain joint values and status only; do not paste credentials into logs.
