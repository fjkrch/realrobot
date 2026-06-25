# Real Robot README

Use this folder when a task may move physical OpenArm motors.

## Required First Reads

1. [REAL_ROBOT_MOVE.md](REAL_ROBOT_MOVE.md)
2. [../reference/OPENARM_JOINT_LIMITS.md](../reference/OPENARM_JOINT_LIMITS.md)
3. [real_robot_calibration.md](real_robot_calibration.md)

Use `OPENARM_ROBOT_HANDOFF.md` only if you need the private password. Do not
paste credential content into chat or commits.

## Real Motion Rules

- Person physically at robot.
- E-stop reachable.
- 24 V motor power on.
- CAN up and tested.
- Commands include the script-specific acknowledgement:
  `--i-am-at-robot`, or the exact `--real-confirm` phrase for the guarded
  SmolVLA mirror.
- Avoid calibration paths. Current scripts use `connect(calibrate=False)`.

## Real Motion Scripts

| Task | Script | Doc |
|---|---|---|
| Gripper | Jetson-side `scripts/open_gripper_small.py` | [REAL_ROBOT_MOVE.md](REAL_ROBOT_MOVE.md) |
| One joint | [../../scripts/move_joint.py](../../scripts/move_joint.py) | [REAL_ROBOT_MOVE.md](REAL_ROBOT_MOVE.md) |
| All 7 joints of one arm | [../../scripts/move_arm.py](../../scripts/move_arm.py) | [REAL_ROBOT_MOVE.md](REAL_ROBOT_MOVE.md) |
| Guarded SmolVLA Isaac mirror | [../../scripts/openarm_safe_real_mirror.py](../../scripts/openarm_safe_real_mirror.py) | [SMOLVLA_SIM_TO_REAL_MIRROR.md](SMOLVLA_SIM_TO_REAL_MIRROR.md) |
| Saved upsampled height episode replay | [../../scripts/replay_openarm_saved_episode_real.py](../../scripts/replay_openarm_saved_episode_real.py) | This README |
| Saved episode replay launcher | [../../run_openarm_saved_episode_replay.txt](../../run_openarm_saved_episode_replay.txt) | This README |
| Teach / replay cube pick | [../../scripts/pick_cube.py](../../scripts/pick_cube.py) | [PICK_CUBE_SIM_TO_REAL.md](PICK_CUBE_SIM_TO_REAL.md) |
| Release torque | Jetson-side `scripts/disable_torque.py` | [REAL_ROBOT_MOVE.md](REAL_ROBOT_MOVE.md) |

## Before Real Motion

Copy laptop-authored scripts to Jetson if needed:

```bash
cd /home/chyanin/Desktop/realrobot
scp scripts/move_joint.py scripts/move_arm.py scripts/pick_cube.py arms@10.10.10.2:/home/arms/hsi-pre-grasp/scripts/
```

On Jetson:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
sudo ./scripts/can_up.sh
```

Then follow [REAL_ROBOT_MOVE.md](REAL_ROBOT_MOVE.md).

## Saved Height Episode Replay

This replays saved upsampled NPZ episodes for one selected table height. It
reads only the `action` key, refuses non-upsampled paths, refuses commands
outside real joint limits, and sends exactly one saved command at the dataset
rate. The original `10hz` family uses `2 deg/command` at `0.1 s`; the lower
`20hz400` family uses `1.5 deg/command` at `0.05 s`.

The first saved command is zero pose. The runner does not move the robot to
zero; the operator must place the real arm close to the first saved command.
If the live readback is not close, real replay is refused.

Deploy the runner and only the upsampled height folders to the Jetson:

```bash
cd /home/chyanin/Desktop/realrobot
JETSON=arms@192.168.31.50

ssh "$JETSON" 'mkdir -p /home/arms/hsi-pre-grasp/synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height'
scp scripts/replay_openarm_saved_episode_real.py "$JETSON":/home/arms/hsi-pre-grasp/scripts/
scp run_openarm_saved_episode_replay.txt "$JETSON":/home/arms/hsi-pre-grasp/
ssh "$JETSON" 'mkdir -p /home/arms/hsi-pre-grasp/docs/real'
scp docs/real/run_openarm_saved_episode_replay.txt "$JETSON":/home/arms/hsi-pre-grasp/docs/real/
scp -r \
  synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height/h125cm_upsampled \
  synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height/h122p5cm_upsampled \
  synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height/h120cm_upsampled \
  synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height/h117p5cm_upsampled \
  synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height/h115cm_upsampled \
  "$JETSON":/home/arms/hsi-pre-grasp/synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height/

ssh "$JETSON" 'mkdir -p /home/arms/hsi-pre-grasp/synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height_20hz400'
scp -r \
  synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height_20hz400/h112p5cm_upsampled \
  synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height_20hz400/h110cm_upsampled \
  synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height_20hz400/h107p5cm_upsampled \
  "$JETSON":/home/arms/hsi-pre-grasp/synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height_20hz400/
```

Dry-run on the Jetson first. This does not touch CAN or motors:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
HEIGHT=125 DRY_RUN_NO_SLEEP=1 bash run_openarm_saved_episode_replay.txt
```

Real replay requires the operator at the robot, e-stop ready, the selected table
height set, CAN up, and both explicit confirmation flags. Real mode refuses a
rate that does not match the selected dataset family, so the wall-clock command
period stays exactly like the Isaac replay:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
sudo ./scripts/can_up.sh

HEIGHT=125 REAL=1 CONFIRM_HEIGHT=1 bash run_openarm_saved_episode_replay.txt
```

Supported `10hz` heights are `125`, `122.5`, `120`, `117.5`, and `115`.
Supported `20hz400` heights currently present are `112.5`, `110`, and `107.5`.

Examples:

```bash
HEIGHT=122.5 DRY_RUN_NO_SLEEP=1 bash run_openarm_saved_episode_replay.txt
HEIGHT=122.5 REAL=1 CONFIRM_HEIGHT=1 bash run_openarm_saved_episode_replay.txt
HEIGHT=112.5 DRY_RUN_NO_SLEEP=1 bash run_openarm_saved_episode_replay.txt
HEIGHT=112.5 REAL=1 CONFIRM_HEIGHT=1 bash run_openarm_saved_episode_replay.txt
```

## Guarded SmolVLA Mirror

Use normal/default Isaac rendering and camera resolution unless you are
debugging performance. With the Isaac viewer on the 8 GB GPU, use
`--policy-device cpu`; putting both viewer and policy on `cuda:0` can run out of
memory while loading SmolVLA.

```bash
cd /home/chyanin/Desktop/realrobot

/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/interactive_vla_isaac.py \
  --viewer \
  --device cuda:0 \
  --policy-device cpu \
  --mirror-dry-run synthetic_smolvla/reports/interactive_vla_real_shadow_trace.jsonl \
  --mirror-real \
  --prepare-real-start-pose \
  --read-real-state \
  --real-confirm "I am at the robot with e-stop ready" \
  --mirror-rate-hz 2.0 \
  --max-joint-delta-deg 3.0 \
  --watchdog-timeout-sec 2.0 \
  --hold-interval-sec 0.2
```

## Sim First, Save Trajectory, Replay Later

Run Isaac only and save the model commands. This does not contact the real
robot:

```bash
cd /home/chyanin/Desktop/realrobot

/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/interactive_vla_isaac.py \
  --viewer \
  --device cuda:0 \
  --policy-device cpu \
  --mirror-dry-run synthetic_smolvla/reports/sim_first_record_trace.jsonl \
  --save-trajectory-dir synthetic_smolvla/reports/sim_first_saved_trajectories
```

At the prompt, type one task. After it finishes, the script prints
`trajectory_jsonl`; use that file for the separate replay.
The trajectory JSONL saves VLA policy commands, not measured sim poses:
`raw_policy_command_deg` is the direct VLA output, `command_deg` is the clamped
VLA command used for sim apply and optional real replay, and
`observation_state_deg` is the sim state observed before that policy step.

Audit the saved trajectory without moving the robot:

```bash
cd /home/chyanin/Desktop/realrobot

TRAJ=synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_commands.jsonl

python3 synthetic_smolvla/scripts/replay_openarm_trajectory.py \
  --trajectory "$TRAJ" \
  --expected-steps 50 \
  --audit-output-json synthetic_smolvla/reports/sim_first_saved_trajectories/task001_audit.json
```

If the audit fails because the model jumps more than `3 deg` between saved
commands, split the trajectory into smaller replay steps and audit the split
file:

```bash
cd /home/chyanin/Desktop/realrobot

TRAJ=synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_commands.jsonl
SPLIT_TRAJ=synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_split_3deg_commands.jsonl

python3 synthetic_smolvla/scripts/split_openarm_trajectory.py \
  --trajectory "$TRAJ" \
  --output "$SPLIT_TRAJ" \
  --max-joint-delta-deg 3.0 \
  --summary-json synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_split_3deg_summary.json

python3 synthetic_smolvla/scripts/replay_openarm_trajectory.py \
  --trajectory "$SPLIT_TRAJ" \
  --audit-output-json synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_split_3deg_audit.json
```

Later, when physically at the robot with e-stop ready, replay the saved
trajectory through the guarded real path:

```bash
cd /home/chyanin/Desktop/realrobot

TRAJ=synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_split_3deg_commands.jsonl

python3 synthetic_smolvla/scripts/replay_openarm_trajectory.py \
  --trajectory "$TRAJ" \
  --mirror-real \
  --prepare-real-start-pose \
  --read-real-state \
  --real-confirm "I am at the robot with e-stop ready" \
  --replay-rate-hz 2.0 \
  --max-joint-delta-deg 3.0 \
  --watchdog-timeout-sec 2.0 \
  --hold-interval-sec 0.2 \
  --replay-log synthetic_smolvla/reports/sim_first_saved_trajectories/task001_real_replay_log.jsonl
```

Replay streams the split file continuously at `--replay-rate-hz`. After replay,
the script holds the final pose. Type `reset` to move back to the configured
start pose, `hold` to keep holding, or `q` to release/exit.
The start-pose move uses the same streamed target path, so `--replay-rate-hz`
and `--max-joint-delta-deg` also apply while moving to the initial pose before
replay.

For a later gripper-enabled test, re-split with gripper deltas included, audit
with `--enable-gripper-real`, and replay more slowly:

```bash
cd /home/chyanin/Desktop/realrobot

TRAJ=synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_commands.jsonl
GRIPPER_TRAJ=synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_split_3deg_arm_gripper_commands.jsonl

python3 synthetic_smolvla/scripts/split_openarm_trajectory.py \
  --trajectory "$TRAJ" \
  --output "$GRIPPER_TRAJ" \
  --max-joint-delta-deg 3.0 \
  --include-gripper-delta \
  --summary-json synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_split_3deg_arm_gripper_summary.json

python3 synthetic_smolvla/scripts/replay_openarm_trajectory.py \
  --trajectory "$GRIPPER_TRAJ" \
  --enable-gripper-real \
  --audit-output-json synthetic_smolvla/reports/sim_first_saved_trajectories/task001_red_cube_split_3deg_arm_gripper_audit.json
```

Only after that audit passes, replay on the real robot with the operator at the
robot and e-stop ready:

```bash
python3 synthetic_smolvla/scripts/replay_openarm_trajectory.py \
  --trajectory "$GRIPPER_TRAJ" \
  --mirror-real \
  --prepare-real-start-pose \
  --read-real-state \
  --real-confirm "I am at the robot with e-stop ready" \
  --enable-gripper-real \
  --replay-rate-hz 0.5 \
  --max-joint-delta-deg 3.0 \
  --real-start-pose-timeout-sec 60.0 \
  --watchdog-timeout-sec 2.0 \
  --hold-interval-sec 0.2
```

The initial start-pose move uses the normal guarded helper `prepare_start`
operation. After that, the script reads back the real joint state and aborts if
the configured start pose is not reached within tolerance. The saved trajectory
replay still uses the configured replay rate and max joint delta checks.
If the start pose times out, rerun with a longer `--real-start-pose-timeout-sec`
only after checking the robot is clear and the e-stop is ready; do not relax the
tolerance unless you intentionally accept a looser start pose.

Optional separate zero-then-start preparation:

```bash
python3 synthetic_smolvla/scripts/prepare_openarm_zero_then_start.py \
  --mirror-real \
  --real-confirm "I am at the robot with e-stop ready" \
  --enable-gripper-real \
  --max-joint-delta-deg 1.0 \
  --zero-stage-sec 8.0 \
  --real-start-pose-timeout-sec 120.0 \
  --watchdog-timeout-sec 2.0 \
  --hold-interval-sec 0.03
```

This script first sends an arm-zero staging pose for `--zero-stage-sec` without
requiring stable zero, then prepares and audits the configured sim start pose.
It holds the final start pose until Ctrl-C. It is dry-run unless `--mirror-real`
is present. Add `--no-hold-final` if you only want the two-stage preparation
check and then release.

To continue directly from zero stage to init to VLA replay in one helper
session, use the combined runner:

```bash
python3 synthetic_smolvla/scripts/replay_openarm_zero_then_start_trajectory.py \
  --trajectory "$UPSAMPLED_TRAJ" \
  --mirror-real \
  --read-real-state \
  --real-confirm "I am at the robot with e-stop ready" \
  --enable-gripper-real \
  --replay-rate-hz 25.0 \
  --max-joint-delta-deg 1.0 \
  --real-helper-max-rel-deg 90.0 \
  --zero-stage-sec 8.0 \
  --real-start-pose-timeout-sec 120.0 \
  --real-start-pose-tolerance-deg 5.0 \
  --real-start-pose-hold-sec 0.05 \
  --real-start-pose-samples 5 \
  --watchdog-timeout-sec 2.0 \
  --hold-interval-sec 0.03 \
  --replay-log "${UPSAMPLED_TRAJ%.jsonl}_zero_start_real_replay_25hz_log.jsonl"
```

Fixed 10 samples per VLA command:

```bash
RAW_TRAJ="synthetic_smolvla/reports/sim_first_20260618_124358/trajectories/task001_red_cube_commands.jsonl"
FIXED10_TRAJ="${RAW_TRAJ%.jsonl}_fixed10_arm_gripper.jsonl"

python3 synthetic_smolvla/scripts/split_openarm_trajectory.py \
  --trajectory "$RAW_TRAJ" \
  --output "$FIXED10_TRAJ" \
  --samples-per-command 10 \
  --include-gripper-delta \
  --summary-json "${FIXED10_TRAJ%.jsonl}_summary.json"

python3 synthetic_smolvla/scripts/replay_openarm_zero_then_start_trajectory.py \
  --trajectory "$FIXED10_TRAJ" \
  --enable-gripper-real \
  --replay-rate-hz 25.0 \
  --max-joint-delta-deg 1.0 \
  --real-helper-max-rel-deg 90.0 \
  --trajectory-max-joint-delta-deg 13.0 \
  --zero-stage-sec 8.0 \
  --real-start-pose-timeout-sec 120.0 \
  --real-start-pose-tolerance-deg 5.0 \
  --real-start-pose-hold-sec 0.05 \
  --real-start-pose-samples 5 \
  --audit-output-json "${FIXED10_TRAJ%.jsonl}_audit.json"
```

The fixed-10 file has exactly 10 saved samples per raw VLA command. If the raw
VLA has a large jump, the real mirror still inserts internal 1-degree safety
substeps before sending to the helper.
The `--real-start-pose-samples 5` flag splits init/start preparation into 5
sampled OpenArm `prepare_start` targets. The `--real-helper-max-rel-deg 90.0`
flag prevents each init sample from creeping at the small VLA sampling delta;
VLA replay still uses `--max-joint-delta-deg`.

Zero pose only, using the same guarded default OpenArm position method:

```bash
python3 synthetic_smolvla/scripts/prepare_openarm_zero_pose.py \
  --mirror-real \
  --real-confirm "I am at the robot with e-stop ready" \
  --enable-gripper-real \
  --max-joint-delta-deg 1.0 \
  --real-pose-timeout-sec 120.0 \
  --real-pose-tolerance-deg 5.0 \
  --real-pose-hold-sec 0.05 \
  --watchdog-timeout-sec 2.0 \
  --hold-interval-sec 0.03
```

This does not continue to init or VLA replay. The safe zero target clamps
`joint_4` to `2.0 deg` if the configured limits reject `0.0 deg`, and now sends
the gripper to `0.0 deg` when `--enable-gripper-real` is present.

Plot folders:

```text
synthetic_smolvla/reports/sim_first_20260618_124358/plots
```

Raw VLA vs resampled plot:

```bash
python3 synthetic_smolvla/scripts/plot_openarm_resampling.py \
  --raw "$RAW_TRAJ" \
  --resampled "$FIXED10_TRAJ" \
  --output-dir synthetic_smolvla/reports/sim_first_20260618_124358/plots \
  --prefix task001_red_cube_fixed10 \
  --title "Right Arm VLA Fixed-10 Resampling"
```

Real readback vs VLA target plot, after rerunning replay with the updated logger:

```bash
REAL_LOG="${FIXED10_TRAJ%.jsonl}_zero_start_real_replay_25hz_log.jsonl"

python3 synthetic_smolvla/scripts/plot_openarm_real_vs_vla.py \
  --replay-log "$REAL_LOG" \
  --output-dir synthetic_smolvla/reports/sim_first_20260618_124358/plots \
  --prefix task001_red_cube_fixed10
```

Older replay logs do not contain per-command `real_state_deg`, so they cannot
produce the real-vs-VLA plot. Rerun replay once with the current scripts first.
