# Agent Handoff README

Read this first if you are a new agent.

## Safety Boundary

- Sim-only: `jetson_isaaclab_command.py`, Isaac mirror servers, and all commands
  sent to `http://10.10.10.1:8765`.
- Real motion: `move_joint.py`, `move_arm.py`, `pick_cube.py` real replay, and
  Jetson-side robot scripts such as `open_gripper_small.py`.
- Real motion requires a human at the robot, e-stop ready, and `--i-am-at-robot`.

## Read Order

1. [../../README.md](../../README.md)
2. [../../SUMMARY.md](../../SUMMARY.md)
3. [../reference/OPENARM_JOINT_LIMITS.md](../reference/OPENARM_JOINT_LIMITS.md)
4. For the current VLA training state, read
   [SMOLVLA_TRAINING_HANDOFF.md](SMOLVLA_TRAINING_HANDOFF.md)
5. Pick the task-specific folder:
   - [../sim/README.md](../sim/README.md)
   - [../real/README.md](../real/README.md)
   - [../troubleshooting/README.md](../troubleshooting/README.md)

## Current SmolVLA Situation

If the user asks another agent to continue the VLA work, start with
[SMOLVLA_TRAINING_HANDOFF.md](SMOLVLA_TRAINING_HANDOFF.md), then read the
dataset audit:

`../../synthetic_smolvla/reports/openarm_success_filtered_14000_dataset_audit.md`

Current state:

- The task is target-conditioned pick-and-lift, not place/drop.
- A 14,000-source oracle run produced 6,280 success-filtered episodes.
- SmolVLA trained for 10,000 steps and reached low loss, but Isaac VLA eval was
  still 0 successful lifts in audited trials.
- The dataset is structurally valid, but not good enough for reliable physics:
  it has five sparse keyframes per episode and static placeholder images, not
  real Isaac camera frames with arm/gripper/contact/lift motion.

Update (2026-06-17): the corrected dataset has been built —
`synthetic_smolvla/datasets/openarm_dense_isaac_camera_v1` (940 dense episodes,
real Isaac camera frames per step, measured state + commanded action, 0
wrong-object, balanced targets). It is verified loadable and non-static. SmolVLA
is finetuned from `lerobot/smolvla_base` on it; the Isaac physics eval is the
remaining gate. See the dataset report
`synthetic_smolvla/reports/dense_isaac_camera_v1_dataset.md` and the
[SmolVLA training handoff](SMOLVLA_TRAINING_HANDOFF.md). The old 14k dataset is
kept as evidence, not deleted.

## Sensitive File

`../real/OPENARM_ROBOT_HANDOFF.md` contains the robot SSH password and is
ignored by git. Do not paste its credential content into chat or commits.

Use [../real/real_robot_calibration.md](../real/real_robot_calibration.md) when
you need the same operational notes without exposing the password.

## Current Architecture

| Part | File |
|---|---|
| Clean Isaac mirror server | [../../scripts/openarm_default_pose_live_server.py](../../scripts/openarm_default_pose_live_server.py) |
| Clean Isaac launcher | [../../run_default_openarm_mirror.txt](../../run_default_openarm_mirror.txt) |
| Jetson sim command client | [../../scripts/jetson_isaaclab_command.py](../../scripts/jetson_isaaclab_command.py) |
| Real single-joint motion | [../../scripts/move_joint.py](../../scripts/move_joint.py) |
| Real whole-arm motion | [../../scripts/move_arm.py](../../scripts/move_arm.py) |
| Cube-pick workflow | [../../scripts/pick_cube.py](../../scripts/pick_cube.py) |

## Before Editing

- Check `git status --short`.
- Assume untracked or modified files may be intentional.
- Keep root `README.md` and `SUMMARY.md` as the start-here files.
- Prefer adding navigation docs over breaking known command paths.
