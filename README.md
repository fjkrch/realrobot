# realrobot Agent README

Start here when another agent or operator opens `/home/chyanin/Desktop/realrobot`.

This folder is a laptop-side control and documentation workspace for OpenArm.
The laptop runs Isaac Sim / Isaac Lab and mirror servers. The real robot is
driven from the Jetson at `/home/arms/hsi-pre-grasp`.

## First Rule

- `jetson_isaaclab_command.py` commands are simulation only.
- Real robot motion happens only through scripts that require
  `--i-am-at-robot`, such as `move_joint.py`, `move_arm.py`,
  `pick_cube.py`, or the Jetson-side `open_gripper_small.py`.
- Before real motion: person at robot, e-stop ready, 24 V motor power on.

## Folder Read Order

| Need | Read first | Then read |
|---|---|---|
| Fast orientation | [docs/agent-handoff/README.md](docs/agent-handoff/README.md) | [SUMMARY.md](SUMMARY.md) |
| Isaac-only mirror | [docs/sim/README.md](docs/sim/README.md) | [DEFAULT_OPENARM_ISAAC_MIRROR.md](docs/sim/DEFAULT_OPENARM_ISAAC_MIRROR.md) |
| Whole-body sim commands | [docs/sim/README.md](docs/sim/README.md) | [WHOLE_BODY_OPENARM_ISAAC_MIRROR.md](docs/sim/WHOLE_BODY_OPENARM_ISAAC_MIRROR.md) |
| Current SmolVLA continuation | [docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md](docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md) | [dense_isaac_camera_v1_dataset.md](synthetic_smolvla/reports/dense_isaac_camera_v1_dataset.md) (corrected dataset), [14k audit](synthetic_smolvla/reports/openarm_success_filtered_14000_dataset_audit.md) (why it was corrected) |
| Synthetic SmolVLA training plan | [docs/sim/README.md](docs/sim/README.md) | [SMOLVLA_SYNTHETIC_ONLY_PLAN.md](docs/sim/SMOLVLA_SYNTHETIC_ONLY_PLAN.md) |
| Synthetic SmolVLA implementation | [synthetic_smolvla/README.md](synthetic_smolvla/README.md) | [SMOLVLA_SYNTHETIC_ONLY_PLAN.md](docs/sim/SMOLVLA_SYNTHETIC_ONLY_PLAN.md) |
| Learned RL pick oracle | [docs/sim/OPENARM_RL_GRASP_PLAN.md](docs/sim/OPENARM_RL_GRASP_PLAN.md) | [synthetic_smolvla/rl/openarm_pick_env.py](synthetic_smolvla/rl/openarm_pick_env.py) |
| Real robot motion | [docs/real/README.md](docs/real/README.md) | [REAL_ROBOT_MOVE.md](docs/real/REAL_ROBOT_MOVE.md) |
| Cube pick sim-to-real | [docs/real/README.md](docs/real/README.md) | [PICK_CUBE_SIM_TO_REAL.md](docs/real/PICK_CUBE_SIM_TO_REAL.md) |
| Limits / clamps | [docs/reference/README.md](docs/reference/README.md) | [OPENARM_JOINT_LIMITS.md](docs/reference/OPENARM_JOINT_LIMITS.md) |
| Something is broken | [docs/troubleshooting/README.md](docs/troubleshooting/README.md) | The symptom-specific doc listed there |

## Important Root Files

| File | Purpose |
|---|---|
| [SUMMARY.md](SUMMARY.md) | Flat index of files and scripts. |
| [docs/real/REAL_ROBOT_MOVE.md](docs/real/REAL_ROBOT_MOVE.md) | Real robot motion commands and safety. |
| [docs/sim/DEFAULT_OPENARM_ISAAC_MIRROR.md](docs/sim/DEFAULT_OPENARM_ISAAC_MIRROR.md) | Clean Isaac mirror: OpenArm only, no cube/task. |
| [docs/sim/WHOLE_BODY_OPENARM_ISAAC_MIRROR.md](docs/sim/WHOLE_BODY_OPENARM_ISAAC_MIRROR.md) | Sim-only joint and whole-body commands. |
| [docs/sim/REAL_JETSON_TO_ISAAC_SIMULATION.md](docs/sim/REAL_JETSON_TO_ISAAC_SIMULATION.md) | Full Jetson-to-laptop mirror guide. |
| [docs/sim/JETSON_ISAACLAB_MIRROR.md](docs/sim/JETSON_ISAACLAB_MIRROR.md) | Short Jetson-to-Isaac guide. |
| [docs/sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md](docs/sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md) | Local Isaac Lab / Isaac Sim setup and troubleshooting. |
| [docs/sim/SMOLVLA_SYNTHETIC_ONLY_PLAN.md](docs/sim/SMOLVLA_SYNTHETIC_ONLY_PLAN.md) | Synthetic-only SmolVLA data, training, and evaluation plan. |
| [docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md](docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md) | Current SmolVLA status, completed run, dataset warning, and next-agent checklist. |
| [synthetic_smolvla/reports/openarm_success_filtered_14000_dataset_audit.md](synthetic_smolvla/reports/openarm_success_filtered_14000_dataset_audit.md) | Audit explaining why the old 14k dataset was corrected. |
| [synthetic_smolvla/reports/dense_isaac_camera_v1_dataset.md](synthetic_smolvla/reports/dense_isaac_camera_v1_dataset.md) | Corrected dense Isaac-camera dataset (v1): build, verification, audit-gap resolution, training, eval. |
| [synthetic_smolvla/README.md](synthetic_smolvla/README.md) | Simulation-only SmolVLA scaffold: configs, limit contract, oracle manifests, preflight tools. |
| [docs/reference/OPENARM_JOINT_LIMITS.md](docs/reference/OPENARM_JOINT_LIMITS.md) | Real and safe simulation joint limits. |
| [docs/real/real_robot_calibration.md](docs/real/real_robot_calibration.md) | Commit-safe robot handoff notes with password redacted. |
| `docs/real/OPENARM_ROBOT_HANDOFF.md` | Private handoff with password. Do not commit or paste publicly. |

## Script Map

| Script | Runs on | Purpose |
|---|---|---|
| [scripts/openarm_default_pose_live_server.py](scripts/openarm_default_pose_live_server.py) | Laptop | Always-open clean Isaac mirror. |
| [run_default_openarm_mirror.txt](run_default_openarm_mirror.txt) | Laptop | Launcher for the clean Isaac mirror. |
| [scripts/isaaclab_python.sh](scripts/isaaclab_python.sh) | Laptop | Short wrapper for `conda run -n env_isaaclab ./isaaclab.sh -p ...`. |
| [scripts/install_isaaclab_shortcuts.sh](scripts/install_isaaclab_shortcuts.sh) | Laptop | Installs `~/IsaacLab` shortcuts for the wrapper and mirror launcher. |
| [scripts/jetson_isaaclab_command.py](scripts/jetson_isaaclab_command.py) | Jetson | Sends sim-only commands to laptop. |
| [scripts/isaaclab_command_server.py](scripts/isaaclab_command_server.py) | Laptop | Older per-command Isaac bridge. |
| [scripts/move_joint.py](scripts/move_joint.py) | Jetson | Move one real joint. Requires `--i-am-at-robot`. |
| [scripts/move_arm.py](scripts/move_arm.py) | Jetson | Move all 7 joints of one real arm. Requires `--i-am-at-robot`. |
| [scripts/pick_cube.py](scripts/pick_cube.py) | Jetson | Teach, sim-check, and real-replay cube pick. |
| [synthetic_smolvla/scripts/collect_oracle_demos.py](synthetic_smolvla/scripts/collect_oracle_demos.py) | Laptop | Simulation-only oracle manifest generator (scaffold; hard-coded labels). |
| [synthetic_smolvla/scripts/oracle_pick_ik.py](synthetic_smolvla/scripts/oracle_pick_ik.py) | Laptop | Real closed-loop IK oracle; picks in physics and measures lift. |
| [synthetic_smolvla/scripts/collect_dense_isaac_dataset.py](synthetic_smolvla/scripts/collect_dense_isaac_dataset.py) | Laptop | Corrected dense dataset collector: real Isaac camera frames + measured state + commanded action per step. |
| [synthetic_smolvla/scripts/verify_dense_dataset.py](synthetic_smolvla/scripts/verify_dense_dataset.py) | Laptop | Pre-training gate: LeRobot load + shapes + non-static-frame check. |
| [synthetic_smolvla/scripts/train_smolvla.py](synthetic_smolvla/scripts/train_smolvla.py) | Laptop | SmolVLA training preflight + launcher (finetune from `lerobot/smolvla_base`). |
| [synthetic_smolvla/scripts/eval_vla_isaac.py](synthetic_smolvla/scripts/eval_vla_isaac.py) | Laptop | Closed-loop Isaac physics eval of a trained SmolVLA policy (measured lift). |
| [synthetic_smolvla/scripts/probe_openarm_reach.py](synthetic_smolvla/scripts/probe_openarm_reach.py) | Laptop | Maps the active arm's reachable workspace via differential IK. |
| [synthetic_smolvla/rl/train_rl.py](synthetic_smolvla/rl/train_rl.py) | Laptop | PPO (rsl_rl) trainer for the learned OpenArm pick policy. |
| [synthetic_smolvla/rl/openarm_pick_env.py](synthetic_smolvla/rl/openarm_pick_env.py) | Laptop | DirectRLEnv for the target-conditioned OpenArm pick task. |

## Quick Commands

Start clean Isaac mirror on laptop:

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt
```

Send sim-only command from Jetson:

```bash
cd /home/arms/hsi-pre-grasp
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
```

Real robot docs start here:

```text
docs/real/README.md
```

Troubleshooting starts here:

```text
docs/troubleshooting/README.md
```

Synthetic-only SmolVLA scaffold smoke test:

```bash
python3 synthetic_smolvla/scripts/make_scene.py --dry-run
python3 synthetic_smolvla/scripts/collect_oracle_demos.py --episodes 8 --output /tmp/openarm_oracle_demo.jsonl
```
