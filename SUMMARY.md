# realrobot — Folder Summary

Index of everything in `/home/chyanin/Desktop/realrobot`. This laptop runs the
**Isaac Sim** side and holds helper scripts. The **real OpenArm robot** is driven
from the **Jetson** (`/home/arms/hsi-pre-grasp`), not from this folder.

## Two Machines

| Machine | Role | LAN | Wi-Fi |
|---|---|---|---|
| Laptop | Isaac Sim / Lab + mirror server + scripts | `10.10.10.1` | — |
| Jetson | Drives the physical robot over CAN | `10.10.10.2` | `192.168.31.50` |

## The One Rule: Sim vs Real

- **Isaac / mirror commands = simulation only.** They never move the real robot.
  All `jetson_isaaclab_command.py` "joint"/"whole body"/"pick" commands are sim.
- **Real motion** happens only via the Jetson scripts that require
  `--i-am-at-robot` (gripper, move_joint, move_arm, teleop).
- Real motion needs a **person at the rig, e-stop ready, 24 V on**.

## Markdown Docs

| File | What it covers | Sim/Real |
|---|---|---|
| [README.md](README.md) | Start-here agent README and routing map | — |
| [SUMMARY.md](SUMMARY.md) | This index | — |
| [docs/README.md](docs/README.md) | Organized folder navigation layer | — |
| [docs/real/REAL_ROBOT_MOVE.md](docs/real/REAL_ROBOT_MOVE.md) | Consolidated real-robot motion commands (gripper, joint, whole-arm, teleop, stop) | Real |
| [docs/real/PICK_CUBE_SIM_TO_REAL.md](docs/real/PICK_CUBE_SIM_TO_REAL.md) | Scripted cube pick: teach poses by hand, verify in Isaac, replay on real | Sim+Real |
| `docs/real/OPENARM_ROBOT_HANDOFF.md` | Full operator handoff: SSH, CAN, camera, gripper, teleop. **Contains password — do not commit.** | Real |
| [docs/real/real_robot_calibration.md](docs/real/real_robot_calibration.md) | Same handoff content with the password redacted (commit-safe copy) | Real |
| [docs/sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md](docs/sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md) | Running Isaac Sim / Isaac Lab on this laptop | Sim |
| [docs/sim/REAL_JETSON_TO_ISAAC_SIMULATION.md](docs/sim/REAL_JETSON_TO_ISAAC_SIMULATION.md) | Jetson -> laptop Isaac mirror bridge (full guide) | Sim |
| [docs/sim/JETSON_ISAACLAB_MIRROR.md](docs/sim/JETSON_ISAACLAB_MIRROR.md) | Jetson -> Isaac mirror, short version | Sim |
| [docs/sim/DEFAULT_OPENARM_ISAAC_MIRROR.md](docs/sim/DEFAULT_OPENARM_ISAAC_MIRROR.md) | Clean default-pose Isaac mirror (no cube/task) | Sim |
| [docs/sim/WHOLE_BODY_OPENARM_ISAAC_MIRROR.md](docs/sim/WHOLE_BODY_OPENARM_ISAAC_MIRROR.md) | Sim-only per-joint / whole-body Isaac commands | Sim |
| [docs/sim/SMOLVLA_SYNTHETIC_ONLY_PLAN.md](docs/sim/SMOLVLA_SYNTHETIC_ONLY_PLAN.md) | Synthetic-only SmolVLA dataset, training, and evaluation plan | Sim |
| [synthetic_smolvla/README.md](synthetic_smolvla/README.md) | First implementation slice for the synthetic-only SmolVLA plan | Sim |
| [docs/reference/OPENARM_JOINT_LIMITS.md](docs/reference/OPENARM_JOINT_LIMITS.md) | Real robot API limits and safe Isaac limits | Sim+Real |

## Docs Folders

| Folder | What to read there |
|---|---|
| [docs/agent-handoff](docs/agent-handoff/README.md) | New agent read order and safety boundary |
| [docs/sim](docs/sim/README.md) | Isaac-only mirror and sim command routing |
| [docs/real](docs/real/README.md) | Physical robot motion routing |
| [docs/troubleshooting](docs/troubleshooting/README.md) | Symptom-to-file troubleshooting map |
| [docs/reference](docs/reference/README.md) | Limits, paths, addresses, and static facts |
| [synthetic_smolvla](synthetic_smolvla/README.md) | Simulation-only SmolVLA configs, safety contract, oracle manifests, and reports |

## Scripts

Helper scripts:

| Script | Purpose | Runs on | Sim/Real |
|---|---|---|---|
| `isaaclab_command_server.py` | Laptop mirror server (launches Isaac per command) | Laptop | Sim |
| `openarm_default_pose_live_server.py` | Always-open Isaac default-pose server | Laptop | Sim |
| `jetson_isaaclab_command.py` | Client to send mirror commands to the laptop | Jetson | Sim |
| `move_joint.py` | Move ONE real arm joint to an angle | Jetson* | Real |
| `move_arm.py` | Move all 7 real joints at once ("whole body") | Jetson* | Real |
| `pick_cube.py` | Scripted cube pick (teach / sim-only / real replay) | Jetson* | Sim+Real |
| `synthetic_smolvla/scripts/make_scene.py` | Validate four-object scene config and write a manifest | Laptop | Sim |
| `synthetic_smolvla/scripts/collect_oracle_demos.py` | Generate simulation-only oracle JSONL manifests | Laptop | Sim |
| `synthetic_smolvla/scripts/eval_smolvla.py` | Evaluate manifest labels and write markdown metrics | Laptop | Sim |
| `synthetic_smolvla/scripts/stress_test.py` | Generate all-objects-visible manifest stress reports | Laptop | Sim |
| `synthetic_smolvla/scripts/train_smolvla.py` | Preflight + launch SmolVLA training (finetune from `lerobot/smolvla_base`) | Laptop | Sim |
| `synthetic_smolvla/scripts/collect_dense_isaac_dataset.py` | Corrected dense dataset: real Isaac camera + measured state + commanded action per step | Laptop | Sim |
| `synthetic_smolvla/scripts/verify_dense_dataset.py` | Pre-training gate: LeRobot load, shapes, non-static-frame check | Laptop | Sim |
| `synthetic_smolvla/scripts/eval_vla_isaac.py` | Closed-loop Isaac physics eval of a trained SmolVLA policy (measured lift) | Laptop | Sim |

\* `move_joint.py` and `move_arm.py` are written here but must be `scp`'d to the
Jetson to run. The robot-side gripper/teleop/disable scripts
(`open_gripper_small.py`, `teleop_native.py`, `disable_torque.py`) live on the
Jetson under `/home/arms/hsi-pre-grasp/scripts/`.

Other:
- `run_default_openarm_mirror.txt` — launches the clean default-pose Isaac server.
- `scripts/isaaclab_python.sh` — shorter wrapper for Isaac Lab Python scripts:
  `scripts/isaaclab_python.sh path/to/script.py`.
- `scripts/install_isaaclab_shortcuts.sh` — recreates
  `/home/chyanin/IsaacLab/isaaclab_python.sh` and
  `/home/chyanin/IsaacLab/run_openarm_mirror.sh` symlinks.
- `logs/command_bridge/` — laptop mirror command logs.

## Quick Start

### A. Watch in Isaac only (no real motion)

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt
# then from the Jetson:
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
```

### B. Move the real robot (person at rig, e-stop ready)

Copy the motion scripts to the Jetson once (from the laptop):

```bash
cd /home/chyanin/Desktop/realrobot
scp scripts/move_joint.py scripts/move_arm.py arms@10.10.10.2:/home/arms/hsi-pre-grasp/scripts/
```

Then on the Jetson:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
sudo ./scripts/can_up.sh

# gripper (existing)
python scripts/open_gripper_small.py --port can0 --side right --target-deg -10 --i-am-at-robot --yes

# one joint
python scripts/move_joint.py --port can0 --side right --joint 1 --delta-deg 3 --i-am-at-robot

# all 7 joints
python scripts/move_arm.py "whole body right 0 5 -10 20 0 0 0" --i-am-at-robot
```

### Stop / release torque (Jetson)

```bash
python scripts/disable_torque.py --port can0 --side right
```

## Do Not Commit

`docs/real/OPENARM_ROBOT_HANDOFF.md` contains the robot SSH password. Keep it
local. The commit-safe equivalent is
`docs/real/real_robot_calibration.md` (password redacted).
