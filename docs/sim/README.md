# Simulation README

Use this folder when the task should happen in Isaac only.

## Best First Choice

For default OpenArm pose, no cube, no pick task, no episode reset:

1. Read [DEFAULT_OPENARM_ISAAC_MIRROR.md](DEFAULT_OPENARM_ISAAC_MIRROR.md).
2. Start:

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt
```

3. From Jetson, send sim-only commands:

```bash
cd /home/arms/hsi-pre-grasp
python scripts/jetson_isaaclab_command.py "joint right 1 10 deg"
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
```

## Whole Body Sim Commands

Read [WHOLE_BODY_OPENARM_ISAAC_MIRROR.md](WHOLE_BODY_OPENARM_ISAAC_MIRROR.md)
when the user asks to test joints, whole arm, or whole body in Isaac first.

## Jetson To Isaac Bridge

Read:

- [JETSON_ISAACLAB_MIRROR.md](JETSON_ISAACLAB_MIRROR.md) for short commands.
- [REAL_JETSON_TO_ISAAC_SIMULATION.md](REAL_JETSON_TO_ISAAC_SIMULATION.md) for the full guide.

## Isaac Local Setup

Read [ISAACLAB_ISAACSIM_LOCAL_GUIDE.md](ISAACLAB_ISAACSIM_LOCAL_GUIDE.md)
when Isaac Sim, Isaac Lab, conda, GUI, GPU, or imports fail.

## Synthetic SmolVLA Training

Read [SMOLVLA_SYNTHETIC_ONLY_PLAN.md](SMOLVLA_SYNTHETIC_ONLY_PLAN.md) when the
task is synthetic-only SmolVLA data generation, training, or evaluation for
OpenArm object pickup.

Implementation scaffold:

```bash
python3 synthetic_smolvla/scripts/make_scene.py --dry-run
python3 synthetic_smolvla/scripts/collect_oracle_demos.py --episodes 8 --output /tmp/openarm_oracle_demo.jsonl
```

The scaffold lives at [../../synthetic_smolvla](../../synthetic_smolvla/README.md).

## Safety Reminder

Anything through `jetson_isaaclab_command.py` is sim-only. It does not move the
physical robot.
