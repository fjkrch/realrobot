# Whole Body OpenArm Isaac Mirror

This file is for testing OpenArm arm joints in Isaac first, without moving the
real robot.

Use `jetson_isaaclab_command.py` for sim-only commands. Do not use real robot
motion scripts when you only want Isaac.

## 1. Start Isaac On Laptop

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt
```

This opens only OpenArm, ground, and light. There is no cube, no pick task, and
no episode reset.

## 2. Send Sim-Only Joint Commands From Jetson

```bash
cd /home/arms/hsi-pre-grasp
python scripts/jetson_isaaclab_command.py "joint right 1 10 deg"
```

This means:

```text
right arm, joint 1, target 10 degrees in Isaac only
```

More examples:

```bash
python scripts/jetson_isaaclab_command.py "joint right 2 -15 deg"
python scripts/jetson_isaaclab_command.py "right j3=20"
python scripts/jetson_isaaclab_command.py "left j1=-10"
python scripts/jetson_isaaclab_command.py "openarm_right_joint4=30"
```

Return to default pose:

```bash
python scripts/jetson_isaaclab_command.py "default"
```

## 3. Set One Arm With 7 Values

Right arm:

```bash
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
```

Left arm:

```bash
python scripts/jetson_isaaclab_command.py "whole body left 0 5 -10 20 0 0 0"
```

The 7 values are:

```text
joint1 joint2 joint3 joint4 joint5 joint6 joint7
```

## 4. Set Both Arms With 14 Values

```bash
python scripts/jetson_isaaclab_command.py "whole body 0 0 0 0 0 0 0 0 10 -10 20 0 0 0"
```

The first 7 values are the left arm. The next 7 values are the right arm.

```text
left_j1 left_j2 left_j3 left_j4 left_j5 left_j6 left_j7 right_j1 right_j2 right_j3 right_j4 right_j5 right_j6 right_j7
```

## 5. Status

```bash
python scripts/jetson_isaaclab_command.py --status
```

The status response shows the last accepted command and the parsed joint targets
in degrees.

## Safety

These commands are sim-only:

```bash
python scripts/jetson_isaaclab_command.py "joint right 1 10 deg"
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
```

These are real robot commands and should wait until a person is at the robot:

```bash
python scripts/open_gripper_small.py ...
python scripts/teleop_native.py ...
```
