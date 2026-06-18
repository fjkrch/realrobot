# OpenArm Joint Limits

This file records the joint limits to use for real robot safety and Isaac
simulation clamping.

Source used here:

- Real robot API limits already recorded in `../real/REAL_ROBOT_MOVE.md`
- `move_joint.py` / `move_arm.py` read `robot.config.joint_limits` and refuse
  out-of-range targets on the real robot
- Isaac uses radians internally, so degree limits are converted to radians for
  simulation

## Real Robot API Limits

Values are in degrees.

| Joint | Right arm `can0` | Left arm `can1` |
|---|---:|---:|
| `joint_1` | `-75 .. 75` | `-75 .. 75` |
| `joint_2` | `-9 .. 90` | `-90 .. 9` |
| `joint_3` | `-85 .. 85` | `-85 .. 85` |
| `joint_4` | `0 .. 135` | `0 .. 135` |
| `joint_5` | `-85 .. 85` | `-85 .. 85` |
| `joint_6` | `-40 .. 40` | `-40 .. 40` |
| `joint_7` | `-80 .. 80` | `-80 .. 80` |
| `gripper` | `-65 .. 0` | `-65 .. 0` |

Important:

- `joint_2` is mirrored between right and left arms.
- Gripper: `0 deg` is closed, `-65 deg` is open.
- Positive gripper values are not allowed.

## Safe Simulation Limits

Use these for Isaac command clamping. They are slightly inside the real robot
limits by `2 deg` on arm joints.

Gripper is kept exact at `-65 .. 0 deg` because that mapping is already narrow
and explicit.

| Joint | Right safe deg | Right safe rad | Left safe deg | Left safe rad |
|---|---:|---:|---:|---:|
| `joint_1` | `-73 .. 73` | `-1.274 .. 1.274` | `-73 .. 73` | `-1.274 .. 1.274` |
| `joint_2` | `-7 .. 88` | `-0.122 .. 1.536` | `-88 .. 7` | `-1.536 .. 0.122` |
| `joint_3` | `-83 .. 83` | `-1.449 .. 1.449` | `-83 .. 83` | `-1.449 .. 1.449` |
| `joint_4` | `2 .. 133` | `0.035 .. 2.321` | `2 .. 133` | `0.035 .. 2.321` |
| `joint_5` | `-83 .. 83` | `-1.449 .. 1.449` | `-83 .. 83` | `-1.449 .. 1.449` |
| `joint_6` | `-38 .. 38` | `-0.663 .. 0.663` | `-38 .. 38` | `-0.663 .. 0.663` |
| `joint_7` | `-78 .. 78` | `-1.361 .. 1.361` | `-78 .. 78` | `-1.361 .. 1.361` |
| `gripper` | `-65 .. 0` | sim finger `0.044 .. 0.000 m` | `-65 .. 0` | sim finger `0.044 .. 0.000 m` |

## Gripper Mapping

The real gripper uses degrees. Isaac finger joints use meters per finger.

```text
0 deg   -> 0.000 m/finger closed
-65 deg -> 0.044 m/finger fully open
-10 deg -> 0.0068 m/finger
```

Formula:

```text
sim_finger_m = (-clamp(real_deg, -65, 0) / 65) * 0.044
```

## Commands That Must Stay Inside These Limits

Sim-only Isaac commands:

```bash
python scripts/jetson_isaaclab_command.py "joint right 1 10 deg"
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
python scripts/jetson_isaaclab_command.py "gripper target -10 deg"
```

Real robot commands:

```bash
python scripts/move_joint.py --port can0 --side right --joint 1 --target-deg 10 --i-am-at-robot --yes
python scripts/move_arm.py "whole body right 0 5 -10 20 0 0 0" --i-am-at-robot --yes
python scripts/open_gripper_small.py --port can0 --side right --target-deg -10 --i-am-at-robot --yes
```

Real robot scripts read live API limits before moving. Isaac should clamp to the
safe simulation limits above.
