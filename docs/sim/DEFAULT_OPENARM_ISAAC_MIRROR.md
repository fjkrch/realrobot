# Default OpenArm Isaac Mirror

This is the clean standalone mirror. It is separate from the
`hsi_pregrasp_refusal` Isaac task project.

It opens only:

- OpenArm robot
- ground plane
- light

It does not open:

- cube task
- pick state machine
- episode reset
- object randomization

## Laptop: Start Isaac

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt
```

If your terminal is already in Isaac Lab:

```bash
cd /home/chyanin/IsaacLab
./run_openarm_mirror.sh
```

That launcher uses `scripts/isaaclab_python.sh`, so you do not need to manually
type `conda run -n env_isaaclab ./isaaclab.sh -p ...`.

To check what it will run without opening Isaac:

```bash
ISAACLAB_DRY_RUN=1 bash run_default_openarm_mirror.txt
```

From `/home/chyanin/IsaacLab`:

```bash
ISAACLAB_DRY_RUN=1 ./run_openarm_mirror.sh
```

Keep this terminal open. Isaac listens on:

```text
http://10.10.10.1:8765
```

## Jetson: Send Joint Commands Only To Isaac

These commands do not move the real robot:

```bash
cd /home/arms/hsi-pre-grasp
python scripts/jetson_isaaclab_command.py "joint right 1 10 deg"
python scripts/jetson_isaaclab_command.py "right j2=-15"
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
python scripts/jetson_isaaclab_command.py "default"
```

For bimanual mode, a command without `left` or `right` uses `CONTROL_ARM`.
Default is `CONTROL_ARM=right`, so this moves the right arm in Isaac:

```bash
python scripts/jetson_isaaclab_command.py "joint 1 10 deg"
```

Both arms can be set with 14 degree values:

```bash
python scripts/jetson_isaaclab_command.py "whole body 0 0 0 0 0 0 0 0 10 -10 20 0 0 0"
```

The first 7 values are left joints. The next 7 values are right joints.

## Jetson: Send Real Gripper Command And Mirror It

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python scripts/open_gripper_small.py \
  --port can0 \
  --side right \
  --target-deg -10 \
  --i-am-at-robot \
  --yes
```

This sends the same target to Isaac:

```text
gripper target -10.000 deg
```

Mapping:

```text
0 deg   -> 0.000 m/finger closed
-65 deg -> 0.044 m/finger fully open
-10 deg -> about 0.0068 m/finger
```

## Manual Test From Jetson

```bash
python scripts/jetson_isaaclab_command.py "joint right 1 10 deg"
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
python scripts/jetson_isaaclab_command.py "gripper target -10.000 deg"
python scripts/jetson_isaaclab_command.py "gripper target 0.000 deg"
python scripts/jetson_isaaclab_command.py "open gripper"
python scripts/jetson_isaaclab_command.py "close gripper"
python scripts/jetson_isaaclab_command.py --status
```

## Options

Default is bimanual robot, right gripper, closed default gripper:

```bash
OPENARM_SETUP=bimanual CONTROL_ARM=right DEFAULT_GRIPPER_DEG=0 bash run_default_openarm_mirror.txt
```

Headless test:

```bash
HEADLESS=1 HOST=127.0.0.1 PORT=18766 bash run_default_openarm_mirror.txt
```

Do not run another server on the same port `8765` at the same time.
