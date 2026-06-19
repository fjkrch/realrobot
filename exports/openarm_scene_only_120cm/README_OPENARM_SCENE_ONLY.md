# OpenArm Scene Only Package

This package contains only the simulation scene launcher and the measured-layout scene config.
It does not include SmolVLA checkpoints, datasets, real robot control scripts, or credentials.

## Scene

- Table height: 75 cm
- Table edge/width: 150 cm
- Robot height metadata: 125 cm
- Robot base: 43 cm
- Table-to-robot gap: 30 cm
- Robot reset pose: all zero joints

## Requirements

The machine that opens this scene must already have IsaacLab installed with the OpenArm IsaacLab asset available.
Use that machine's own `isaaclab_python.sh` path.

## Run

From the unzipped folder:

```bash
cd openarm_scene_only_120cm

/path/to/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_user_table_layout_zero_pose.yaml \
  --device cuda:0 \
  --steps 1000000
```

On Chyanin's machine, the command is:

```bash
cd /home/chyanin/Desktop/realrobot/exports/openarm_scene_only_120cm

/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_user_table_layout_zero_pose.yaml \
  --device cuda:0 \
  --steps 1000000
```

## Validate Without Viewer

```bash
python3 synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_user_table_layout_zero_pose.yaml \
  --dry-run
```
