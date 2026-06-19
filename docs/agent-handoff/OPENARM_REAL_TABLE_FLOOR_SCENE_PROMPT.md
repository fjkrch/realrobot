# Prompt For Next Agent: Match Isaac Scene To Real Table And Floor

Copy this prompt into the next agent.

```text
You are continuing /home/chyanin/Desktop/realrobot.

Goal:
Make the OpenArm Isaac simulation scene visually closer to the real table/floor
setup shown by the user, then validate the robot-view camera image before any
new SmolVLA dataset collection or training.

This is SIMULATION ONLY. Do not move the real robot. Do not SSH. Do not read or
expose real robot credentials.

Real appearance target from user photo:
- White or very light cream tabletop.
- Black/dark table edge/rim around the near side.
- Grey speckled floor, not the default plain grid floor.
- Light wall/background with a dark baseboard/trim near the floor if practical.
- Keep measured geometry:
  - table height 75 cm
  - table size/edge 150 cm
  - gap from table to robot 30 cm
  - robot/base centered at y = 0
  - robot height metadata 125 cm
  - red cube y = -0.08, green cube y = +0.08, so midpoint is y = 0
  - robot-view camera centered at y = 0 and aimed between red and green

Read first:
1. README.md
2. docs/agent-handoff/SMOLVLA_ROBOT_VIEW_ZERO_TRAINING_PROMPT.md
3. synthetic_smolvla/configs/scene_openarm_user_table_robot_view.yaml
4. synthetic_smolvla/configs/scene_openarm_robot_view_zero_train_v1.yaml
5. synthetic_smolvla/configs/dataset_robot_view_zero_v1.yaml
6. synthetic_smolvla/scripts/make_scene.py
7. synthetic_smolvla/scripts/sim_contract.py

What to implement:
1. Keep the existing red-table configs available for comparison. Do not destroy
   them.
2. Create new real-appearance scene configs, for example:
   - synthetic_smolvla/configs/scene_openarm_real_table_robot_view.yaml
     for 1280x720 human/viewer checking.
   - synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml
     for 256x256 square SmolVLA training.
3. Extend synthetic_smolvla/scripts/make_scene.py only as much as needed so the
   scene can render:
   - a white/light tabletop
   - a dark front/edge rim or border on the table
   - a grey speckled or checker/noisy floor material
   - optional wall/background and baseboard, if simple and stable
4. Prefer explicit simple geometry over fragile fancy materials:
   - tabletop cuboid: light surface
   - dark edge/rim cuboids on visible table sides
   - large floor plane/cuboid with grey material
   - optional small dark baseboard cuboid and light wall plane/cuboid
5. Keep object positions, robot pose, workspace bounds, and camera center
   consistent with the current robot-view zero scene unless a visual issue
   proves a small camera adjustment is needed.
6. Training scene must stay square camera resolution [256, 256].
7. Human-view scene can stay [1280, 720].
8. Keep all-zero reset pose:
   right joints 1-7 = 0.0 deg, gripper = 0.0 deg.

Validation commands:

Check GPU/processes first:
```bash
cd /home/chyanin/Desktop/realrobot
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
ps -eo pid,ppid,stat,cmd | rg "make_scene.py|interactive_vla_isaac.py|isaaclab_python|Isaac-Sim|kit" | rg -v "rg" || true
```

Render the human-view scene:
```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_robot_view.yaml \
  --headless \
  --device cuda:0 \
  --steps 120 \
  --manifest synthetic_smolvla/reports/real_table_robot_view_manifest.json \
  --save-camera-rgb synthetic_smolvla/reports/real_table_robot_view.ppm
```

Convert to PNG:
```bash
python3 - <<'PY'
from PIL import Image
Image.open("synthetic_smolvla/reports/real_table_robot_view.ppm").save(
    "synthetic_smolvla/reports/real_table_robot_view.png"
)
PY
```

Render the square training scene:
```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --headless \
  --device cuda:0 \
  --steps 120 \
  --manifest synthetic_smolvla/reports/real_table_zero_train_camera_manifest.json \
  --save-camera-rgb synthetic_smolvla/reports/real_table_zero_train_camera.ppm
```

Convert to PNG:
```bash
python3 - <<'PY'
from PIL import Image
Image.open("synthetic_smolvla/reports/real_table_zero_train_camera.ppm").save(
    "synthetic_smolvla/reports/real_table_zero_train_camera.png"
)
PY
```

Acceptance checks:
- Table is light/white like the photo, not red.
- Near table edge/rim is black/dark.
- Floor is grey and visually closer to the real speckled floor than the default
  Isaac grid.
- Camera view is from robot side, centered between red and green.
- All four objects are visible in the robot-view camera.
- Training camera is square 256x256.
- Scene config still passes validate_scene_config.
- No real robot connection or motion occurs.

After visual validation:
Update docs/agent-handoff/SMOLVLA_ROBOT_VIEW_ZERO_TRAINING_PROMPT.md to point
the next dataset collection to the new real-appearance training scene:
synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml

Do not start full dataset collection or training until the user approves the
rendered PNGs.

Deliverables:
- New scene config files.
- Any scoped make_scene.py changes.
- PNG renders:
  - synthetic_smolvla/reports/real_table_robot_view.png
  - synthetic_smolvla/reports/real_table_zero_train_camera.png
- Short audit note explaining how the scene matches the real photo and what is
  still approximate.
```
