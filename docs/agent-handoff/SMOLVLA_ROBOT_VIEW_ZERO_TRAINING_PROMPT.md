# Prompt For Next Agent: Train Robot-View Zero-Pose SmolVLA

Copy this prompt into the next agent.

```text
You are continuing /home/chyanin/Desktop/realrobot.

Goal:
Train and validate a new SmolVLA model for the OpenArm Isaac scene using the
measured user table layout, approximate robot-view camera, and all-zero robot
start pose. This is SIMULATION ONLY. Do not move or connect to the real robot.
Do not read or expose real robot credentials.

Read first:
1. README.md
2. docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md
3. synthetic_smolvla/reports/dense_isaac_camera_v1_dataset.md
4. synthetic_smolvla/scripts/collect_dense_isaac_dataset.py
5. synthetic_smolvla/scripts/verify_dense_dataset.py
6. synthetic_smolvla/scripts/train_smolvla.py
7. synthetic_smolvla/scripts/eval_vla_isaac.py
8. synthetic_smolvla/scripts/sim_contract.py

New scene/configs already prepared:
- Human-view scene:
  synthetic_smolvla/configs/scene_openarm_user_table_robot_view.yaml
- Training scene, square camera, all-zero reset pose:
  synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml
- Dataset spec:
  synthetic_smolvla/configs/dataset_robot_view_zero_v1.yaml
- Train spec:
  synthetic_smolvla/configs/train_real_table_zero_v1.yaml

Important previous training recipe:
- Previous good dataset:
  synthetic_smolvla/datasets/openarm_dense_isaac_camera_v1
- Previous good checkpoint:
  synthetic_smolvla/checkpoints/smolvla_openarm_dense_isaac_camera_v1/checkpoints/015000/pretrained_model
- Previous model was finetuned from lerobot/smolvla_base.
- Previous dense dataset format:
  - fps = 10
  - 50 steps per episode
  - observation.images.camera1 = real Isaac camera tensor
  - observation.state = measured 7 active-arm joints + gripper, degrees
  - action = commanded clamped IK target, degrees
  - only successful correct-object lifts are kept
  - wrong-object lifts are rejected
  - object-object collision/contact episodes are rejected
  - gripper/table collision, scrape, or penetration episodes are rejected
  - gripper close command is capped at -3 deg for collection and eval
- Training used:
  - batch_size 1
  - steps 20000
  - save_freq 5000
  - policy_path lerobot/smolvla_base
  - n_action_steps 10
  - chunk_size 50, but do not override chunk_size when finetuning from smolvla_base
- New robot-view zero-pose dataset format requested by user:
  - fps = 10 metadata
  - 100 recorded frames per episode
  - phase schedule: 28 approach, 24 descend, 16 close, 24 lift, 8 hold
  - movement sampling: 10 physics substeps per recorded frame
  - lift waypoint is only 5 cm above grasp waypoint (`--lift-offset-m 0.05`)
  - keep finetune chunk_size at 50 because smolvla_base action-head shapes expect it

Hard requirements:
1. Keep this simulation-only. No CAN, no SSH, no real robot motion.
2. Confirm the new camera frames show the object row from the robot-view camera.
   Make the robot-view camera as close to the real robot camera as practical.
   If camera placement changes materially, render PNGs again before collecting.
3. Confirm the robot base and camera are centered between red and green:
   - red_cube y = -0.08
   - green_cube y = +0.08
   - camera eye y = 0.0
   - camera target y = 0.0
4. Preserve all-zero reset pose in the training scene:
   active-arm joints 1-7 = 0.0 deg, gripper = 0.0 deg.
5. Do a small smoke collection first. Do not launch full training until:
   - LeRobot dataset loads
   - images are non-static across an episode
   - state and action shapes are both (8,)
   - episode length is 100 frames
   - at least some correct-object lift successes exist
   - sample frames look correct
6. If the all-zero/measured-layout IK cannot lift objects, stop and write an
   audit note explaining the failure. Do not blindly train a zero-success dataset.
7. Use only one active arm for training. Latest user instruction prefers the
   left arm with the right arm locked/passive. If the left-arm version cannot
   produce clean successful lifts, write an audit before falling back.
8. Train only on clean episodes:
   - reject object-object collision/contact
   - reject object sweeps, drags, or large table-surface slides before lift
   - reject gripper/finger/table contact, scraping, or penetration
   - reject wrong-object lifts
9. Gripper close command must not close past -3 deg. If -3 deg cannot grip
   anything in smoke collection, the user allows trying -2 deg, but document
   the change before recollecting.
10. Lift only 5 cm above the grasp waypoint. Do not train on earlier 15 cm lift
    datasets after this instruction.
11. Use the existing robot joint-limit reference/contract for all dataset
    actions and eval clamps:
    docs/reference/OPENARM_JOINT_LIMITS.md and synthetic_smolvla/scripts/sim_contract.py.
12. If storage is insufficient, old generated datasets/checkpoints may be
    deleted, but do not delete source code or docs; note what was removed.

Suggested workflow:

Step 0 - make sure GPU is free:
```bash
cd /home/chyanin/Desktop/realrobot
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

Step 1 - render/check the new robot-view training camera:
```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --headless \
  --device cuda:0 \
  --steps 120 \
  --manifest synthetic_smolvla/reports/real_table_zero_v1_camera_manifest.json \
  --save-camera-rgb synthetic_smolvla/reports/real_table_zero_v1_camera.ppm
```

Convert/view:
```bash
python3 - <<'PY'
from PIL import Image
Image.open("synthetic_smolvla/reports/real_table_zero_v1_camera.ppm").save(
    "synthetic_smolvla/reports/real_table_zero_v1_camera.png"
)
PY
```

Step 2 - smoke collect a tiny dataset:
```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/collect_dense_isaac_dataset.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --dataset-root synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke \
  --repo-id local/openarm_real_table_zero_v1_smoke \
  --num-envs 4 \
  --rounds 2 \
  --seed 22000 \
  --target-weights 1.6,1,1,1 \
  --approach-steps 28 \
  --descend-steps 24 \
  --close-steps 16 \
  --lift-steps 24 \
  --hold-steps 8 \
  --substeps 10 \
  --lift-offset-m 0.05 \
  --fps 10 \
  --grasp-close-deg -3.0 \
  --max-gripper-close-deg -3.0 \
  --manifest synthetic_smolvla/reports/real_table_zero_v1_smoke_manifest.jsonl \
  --report synthetic_smolvla/reports/real_table_zero_v1_smoke_dataset.md \
  --sample-frame-dir synthetic_smolvla/reports/real_table_zero_v1_smoke_samples \
  --device cuda:0 \
  --overwrite
```

Step 3 - verify the smoke dataset:
```bash
conda run --no-capture-output -n env_isaaclab python \
  synthetic_smolvla/scripts/verify_dense_dataset.py \
  --root synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke \
  --repo-id local/openarm_real_table_zero_v1_smoke
```

Also inspect:
```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("synthetic_smolvla/reports/real_table_zero_v1_smoke_manifest.jsonl")
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
print("source", len(rows))
print("kept", sum(r["kept"] for r in rows))
print("success", sum(r["success_label"] for r in rows))
print("wrong", sum(r["wrong_object_lifted"] for r in rows))
print("targets", {t: sum(r["target_object"] == t for r in rows) for t in sorted({r["target_object"] for r in rows})})
if rows:
    print("episode_len", rows[0]["episode_len"])
PY
```

If smoke has zero kept successes, stop. Write:
synthetic_smolvla/reports/real_table_zero_v1_failure_audit.md
with the command, logs, sample frames, and the safest next fix.

Step 4 - full dataset collection only after smoke passes:
```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/collect_dense_isaac_dataset.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --dataset-root synthetic_smolvla/datasets/openarm_real_table_zero_v1 \
  --repo-id local/openarm_real_table_zero_v1 \
  --num-envs 16 \
  --rounds 130 \
  --seed 22000 \
  --target-weights 1.6,1,1,1 \
  --approach-steps 28 \
  --descend-steps 24 \
  --close-steps 16 \
  --lift-steps 24 \
  --hold-steps 8 \
  --substeps 10 \
  --lift-offset-m 0.05 \
  --fps 10 \
  --grasp-close-deg -3.0 \
  --max-gripper-close-deg -3.0 \
  --manifest synthetic_smolvla/reports/real_table_zero_v1_manifest.jsonl \
  --report synthetic_smolvla/reports/real_table_zero_v1_dataset.md \
  --sample-frame-dir synthetic_smolvla/reports/real_table_zero_v1_samples \
  --device cuda:0 \
  --overwrite
```

Step 5 - verify full dataset before training:
```bash
conda run --no-capture-output -n env_isaaclab python \
  synthetic_smolvla/scripts/verify_dense_dataset.py \
  --root synthetic_smolvla/datasets/openarm_real_table_zero_v1 \
  --repo-id local/openarm_real_table_zero_v1
```

Step 6 - prepare a train command:
```bash
conda run --no-capture-output -n env_isaaclab python synthetic_smolvla/scripts/train_smolvla.py \
  --train-config synthetic_smolvla/configs/train_real_table_zero_v1.yaml \
  --batch-size 1 \
  --steps 20000 \
  --overwrite-output-dir \
  --output synthetic_smolvla/reports/train_real_table_zero_v1_preflight.json \
  --command-output synthetic_smolvla/reports/train_real_table_zero_v1.sh
```

Step 7 - train:
```bash
conda run --no-capture-output -n env_isaaclab python synthetic_smolvla/scripts/train_smolvla.py \
  --train-config synthetic_smolvla/configs/train_real_table_zero_v1.yaml \
  --batch-size 1 \
  --steps 20000 \
  --run \
  --overwrite-output-dir \
  --output synthetic_smolvla/reports/train_real_table_zero_v1_full.json \
  --command-output synthetic_smolvla/reports/train_real_table_zero_v1_full.sh
```

Step 8 - evaluate latest complete checkpoint. Prefer 020000 if it exists,
otherwise use 015000 or the latest complete checkpoint:
```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/eval_vla_isaac.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --checkpoint synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_v1/checkpoints/020000/pretrained_model \
  --trials 20 \
  --steps-per-trial 100 \
  --headless \
  --device cuda:0 \
  --max-gripper-close-deg -3.0 \
  --record-action-trace \
  --output-jsonl synthetic_smolvla/reports/real_table_zero_v1_ckpt020000_eval20_trace.jsonl \
  --output-md synthetic_smolvla/reports/real_table_zero_v1_ckpt020000_eval20.md
```

If eval succeeds, run 100 trials:
```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/eval_vla_isaac.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --checkpoint synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_v1/checkpoints/020000/pretrained_model \
  --trials 100 \
  --steps-per-trial 100 \
  --headless \
  --device cuda:0 \
  --max-gripper-close-deg -3.0 \
  --output-jsonl synthetic_smolvla/reports/real_table_zero_v1_ckpt020000_eval100.jsonl \
  --output-md synthetic_smolvla/reports/real_table_zero_v1_ckpt020000_eval100.md
```

Deliverables:
- New dataset root and dataset report.
- Sample camera frames proving correct robot-view camera.
- Training report and final checkpoint path.
- Eval 20 and eval 100 reports.
- If anything fails, an audit markdown with command, traceback/log tail, and the
  safest next fix.
```
