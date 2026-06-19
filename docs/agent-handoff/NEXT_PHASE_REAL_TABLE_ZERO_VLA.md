# Next Phase: Real-Table Scene + Zero-Pose SmolVLA Training

Use this file as the next-agent handoff. It combines the scene-matching work and
the later SmolVLA retraining work into one safe, ordered plan.

## Safety Scope

This phase is **simulation only**.

- Do not move the real robot.
- Do not SSH to the robot.
- Do not open CAN.
- Do not run real replay/mirror scripts.
- Do not read, copy, print, or expose credentials from real-robot docs.
- Do not start full dataset collection or training until the rendered scene
  camera images are approved by the user.

## Additional User Constraints Added After Scene Approval

The user approved continuing, then added stricter requirements. Apply these
before collecting or training data:

- Make the robot-view camera as close to the real robot camera as practical.
  Render PNGs for approval again if the camera placement changes materially.
- Train only on clean episodes:
  - no object-object collision/contact
  - no object being swept, dragged, or slid into a different table position
    before a valid lift
  - no gripper/finger/table contact, scraping, or tabletop penetration
  - no wrong-object lift
- Use one active arm only. Prefer the user's latest instruction: use the left
  arm and keep the right arm locked/passive. If the existing left-arm setup
  cannot produce clean successful lifts, write an audit before falling back.
- Gripper close command should not close past `-3 deg`. If `-3 deg` cannot grip
  any target after a smoke run, the user allows trying `-2 deg`; document that
  change in the audit/report before recollecting.
- Lift only `5 cm` above the grasp waypoint. Use `--lift-offset-m 0.05` for
  collection. Do not train on earlier 15 cm lift datasets after this constraint.
- Use the existing robot joint-limit reference file/contract for all collection,
  training actions, and eval clamps. Do not train outside the robot-safe joint
  limits.
- If storage is insufficient, the user permits deleting old generated datasets
  and checkpoints that are not part of the current approved real-table run.
  Prefer deleting old synthetic/generated artifacts, not source code or docs,
  and note what was removed.

## User Goal

Make Isaac match the real setup more closely, then train a new SmolVLA model.

Real setup appearance from the user photo:

- White or light cream tabletop.
- Black/dark table front edge or rim.
- Grey speckled floor, not the default Isaac grid floor.
- Light wall/background and dark baseboard/trim if practical.
- Table height: 75 cm.
- Table edge/width: 150 cm.
- Gap from table to robot: 30 cm.
- Robot height metadata: 125 cm.
- Robot and camera centered between red and green objects.
- Start pose for training: all-zero robot joints.

## Read First

Read these files before editing:

1. `README.md`
2. `docs/agent-handoff/OPENARM_REAL_TABLE_FLOOR_SCENE_PROMPT.md`
3. `docs/agent-handoff/SMOLVLA_ROBOT_VIEW_ZERO_TRAINING_PROMPT.md`
4. `docs/agent-handoff/SMOLVLA_TRAINING_HANDOFF.md`
5. `synthetic_smolvla/reports/dense_isaac_camera_v1_dataset.md`
6. `synthetic_smolvla/configs/scene_openarm_user_table_robot_view.yaml`
7. `synthetic_smolvla/configs/scene_openarm_robot_view_zero_train_v1.yaml`
8. `synthetic_smolvla/configs/dataset_robot_view_zero_v1.yaml`
9. `synthetic_smolvla/configs/train_robot_view_zero_v1.yaml`
10. `synthetic_smolvla/scripts/make_scene.py`
11. `synthetic_smolvla/scripts/collect_dense_isaac_dataset.py`
12. `synthetic_smolvla/scripts/verify_dense_dataset.py`
13. `synthetic_smolvla/scripts/train_smolvla.py`
14. `synthetic_smolvla/scripts/eval_vla_isaac.py`
15. `synthetic_smolvla/scripts/sim_contract.py`

## Phase 1: Implement Real-Table/Floor Scene

Keep the existing red-table configs for comparison. Create new configs:

- `synthetic_smolvla/configs/scene_openarm_real_table_robot_view.yaml`
- `synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml`

The first is for human/viewer checking, preferably `1280x720`.
The second is for SmolVLA training, and must be square `256x256`.

Implement only scoped changes in `synthetic_smolvla/scripts/make_scene.py`.
Prefer simple robust geometry:

- Light tabletop cuboid.
- Dark front/edge rim cuboids.
- Grey floor material or large grey floor geometry.
- If possible, add speckled/noisy floor appearance.
- Optional light wall and dark baseboard.

Keep this geometry:

- Table size: `[1.50, 1.50, 0.04]` m.
- Table center pose: `[1.265, 0.0, 0.73]` m.
- Robot base pose: `[0.0, 0.0, 0.0]`.
- Object row:
  - orange ball: `[0.68, -0.24, 0.77]`
  - red cube: `[0.68, -0.08, 0.765]`
  - green cube: `[0.68, 0.08, 0.765]`
  - blue cube: `[0.68, 0.24, 0.765]`
- Red/green midpoint is `y = 0.0`.
- Robot-view camera must be centered at `y = 0.0` and aimed near the red/green
  midpoint.

Keep all-zero reset pose:

```yaml
robot:
  reset_pose_deg:
    right:
      joint_1: 0.0
      joint_2: 0.0
      joint_3: 0.0
      joint_4: 0.0
      joint_5: 0.0
      joint_6: 0.0
      joint_7: 0.0
      gripper: 0.0
```

Because `joint_4: 0.0` is outside the stricter reset-pose contract, the scene
may use:

```yaml
allow_out_of_contract_reset_pose: true
```

Command/action clamps must still use `sim_contract.py`.

## Phase 1 Validation

Check GPU/processes:

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

- Table is light/white, not red.
- Near/front table edge is black/dark.
- Floor is grey and closer to the real speckled floor than the Isaac grid.
- Camera view is from the robot side.
- Camera is centered between red and green.
- All four objects are visible.
- Training camera is `256x256`.
- Scene config passes `validate_scene_config`.
- No real robot action occurred.

Stop here and show the PNG paths to the user. Do not continue to dataset/training
until the user approves the scene.

## Phase 2: Update Training Configs To The Approved Scene

After user approval, update:

- `synthetic_smolvla/configs/dataset_robot_view_zero_v1.yaml`
- `synthetic_smolvla/configs/train_robot_view_zero_v1.yaml`
- `docs/agent-handoff/SMOLVLA_ROBOT_VIEW_ZERO_TRAINING_PROMPT.md`

Point dataset collection to:

```text
synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml
```

Dataset requirements:

- `100` recorded frames per episode.
- Phase split:
  - approach: `28`
  - descend: `24`
  - close: `16`
  - lift: `24`
  - hold: `8`
- Movement sampling: `10` physics substeps per recorded frame.
- Lift waypoint: `0.05 m` above the grasp waypoint, per latest user constraint.
- `fps = 10` metadata.
- `observation.images.camera1`: real Isaac camera tensor.
- `observation.state`: measured 7 active-arm joints + gripper, degrees.
- `action`: commanded clamped IK target, degrees.
- Keep only successful correct-object lifts.
- Reject wrong-object lifts.
- Reject any episode with object-object collision/contact during the rollout.
- Reject any episode where the gripper collides with, scrapes, or penetrates the
  table/tabletop during the rollout.
- Reject any episode where an object is swept, dragged, or slid into a
  materially different table position before a valid lift.
- Gripper must not close past `-3 deg` for this training run. Use
  `--grasp-close-deg -3.0` for collection, and make sure eval/action clamping
  also caps the gripper close command at `-3 deg` before using the model.
- Use the active arm and command clamps from `sim_contract.py` /
  `docs/reference/OPENARM_JOINT_LIMITS.md`.

Important: the collector does not automatically read the dataset spec schedule.
Pass the 100-frame schedule explicitly on the CLI.

Training requirements:

- Finetune from `lerobot/smolvla_base`.
- Batch size `1`.
- Train steps `20000`.
- Save every `5000`.
- Keep `chunk_size: 50` when finetuning from `smolvla_base` because the
  pretrained action head expects that shape.
- Use `n_action_steps: 10`.

## Phase 3: Smoke Dataset Collection

Run this only after the user approves the scene image:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/collect_dense_isaac_dataset.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --dataset-root synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke \
  --repo-id local/openarm_real_table_zero_v1_smoke \
  --num-envs 4 \
  --rounds 2 \
  --seed 23000 \
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
  --manifest synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_manifest.jsonl \
  --report synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_dataset.md \
  --sample-frame-dir synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_samples \
  --device cuda:0 \
  --overwrite
```

Verify:

```bash
conda run --no-capture-output -n env_isaaclab python \
  synthetic_smolvla/scripts/verify_dense_dataset.py \
  --root synthetic_smolvla/datasets/openarm_real_table_zero_v1_smoke \
  --repo-id local/openarm_real_table_zero_v1_smoke
```

Inspect smoke manifest:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("synthetic_smolvla/reports/openarm_real_table_zero_v1_smoke_manifest.jsonl")
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

Smoke pass conditions:

- Dataset loads with LeRobot.
- Images move across an episode.
- `observation.state` shape is `(8,)`.
- `action` shape is `(8,)`.
- Episode length is `100`.
- At least some correct-object lifts are kept.
- Wrong-object lifts are `0` or clearly rejected.
- Sample frames visually match the approved scene.

If smoke has zero kept successes, stop. Write:

```text
synthetic_smolvla/reports/openarm_real_table_zero_v1_failure_audit.md
```

Include:

- exact command used
- terminal log or traceback
- sample frame paths
- manifest stats
- safest next fix

Do not train on a zero-success dataset.

## Phase 4: Full Dataset Collection

Only after smoke passes:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/collect_dense_isaac_dataset.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_v1.yaml \
  --dataset-root synthetic_smolvla/datasets/openarm_real_table_zero_v1 \
  --repo-id local/openarm_real_table_zero_v1 \
  --num-envs 16 \
  --rounds 130 \
  --seed 23000 \
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
  --manifest synthetic_smolvla/reports/openarm_real_table_zero_v1_manifest.jsonl \
  --report synthetic_smolvla/reports/openarm_real_table_zero_v1_dataset.md \
  --sample-frame-dir synthetic_smolvla/reports/openarm_real_table_zero_v1_samples \
  --device cuda:0 \
  --overwrite
```

Verify full dataset:

```bash
conda run --no-capture-output -n env_isaaclab python \
  synthetic_smolvla/scripts/verify_dense_dataset.py \
  --root synthetic_smolvla/datasets/openarm_real_table_zero_v1 \
  --repo-id local/openarm_real_table_zero_v1
```

## Phase 5: Train New SmolVLA

Create or update a train config so it points to:

```yaml
training:
  dataset: synthetic_smolvla/datasets/openarm_real_table_zero_v1
  checkpoint_dir: synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_v1
  policy: smolvla
  policy_path: lerobot/smolvla_base
  device: cuda
  batch_size: 1
  steps: 20000
  save_freq: 5000
  chunk_size: 50
  n_action_steps: 10
```

Prepare train command:

```bash
conda run --no-capture-output -n env_isaaclab python synthetic_smolvla/scripts/train_smolvla.py \
  --train-config synthetic_smolvla/configs/train_real_table_zero_v1.yaml \
  --batch-size 1 \
  --steps 20000 \
  --overwrite-output-dir \
  --output synthetic_smolvla/reports/train_real_table_zero_v1_preflight.json \
  --command-output synthetic_smolvla/reports/train_real_table_zero_v1.sh
```

Train:

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

If training fails, inspect the report JSON and write an audit note before retry.

## Phase 6: Evaluate And Iterate Until Model Is Good

Use the latest complete checkpoint. Prefer `020000` if it exists. If not, use
the latest complete checkpoint such as `015000`.

20-trial eval first:

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

If nonzero and promising, run 100 trials:

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

Suggested model-good target:

- Overall success at least similar to or better than the previous dense model
  baseline, preferably `>= 0.70` on 100 trials.
- Wrong-object lift rate `0`.
- Each cube target should have useful success, not only one color.
- If orange ball is weak, document it and either oversample orange or run a
  focused follow-up dataset.

If model is not good:

1. Do not force real robot testing.
2. Write an eval audit:
   `synthetic_smolvla/reports/real_table_zero_v1_eval_audit.md`
3. Include:
   - checkpoint used
   - eval command
   - success by target
   - wrong-object rate
   - sample failures
   - whether actions are moving or static
   - whether camera frames look correct
4. Pick the safest next improvement:
   - more successful kept episodes
   - rebalance target weights
   - improve camera placement
   - adjust object positions only if they remain consistent with real setup
   - improve oracle/IK if zero-pose layout has poor lift rate
5. Recollect/validate/retrain/evaluate again.

## Final Deliverables

- New real-table scene configs.
- Scoped `make_scene.py` changes.
- Rendered PNGs proving the scene view.
- Dataset root and dataset report.
- Verification output for smoke and full datasets.
- Training report and checkpoint path.
- Eval 20 and eval 100 reports.
- Audit notes for any failure or retraining loop.

Do not perform real robot motion in this phase.
