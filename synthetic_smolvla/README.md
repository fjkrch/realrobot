# Synthetic SmolVLA OpenArm Pipeline

This folder implements the simulation-only pipeline from
`docs/sim/SMOLVLA_SYNTHETIC_ONLY_PLAN.md`.

It is intentionally separate from the real robot scripts. Nothing here opens a
CAN interface or moves physical motors.

## Status

> **Critical caveat (2026-06-16).** The "100% success" numbers below come from
> the oracle *scaffold*, which writes a hard-coded `success_label: True` without
> touching physics. When the same joint plan was stepped through real Isaac
> physics it picked **nothing** (0/12). A new physics-measured IK oracle now
> picks for real. Fixed object positions: 5/8 (cubes 5/6). Randomized 100/object
> (400 episodes, seed 4000): **157/400 (39.3%)** — red 48, blue 47, green 42,
> orange_ball 20, **0 wrong-object lifts**. Success is position-sensitive (the
> limit clamp fires on 97% of episodes). See
> [Real IK Oracle (physics-measured)](#real-ik-oracle-physics-measured) below.
> Treat the old manifest-based reports as pipeline plumbing checks, not picks.

Implemented and smoke/full validated:

- Safe OpenArm simulation limit contract.
- Four-object Isaac Lab scene with OpenArm, table, fixed RGB camera, and
  orange/red/green/blue objects.
- Language instruction to target-object selection.
- Deterministic oracle action-plan scaffold with limit-checked actions.
- JSONL oracle-manifest generation.
- Local LeRobot dataset export with RGB/state/action/task data.
- Manifest-only evaluation and stress-test report generation.
- Training preflight and generated local-only SmolVLA commands for the
  installed LeRobot version.
- One-step V1 and V2 SmolVLA smoke training in CUDA.

Generated outputs:

- `datasets/openarm_synth_v1`: 1000 episodes, 5000 frames.
- `datasets/openarm_synth_v2`: 5000 episodes, 25000 frames.
- `reports/oracle_acceptance.md`: 100 oracle trials, 100% manifest success.
- `reports/eval_v1.md`: 1000 V1 manifest trials, 100% success.
- `reports/stress_test_v2.md`: 1000 all-objects-visible stress trials,
  100% manifest success.
- `reports/train_v1.sh` and `reports/train_v2.sh`: full training launchers.
- `checkpoints/smolvla_openarm_synth_v1_smoke` and
  `checkpoints/smolvla_openarm_synth_v2_smoke`: one-step smoke checkpoints.

Still pending for learned-policy performance:

- Run the full 3000-step V1 and 10000-step V2 GPU training jobs.
- Evaluate the trained checkpoints in Isaac physics rollouts.

Note: exported RGB frames are deterministic synthetic top-down placeholders
that exercise the LeRobot/SmolVLA stack. The Isaac scene itself renders a real
camera tensor and is ready for replacing the placeholder export path with
captured Isaac camera frames.

## Quick Smoke Test

Validate scene config:

```bash
python3 synthetic_smolvla/scripts/make_scene.py --dry-run
```

Launch the actual headless Isaac scene through the local helper:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/make_scene.py \
  --headless --steps 2 --device cuda:0 \
  --manifest synthetic_smolvla/reports/isaac_scene_manifest.json
```

Generate a small oracle manifest:

```bash
python3 synthetic_smolvla/scripts/collect_oracle_demos.py --episodes 8 --output /tmp/openarm_oracle_demo.jsonl
```

Evaluate that manifest:

```bash
python3 synthetic_smolvla/scripts/eval_smolvla.py --manifest /tmp/openarm_oracle_demo.jsonl --output /tmp/openarm_eval.md
```

Run a small all-objects-visible stress test:

```bash
python3 synthetic_smolvla/scripts/stress_test.py --episodes 16 --output /tmp/openarm_stress.md
```

## Measured Table Layout Scene

User-measured layout scene:

- table height: 75 cm
- table edge/width: 150 cm
- robot height: 125 cm
- robot base: 43 cm
- table-to-robot gap: 30 cm
- calculated horizontal span: 223 cm
- robot base pose and joint reset pose: all zero

Config:

```text
synthetic_smolvla/configs/scene_openarm_user_table_layout_zero_pose.yaml
```

Validate:

```bash
python3 synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_user_table_layout_zero_pose.yaml \
  --dry-run
```

Launch with viewer:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_user_table_layout_zero_pose.yaml \
  --steps 10000
```

Run the full data-generation/preflight pipeline:

```bash
conda run --no-capture-output -n env_isaaclab \
  python synthetic_smolvla/scripts/run_pipeline.py \
  --mode full --use-conda \
  --report synthetic_smolvla/reports/full_pipeline_status.md
```

Run local-only full training:

```bash
synthetic_smolvla/reports/train_v1.sh
synthetic_smolvla/reports/train_v2.sh
```

## Dense Isaac-Camera Dataset (v1) — CURRENT corrected path

This is the corrected dataset that replaces the weak
`openarm_success_filtered_14000` baseline (5 sparse keyframes + static
placeholder images, which trained to low loss but 0 Isaac lifts). The old
baseline is kept as evidence; it is not deleted.

What it fixes: **dense 50-step rollouts**, **real Isaac camera frames per step**
(verified non-static), distinct **measured state** vs **commanded action**, better
target balance, and per-episode pose/rise/contact metadata. Full write-up:
`reports/dense_isaac_camera_v1_dataset.md`.

| Item | Value |
|---|---|
| Dataset | `datasets/openarm_dense_isaac_camera_v1` (`local/openarm_dense_isaac_camera_v1`) |
| Size | 940 kept episodes / 47,000 frames, 0 wrong-object |
| Image | 256x256x3 real scene camera | 
| Collector | `scripts/collect_dense_isaac_dataset.py` |
| Scene | `configs/scene_openarm_dense_isaac_camera_v1.yaml` |
| Train cfg | `configs/train_dense_isaac_camera_v1.yaml` (finetune `lerobot/smolvla_base`) |

Collect (full run; cameras on, so ~16 envs fits the 8 GB GPU — do NOT use the
camera-off oracle's 500 envs here):

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/collect_dense_isaac_dataset.py \
  --rounds 130 --num-envs 16 --target-weights 1.6,1,1,1 --overwrite \
  --report synthetic_smolvla/reports/dense_isaac_camera_v1_dataset.md
```

Verify before training (loader + non-static-frame gate):

```bash
conda run --no-capture-output -n env_isaaclab python \
  synthetic_smolvla/scripts/verify_dense_dataset.py \
  --root synthetic_smolvla/datasets/openarm_dense_isaac_camera_v1 \
  --repo-id local/openarm_dense_isaac_camera_v1
```

Train (regenerates the command, then runs it):

```bash
conda run --no-capture-output -n env_isaaclab python \
  synthetic_smolvla/scripts/train_smolvla.py \
  --train-config synthetic_smolvla/configs/train_dense_isaac_camera_v1.yaml \
  --command-output synthetic_smolvla/reports/train_dense_isaac_camera_v1.sh \
  --overwrite-output-dir
bash synthetic_smolvla/reports/train_dense_isaac_camera_v1.sh
```

LeRobot 0.4.4: finetuning uses the draccus path directive `--policy.path=<id>`
(`=` form, single token) plus `=`-form `--policy.*` overrides; the
space-separated form is rejected. `scripts/train_smolvla.py` emits the `=` form.

Evaluate in Isaac physics (real camera, closed loop) — 20 trials first, 100 only
after nonzero lifts:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/eval_vla_isaac.py --trials 20
```

## Success-Filtered SmolVLA Dataset

The next dataset path uses the physics-measured IK oracle instead of the fake
scaffold labels. It runs all four objects together, records the requested target
per episode, measures whether the target actually lifts, and exports only
successful target-object lifts.

Smoke test the parallel oracle:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/oracle_pick_ik_parallel.py \
  --num-envs 8 --rounds 1 --device cuda:0 \
  --manifest synthetic_smolvla/reports/oracle_parallel_smoke_all.jsonl \
  --success-manifest synthetic_smolvla/reports/oracle_parallel_smoke_success.jsonl \
  --output synthetic_smolvla/reports/oracle_parallel_smoke_eval.md
```

Run the requested 4000 measured episodes:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/oracle_pick_ik_parallel.py \
  --num-envs 500 --rounds 8 --device cuda:0 \
  --manifest synthetic_smolvla/reports/oracle_parallel_all.jsonl \
  --success-manifest synthetic_smolvla/reports/oracle_parallel_success.jsonl \
  --output synthetic_smolvla/reports/oracle_parallel_eval.md
```

Export successes to LeRobot and regenerate the SmolVLA train script:

```bash
conda run --no-capture-output -n env_isaaclab \
  python synthetic_smolvla/scripts/export_success_filtered_dataset.py \
  --dataset-config synthetic_smolvla/configs/dataset_success_filtered.yaml \
  --overwrite --prepare-train \
  --report synthetic_smolvla/reports/success_filtered_dataset.md
```

Train:

```bash
synthetic_smolvla/reports/train_success_filtered.sh
```

Completed 4000-source result: 1799 kept successes, 0 wrong-object lifts.

The follow-up 10000-source run also completed:

```text
10000 source episodes -> 4481 kept successes, 0 wrong-object lifts
dataset: synthetic_smolvla/datasets/openarm_success_filtered_extra10000
report: synthetic_smolvla/reports/success_filtered_extra10000_dataset.md
```

The main training artifact is now the combined 14000-source dataset:

```text
14000 source episodes -> 6280 kept successes, 0 wrong-object lifts
dataset: synthetic_smolvla/datasets/openarm_success_filtered_14000
repo id: local/openarm_success_filtered_14000
report: synthetic_smolvla/reports/success_filtered_14000_dataset.md
```

Train on the combined dataset:

```bash
synthetic_smolvla/reports/train_success_filtered_14000.sh
```

Validation completed: LeRobot loads 6280 episodes / 31400 frames, and a
one-step SmolVLA CUDA smoke run completed successfully. The image feature key
is `observation.images.camera1`.

More detail lives in `docs/sim/SUCCESS_FILTERED_SMOLVLA_VLA_PLAN.md`.

## Real IK Oracle (physics-measured)

This section logs the in-progress rework that replaces the fake oracle with one
that actually picks objects in Isaac physics. It mirrors the task-space state
machine in the `hsi_pregrasp_refusal` Isaac task project
(`/home/chyanin/IsaacLab/source/hsi_pregrasp_refusal/hsi_pregrasp_refusal/state_machine.py`),
which uses differential IK with the phases
`approach-above -> approach -> grasp -> lift`.

### Diagnosis (why the old oracle is fake)

1. `scripts/oracle_policy.py` / `scripts/collect_oracle_demos.py` emit a fixed
   list of joint angles and hard-code `success_label: True`,
   `wrong_object_lifted: False`. The datasets and the `oracle_acceptance.md` /
   `eval_v1.md` / `stress_test_v2.md` "100% success" reports are all derived
   from that asserted label, never from physics.
2. `scripts/rollout_oracle_physics.py` *does* step physics with those joint
   angles. Result (`reports/logs/physics_rollout.log`,
   `reports/physics_rollout_manifest.jsonl`): **0/12 success, every
   `target_rise = +0.0000 m`** — the arm never contacts any object.
3. Root cause is reachability, not gripping. `scripts/probe_openarm_reach.py`
   runs differential IK toward each object and misses by **0.20-0.44 m**.

### Root cause: the arm cannot reach the table

The bimanual OpenArm (`OPENARM_BI_HIGH_PD_CFG`) was spawned with no base
translation, i.e. floor-mounted at the world origin (z=0), gravity disabled.
The table sits at z=0.40 with objects resting at z=0.43. From the floor mount
the right-arm TCP only reaches ~z=0.09, so the table is far out of reach.

Fix applied so far: `configs/scene_openarm_four_objects.yaml` now has a
`robot.base_pose_m` field, wired through `scripts/make_scene.py`
(`_openarm_robot_cfg` sets `init_state.pos`). It currently raises the base to
`[0.0, 0.0, 0.40]`.

### Reach map after raising the base to z=0.40

Re-running the probe (`reports/openarm_reach_probe.json`,
`reports/logs/reach_probe_mounted.log`) with the base at (0,0,0.40):

- EE reset pose moves up to `[0.144, -0.251, 0.506]` (now above the table).
- The right arm's only reliably reachable pocket (IK error < 0.03 m) is
  **x≈0.30, y∈[-0.28,-0.08], z≈0.55** — small, shifted to the robot's right
  (-y), close-in (x=0.30), and ~0.15 m above the base.
- At table height (z=0.45) *every* probed point fails (errors 0.11-0.40 m): the
  arm cannot get its TCP low and forward at the same time from this base.
- The four object positions (z=0.45) still miss badly:
  orange 0.20 m, red 0.44 m, green 0.43 m, blue 0.36 m. The TCP ends up high
  (z≈0.65-0.86) — it reaches up/out but not down onto the table.

Conclusion: raising the base was necessary but not sufficient. The reachable
band is ~0.15 m *above* the base and only ~0.30 m forward, while the original
table was +0.03 m above and ~0.45 m forward. So the table + objects were moved
into the pocket (see "Grasp fix" next).

### Grasp fix applied (2026-06-16) -> objects now lift for real

`configs/scene_openarm_four_objects.yaml` was updated to put a smaller table and
all four objects inside the confirmed pocket: objects at **x=0.30,
y in {-0.26,-0.20,-0.14,-0.08}, z=0.55**, table top at z=0.53, base at
(0,0,0.40). The camera target and `workspace_bounds_m` were moved to match.

Running `oracle_pick_ik.py --episodes-per-target 2`
(`reports/oracle_ik_manifest.jsonl`, `reports/oracle_ik_eval.md`,
`reports/logs/oracle_ik.log`):

| Object | Success | Measured lift |
|---|---|---|
| red_cube | 1/2 | +0.117 m |
| green_cube | 2/2 | +0.122 / +0.122 m |
| blue_cube | 2/2 | +0.119 / +0.120 m |
| orange_ball | 0/2 | one rolled off the table (-0.53 m) |
| **Total** | **5/8 (62.5%)** | cubes alone 5/6 (83%) |

This is the real fix: the OpenArm now reaches, closes, and lifts objects in
physics (it picked **nothing**, 0/12, before). No wrong-object lifts.

### 100 episodes/object, randomized (the larger run)

`oracle_pick_ik.py --episodes-per-target 100 --randomize --seed 4000` adds a
per-episode object jitter (x +/-0.02 m, y +/-0.01 m, clamped to the workspace
bounds so objects stay in the pocket and do not overlap) so the 400 episodes are
genuinely diverse (verified: distinct EE trajectories every episode, not the ~2
repeating ones the un-randomized run produced). Outputs
`reports/oracle_ik_eval_n100.md`, `reports/oracle_ik_manifest_n100.jsonl`,
`reports/logs/oracle_ik_n100.log`.

| Target | Success / 100 |
|---|---:|
| red_cube | 48 |
| blue_cube | 47 |
| green_cube | 42 |
| orange_ball | 20 |
| **Total** | **157/400 (39.3%)** |

- **0 wrong-object lifts** across all 400.
- The limit clamp engaged on **387/400 (96.8%)** episodes: across the whole
  pocket the IK rides the joint-limit contract, so a small position jitter often
  tips a grasp from success to miss. That sensitivity (not the controller) is
  why cubes fall from 83% at fixed positions to ~42-48% under jitter, and it is
  the strongest signal that the **base placement** needs the lower+advance pass
  (toward `[0.12,-0.12,0.28]`) to center the pocket in the arm's range before
  picks will hold up under randomization.
- Round ball stays weakest (20%): a parallel-jaw close on a 20 mm sphere often
  rolls it out (some episodes show it knocked off the table, rise ~ -0.53 m).

### Optimization attempts (2026-06-16) — both regressed, so reverted

Two quick levers were tried to raise success and reduce the 97% limit-clamp.
Both made it WORSE; the proven config (x=0.30, position-only IK) is kept.

| Variant | Cubes (12-ep check) | Clamp | Verdict |
|---|---|---|---|
| x=0.30, position-only (baseline) | ~83% | 100% | **best, kept** |
| Spawn closer, x=0.24, position-only | 25% (red 3/3, green/blue 0/3) | 83% | regressed |
| x=0.24 + captured top-down wrist lock | 0/12 | 100% | regressed hard |

What this proves:

- **The limit clamp is not the grasp bottleneck.** At x=0.30 the clamp fires on
  ~100% of episodes yet cubes still grasp 83%. Pulling objects in to x=0.24 cut
  the clamp only to 83% but broke green/blue grasps. The clamp is a safety flag
  (IK riding the contract), not the cause of misses.
- **Grasp success is wrist-orientation sensitive.** Position-only IK lets the
  wrist drift to a position-dependent orientation; at x=0.30 it happens to grasp,
  at x=0.24 it does not. So the real lever is a *correct* top-down grasp wrist.
- **But a *captured* orientation is not a *correct* one.** `--orient-lock`
  (now opt-in, default off) calibrates the wrist from a central position-only
  reach and holds it through descend/grasp/lift. That captured quat
  (`[-0.070, 0.816, 0.034, 0.573]`) is a drifted pose, not true top-down, and
  forcing it on every object drove success to 0/12 (`reports/logs/
  oracle_ik_orient_check.log`). A real fix must *derive* the top-down quaternion
  from the TCP frame axes (rotate the gripper approach axis to world -Z), not
  capture whatever the arm drifted to.

Net: the headline numbers remain the x=0.30 position-only run (157/400 = 39.3%
randomized; cubes 5/6 fixed). The genuinely promising next step is deriving the
correct top-down grasp orientation; the scaffolding for an orientation hold is in
place (`--orient-lock`, pose-mode IK controller) and just needs the right quat.

### RL grasp policy (learned oracle) — `rl/`

Rather than keep hand-tuning the wrist, the grasp is now framed as an RL problem:
let PPO **learn** a closed-loop joint policy that reaches, orients, closes, and
lifts the requested object, rewarded by the object's measured height rise. This
sidesteps the orientation problem entirely — the policy discovers the wrist pose.

Files (`synthetic_smolvla/rl/`):

- `openarm_pick_env.py` — `OpenArmPickEnv` (Isaac Lab `DirectRLEnv`), modelled on
  the stock `direct/franka_cabinet` task.
  - **Scene:** bimanual OpenArm (base at z=0.40), table, and the four objects in
    the reachable pocket; objects get a small per-episode xy jitter.
  - **Target-conditioned:** each episode picks a random target; its one-hot is in
    the observation, so the trained policy is language/target conditioned and can
    replace the scripted oracle for SmolVLA data.
  - **Action (8):** 7 right-arm joint deltas (integrated, clamped to soft limits)
    + 1 gripper open/close. Left arm held at rest.
  - **Observation (34):** scaled right-arm joint pos/vel, gripper width, TCP pos,
    each object's position relative to the TCP, and the target one-hot.
  - **Reward:** reach (`1/(1+d^2)` + near bonus) + measured target lift
    (`30x`) + grip-when-near + success bonus, minus wrong-object lift penalty
    (`20x`) and an action penalty. `_get_dones` ends on lift > 0.08 m, on the
    target falling off the table, or timeout.
- `train_rl.py` — rsl_rl PPO trainer (`OnPolicyRunner` + `RslRlVecEnvWrapper`),
  with a `--smoke` tiny mode. Checkpoints + TensorBoard go to
  `synthetic_smolvla/rl/logs/openarm_pick/<timestamp>/`.

Status — **smoke test PASSED** (`reports/logs/rl_smoke.log`): env builds with 16
envs, PPO ran 2 iterations, reward/`success_rate`/`mean_target_rise_m` logging
works, and `model_0.pt`/`model_1.pt` checkpoints were written. The full training
job (below) has NOT been run yet — the smoke run only proves the stack works.

Run the smoke test:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/rl/train_rl.py --headless --smoke --device cuda:0
```

Full training (long GPU job; watch `success_rate` / `mean_target_rise_m` climb):

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/rl/train_rl.py --headless --num-envs 2048 --max-iterations 1500
# TensorBoard: tensorboard --logdir synthetic_smolvla/rl/logs/openarm_pick
```

Remaining work: run full training to convergence; then add a `play_rl.py` that
rolls out the trained checkpoint per target object and writes a measured success
report (compare to the IK oracle's 39.3%); finally, use the trained policy as the
SmolVLA demonstration oracle (replace `oracle_pick_ik.py`'s manifest source) and
regenerate the datasets with honest, RL-quality labels. Reward shaping
(`reach/lift/wrong/grip` scales in `OpenArmPickEnvCfg`) is the main tuning knob if
success stalls. Full design + plan: `docs/sim/OPENARM_RL_GRASP_PLAN.md`.

### Remaining reliability levers (to reach the >95% Phase-2 target)

The grasp works but is not yet rock-solid. Known issues and the knobs to turn:

1. **Round ball (0/2).** A parallel-jaw close on a 20 mm sphere lets it roll
   out. Options: shrink the grasp target to the ball centre with a tighter
   close, add a top-down orientation (see below), or accept lower ball success.
2. **Marginal cube grasp (one red_cube miss).** Full-close (`gripper 0 deg`)
   on a 35 mm cube is a tight, slightly non-deterministic grip. Try closing to
   a width that snugly matches the cube (partial close, e.g. gripper ~-10 deg),
   a small negative `--grasp-offset-m` so the fingers straddle the body, and
   more `--grasp-steps` to settle before lifting.
3. **`limit_exceeded` fired on all 8 episodes.** The IK solution repeatedly
   wanted joint angles outside the safe contract, so the clamp shaped every
   motion. Picks still succeed, but precision would improve with a base pose
   that puts the pocket more centrally in the arm's range (e.g. lower+advance
   the base toward `[0.12,-0.12,0.28]` and re-probe).
4. **Orientation control.** Switch `oracle_pick_ik.py` to
   `command_type="pose"` with a fixed top-down grasp quaternion for repeatable
   approaches once position-only grips plateau.

After tuning to >95%: point the dataset export at the IK oracle's measured
manifest (not the scaffold) and regenerate V1/V2 with honest labels.

### New oracle script

`scripts/oracle_pick_ik.py` is the real, closed-loop oracle:

- Differential IK (`DifferentialIKController`, position mode, DLS) drives the
  active-arm EE TCP (`openarm_right_ee_tcp`) through
  `above object -> descend -> close gripper -> lift`.
- Every per-iteration IK joint solution is converted to degrees, clamped to the
  `sim_contract.py` limit contract, and converted back before it is commanded;
  a `limit_exceeded` flag records when the clamp had to intervene.
- Success is the **measured** target-object rise (> `--lift-threshold-m`), and
  wrong-object lift is measured the same way. No hard-coded labels.
- Outputs `reports/oracle_ik_manifest.jsonl` + `reports/oracle_ik_eval.md`.

Run it:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/oracle_pick_ik.py \
  --headless --device cuda:0 --episodes-per-target 3
```

Re-map reachability any time the base or table moves:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/scripts/probe_openarm_reach.py --device cuda:0
# -> reports/openarm_reach_probe.json (reach_grid + per-object errors)
```

(See "Remaining reliability levers" above for the path to >95%.)

## Safety Boundary

Use Isaac Sim / Isaac Lab only for this project. Do not call
`scripts/move_joint.py`, `scripts/move_arm.py`, `scripts/pick_cube.py --real`,
or Jetson-side gripper commands while working on this synthetic pipeline.
