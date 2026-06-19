# Handoff: Clean-1000 Single-Left-Arm Dataset — implement prepose-before-IK, then collect

Date: 2026-06-19. **Simulation only.** No real robot, SSH, CAN, replay, or mirror.
Do not train until the user confirms.

## Goal

Recollect exactly **1000 clean** Isaac episodes (250 each of orange_ball, red_cube,
green_cube, blue_cube) for the real-table zero-pose scene, using a **single LEFT arm**,
with strict validity checks, then a markdown + JSON audit. Stop after the dataset passes
audit. Do **not** train.

## Locked user decisions

- Single LEFT arm for all 1000 (not mixed-arm routing).
- Re-center by **moving the object row toward the left arm** (rigid shift; order/identities
  preserved: orange < red < green < blue in +y; NOT a lane swap).
- **Tighten the object layout** so all four fit the left arm's clean window.
- **Give the arm a ready pose BEFORE inverse kinematics** (the key fix — see below).
- Dataset root name exactly: `synthetic_smolvla/datasets/openarm_real_table_zero_v2_clean1000_orange_red_smalljitter_v1`.
- If the left arm still cannot cleanly lift all four at -3 deg after this, **STOP and report**
  (no auto-fallback to routing).
- Real setup: cardboard Dell box ~73 cm tall, ~20 cm robot gap.

## Episode contract (must hold for every kept episode)

100 control steps (approach 28 / descend 24 / close 16 / lift 24 / hold 8), exactly 5 sim
substeps/step, lift waypoint 0.05 m above grasp, gripper close cap **-3 deg** (never closes
past it), state/action shape **[8]**, camera real Isaac robot-view **256x256**, small jitter
x±0.005 / y±0.003. Keep only: success + target_rise ≥ 0.04 m + no wrong-object + no
object-object collision + no object sweep/slide + no finger/table penetration + no
object-pushed-down + no refined action-clip. Discard and recollect bad episodes; never patch.

## What is DONE this session (verified)

1. **Collector** `synthetic_smolvla/scripts/collect_dense_isaac_dataset.py` (compiles):
   - `--jitter-x-m`/`--jitter-y-m` (defaults 0.005/0.003); hardcoded jitter removed.
   - Module-level pure helpers (unit-tested): `finger_table_penetration`, `object_pushed_down`,
     `refined_action_clip`.
   - New CLI: `--finger-table-margin-m 0.003`, `--object-pushdown-margin-m 0.005`,
     `--action-clip-tol-deg 1.0`, `--joint4-startup-tol-deg 3.0`.
   - Finger body ids via `robot.find_bodies("openarm_{side}_.*finger")` (fallback `_hand`);
     `joint4_idx`.
   - `finger_table_state()` + `object_pushdown_state()` (torch); `ik_solve_clamped()` returns
     `(jclamped, hit, jdes)`; in-loop refined-clip + finger + pushdown accumulation.
   - New manifest fields: `min_finger_table_clearance_m`, `tabletop_penetration`,
     `object_pushed_down`, `refined_action_clip`, `max_refined_action_clip_deg`,
     `finger_body_names`, `substeps`, `jitter_x_m`, `jitter_y_m`, `gripper_cmd_min_deg`,
     `gripper_cmd_max_deg`. New rejects on the three authoritative flags (NOT on legacy
     `limit_exceeded`; keep `--drop-limit-exceeded` OFF). New counters/report rows/JSON.
2. **Unit test** `tests/test_clean1000_safety_checks.py` — 12 pass (`pytest` it; no GPU).
3. **Audit script** `synthetic_smolvla/scripts/audit_clean1000_dataset.py` — tested (passes
   clean data, fails on penetration / gripper-past-cap / refined-clip; ignores legacy
   limit_exceeded). Emits md + JSON gates.
4. **make_scene.py** — dome light now config-driven via `appearance.lighting.intensity`
   (default 3000) and `.color`. Lowering it lets a tinted surface read its color (a bright
   3000 dome overexposes the box to white).
5. **Scene configs**:
   - `scene_openarm_real_table_zero_left_centered_v1.yaml` — APPROVED by user (render). Left
     arm, box top z=0.73, 20 cm gap (objects x=0.58), row +0.16, cardboard color, lighting 1100.
   - `scene_openarm_real_table_zero_left_centered_tight_v1.yaml` — **USE THIS ONE**. Same as
     approved but objects clustered to ~0.055 m spacing within the demonstrated reachable band
     (y = 0.078 / 0.133 / 0.188 / 0.243). Validated; NOT yet rendered or smoked.
6. **Feasibility smoke** on the non-tight config FAILED (see below). Report:
   `synthetic_smolvla/reports/openarm_real_table_zero_v2_clean1000_left_feasibility_audit.md`.

## The problem the next agent must fix (root cause)

Smoke (48 attempts, `..._left_centered_v1`): only **1/48** truly clean. red_cube(+0.08) and
green_cube(+0.24) lifted 100% but the oracle DLS IK clipped joint targets by **mean ~105-147
deg, peaks ~460 deg** (`refined_action_clip`), so they are not clean. orange(-0.08)/blue(+0.40)
were out of reach (0 lift). The DLS IK starts from the all-zero reset (a poor/near-singular
seed) and diverges. The OLD "clean" datasets had `limit_exceeded` true on ~100% of kept
episodes → they were almost certainly thrashy too → likely why the prior SmolVLA gives 0 lifts.

**User's directed fix: "have a pose before going to inverse kinematics."** Give the arm a good
ready pose first, so IK is seeded well instead of from all-zeros.

## TASK 1 — Implement `--prepose-to-ready` in the collector

Add a flag (default off for back-compat). When on, per round, AFTER the object reset/settle and
AFTER computing the `above`/`descend`/`lift` waypoints (search for `phase_plan`,
`set_ik_command`, and the `for phase, n_steps in phase_plan:` loop):

1. **Warmup (NOT recorded):** capture the all-zero reset arm config `reset_arm =
   robot.data.joint_pos[:, ent.joint_ids].clone()`. Then `set_ik_command(above)` and run
   ~`--prepose-warmup-steps` (default 120) iterations of `ik_solve_clamped()` + apply +
   `step_phys(small)` so the arm physically converges above the object. Capture
   `ready_arm = robot.data.joint_pos[:, ent.joint_ids].clone()`.
2. **Reset arm to all-zeros** (`robot.write_joint_state_to_sim(default_pos, default_vel)`,
   `robot.reset()`), re-`set_gripper(open_m)`, `step_phys(settle)`. Re-place objects if the
   reset disturbed them (objects are separate assets; usually fine, but re-assert object poses
   to be safe). Re-read `baseline`/waypoints if needed (object poses unchanged).
3. **Record episode:** in the phase loop, make the **approach** phase a clean joint
   interpolation from `reset_arm` to `ready_arm` (NO IK, so no clip): for step k of n,
   `jclamped = reset_arm + (ready_arm - reset_arm) * (k+1)/n`; set `last_arm = jclamped`; do
   NOT touch `refined_clip_hit` in approach. descend/close/lift/hold stay as-is — descend/lift
   IK now seeds from `ready_arm` (good), so it should not thrash. Keep gripper schedule unchanged.

Notes: ready_arm is within safe limits (it's the clamped IK result). The interpolation passes
joint_4 from 0 up through its 2 deg floor early — that's the same allowed zero-start as today;
do not special-clamp it. Record the actual commanded interpolation as the action. Add
`--prepose-warmup-steps` (int, default 120). Keep episode length exactly 100 (approach stays 28;
prepose IS the approach phase, just interpolation instead of IK).

## TASK 2 — Feasibility smoke (gate)

```
/home/chyanin/IsaacLab/isaaclab_python.sh synthetic_smolvla/scripts/collect_dense_isaac_dataset.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_left_centered_tight_v1.yaml \
  --target-weights "1,1,1,1" \
  --dataset-root synthetic_smolvla/datasets/clean1000_smoke_tight_prepose \
  --repo-id local/clean1000_smoke_tight_prepose \
  --manifest synthetic_smolvla/reports/clean1000_smoke_tight_prepose_manifest.jsonl \
  --report synthetic_smolvla/reports/clean1000_smoke_tight_prepose_collect.md \
  --sample-frame-dir synthetic_smolvla/reports/clean1000_smoke_tight_prepose_samples \
  --num-envs 8 --rounds 8 --seed 31000 \
  --substeps 5 --approach-steps 28 --descend-steps 24 --close-steps 16 --lift-steps 24 --hold-steps 8 \
  --lift-offset-m 0.05 --grasp-close-deg -3.0 --max-gripper-close-deg -3.0 \
  --jitter-x-m 0.005 --jitter-y-m 0.003 --overwrite --device cuda:0 --prepose-to-ready
```
Then analyze the manifest per object: lift success, kept, and especially
`refined_action_clip` count + `max_refined_action_clip_deg`. PASS = each of the four objects
yields clean kept lifts and refined-clip is mostly ~0. If still thrashing, raise
`--prepose-warmup-steps`, or move objects fractionally / lower the table, and re-test; if it
cannot be made clean, STOP and report (do not silently relax `--action-clip-tol-deg`).

## TASK 3 — Full collection (only if smoke passes), 250 per object

Run 4 per-object collections on the tight config (so per-target balance is exact) with
`--prepose-to-ready`, `--max-keep 250`, distinct seeds, and the contract flags above. Use
`--target-weights "1,0,0,0"` (orange), `"0,1,0,0"` (red), `"0,0,1,0"` (green), `"0,0,0,1"`
(blue). Per-object roots `synthetic_smolvla/datasets/clean1000_<obj>_left_v1` and matching
manifests/reports/sample dirs. Size `--num-envs/--rounds` from the smoke kept-rate (budget
extra rounds for the hardest object; rerun with a new seed if a run undershoots 250).

## TASK 4 — Merge to 1000

```
python synthetic_smolvla/scripts/merge_lerobot_datasets.py \
  --input synthetic_smolvla/datasets/clean1000_orange_left_v1 \
  --input synthetic_smolvla/datasets/clean1000_red_left_v1 \
  --input synthetic_smolvla/datasets/clean1000_green_left_v1 \
  --input synthetic_smolvla/datasets/clean1000_blue_left_v1 \
  --output-root synthetic_smolvla/datasets/openarm_real_table_zero_v2_clean1000_orange_red_smalljitter_v1 \
  --repo-id local/openarm_real_table_zero_v2_clean1000_orange_red_smalljitter_v1 \
  --fps 10 --max-total-episodes 1000 --overwrite \
  --report synthetic_smolvla/reports/openarm_real_table_zero_v2_clean1000_merge.json
```

## TASK 5 — Audit + verify (required outputs)

```
python synthetic_smolvla/scripts/audit_clean1000_dataset.py \
  --manifest synthetic_smolvla/reports/clean1000_orange_left_v1_manifest.jsonl \
  --manifest synthetic_smolvla/reports/clean1000_red_left_v1_manifest.jsonl \
  --manifest synthetic_smolvla/reports/clean1000_green_left_v1_manifest.jsonl \
  --manifest synthetic_smolvla/reports/clean1000_blue_left_v1_manifest.jsonl \
  --dataset-root synthetic_smolvla/datasets/openarm_real_table_zero_v2_clean1000_orange_red_smalljitter_v1 \
  --repo-id local/openarm_real_table_zero_v2_clean1000_orange_red_smalljitter_v1 \
  --target-counts "orange_ball=250,red_cube=250,green_cube=250,blue_cube=250" \
  --out-md synthetic_smolvla/reports/openarm_real_table_zero_v2_clean1000_audit.md \
  --out-json synthetic_smolvla/reports/openarm_real_table_zero_v2_clean1000_audit.json
```
Also run `verify_dense_dataset.py --root <merged> --repo-id <merged>`. All gates must pass
(per-target 250, target_rise ≥0.04, penetration=0, pushdown=0, refined-clip=0, gripper ≤ -3,
100-step, 5-substep, [8] shapes, 256x256, sample frames). Then STOP and report. Do not train.

## Useful facts / gotchas

- Render PPMs land under `/home/chyanin/IsaacLab/synthetic_smolvla/reports/` (isaaclab cwd);
  convert with PIL to a PNG in the repo reports dir.
- Robot is fixed-base, bimanual; left arm is the +y arm; finger bodies
  `openarm_left_{left,right}_finger`, hand `openarm_left_hand`, tcp `openarm_left_ee_tcp`.
- `--max-keep` is a global kept cap; with single-object `--target-weights` it caps that object
  at exactly N. Other objects still spawn as distractors (needed for wrong-object checks).
- Check GPU free first: `nvidia-smi --query-compute-apps=pid,process_name,used_memory
  --format=csv,noheader,nounits` and `ps -eo pid,cmd | grep -E "collect_dense|make_scene|kit"`.
- Memory note saved at the project memory dir: `clean1000-left-arm-ik-thrash.md`.
- Plan file: `/home/chyanin/.claude/plans/work-in-home-chyanin-desktop-realrobot-s-wild-platypus.md`.

## Read first (next agent)

1. This file.
2. `synthetic_smolvla/reports/openarm_real_table_zero_v2_clean1000_left_feasibility_audit.md`
3. `synthetic_smolvla/scripts/collect_dense_isaac_dataset.py` (esp. the round loop + phase loop)
4. `synthetic_smolvla/configs/scene_openarm_real_table_zero_left_centered_tight_v1.yaml`
5. `synthetic_smolvla/scripts/sim_contract.py`, `audit_clean1000_dataset.py`, `merge_lerobot_datasets.py`
