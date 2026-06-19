# Left-Arm Clean-1000 Feasibility Smoke — STOP and Report

Date: 2026-06-19. Simulation only. No real robot, SSH, CAN, replay, or mirror.

## Decision

**STOPPED before full collection.** The single left arm cannot cleanly lift all four
objects in the approved left-centered scene. Per the user instruction ("if the left arm
cannot cleanly lift all four at -3 deg, STOP and report; do not auto-fall back to
routing"), no full collection was started and no fallback was taken.

## Setup

- Scene: `synthetic_smolvla/configs/scene_openarm_real_table_zero_left_centered_v1.yaml`
  (active_arm: left, base `[0.38,0,0.60]`, cardboard box top z=0.73 m, 20 cm robot gap,
  object row shifted +0.16 m toward the left arm, camera 256x256, USER APPROVED).
- Contract: 100 steps (28/24/16/24/8), substeps 5, lift offset 0.05 m, grasp/cap -3 deg,
  jitter x±0.005 / y±0.003.
- Smoke: `--num-envs 8 --rounds 6` (48 source attempts), balanced target weights.
- Manifest: `synthetic_smolvla/reports/clean1000_smoke_balanced_left_manifest.jsonl`.
- Sample frames: `synthetic_smolvla/reports/clean1000_smoke_balanced_left_samples/`.

## Result (48 source attempts)

| Target | y (m) | n | lift success | kept clean | sole reject reason | max action-clip deg (min/mean/max) |
|---|---:|---:|---:|---:|---|---|
| orange_ball | -0.08 | 7 | 0 | 0 | unreachable (+1 wrong-object) | 11 / 91 / 209 |
| red_cube | +0.08 | 11 | 11 | 0 | refined_action_clip | 6 / 105 / 460 |
| green_cube | +0.24 | 18 | 18 | 1 | refined_action_clip | 0.7 / 147 / 447 |
| blue_cube | +0.40 | 12 | 0 | 0 | unreachable | 25 / 85 / 431 |

Only **1 of 48** episodes was truly clean (a green_cube lift with 0.74 deg clip).

## Findings

1. **Two middle objects (red +0.08, green +0.24) DO lift** — 100% lift success, rise
   0.042–0.049 m — but are rejected because the oracle IK requires **massive joint-limit
   clipping** (mean ~105–147 deg, peaks ~460 deg of single-step IK desired vs the safe
   clamp). These are not clean trajectories; the IK is thrashing against limits and the
   object happens to rise.
2. **Two edge objects (orange -0.08, blue +0.40) do not lift at all** (rise 0.0). They sit
   outside the left arm's reachable lateral window. The left arm's clean window is roughly
   +0.08…+0.24 m (~0.16 m wide); the four-object row spans 0.48 m, so no single re-centering
   shift can fit all four.
3. **The new `refined_action_clip` check is working and is revealing a deeper problem.**
   The legacy `limit_exceeded` flag was true on ~100% of the OLD "clean" routed datasets,
   which (combined with this smoke) indicates those datasets were also full of IK
   joint-limit thrashing — a plausible root cause of why the previously trained SmolVLA
   produces 0 lifts. The old pipeline simply never measured or rejected the clipping.
4. The recorded (clamped) actions are always within safe limits, so nothing unsafe is
   commanded — but the underlying oracle IK is diverging (large single-step DLS updates,
   consistent with near-singular / edge-of-envelope targets and the all-zero reset being a
   poor IK seed).

## Root cause

The all-zero reset pose is a poor IK seed and the left arm's clean (non-singular,
within-limit) reachable envelope is far narrower than the object row. The DLS oracle IK
produces out-of-limit single-step solutions that get clamped, so most "successful" lifts
are not clean. This is a data-generation (oracle IK) quality problem, not just a camera or
centering problem.

## Options for the user

1. **Improve the oracle IK so trajectories stay within joint limits** (better seed than
   all-zeros for the IK only — keep the recorded start at zeros — more DLS damping, smaller
   per-step updates, more iterations, or joint-limit-aware IK), then re-run this smoke.
   Best addresses the root cause and would help every object/arm. Recommended first step.
2. **Routing** (right arm → orange/red, left arm → green/blue). Reaches all four colors,
   but the same IK-thrash problem likely affects routed data too, so it should be combined
   with option 1.
3. **Tighten the object layout** so all four fit the ~0.16 m clean window (reduce spacing).
   Changes the lanes/spacing and makes colors harder to distinguish; needs approval.
4. **Bring objects much closer** (smaller gap than 20 cm) to widen the clean window, then
   re-test — only helps the reach edges, not the IK thrash.
5. **Relax the clip threshold** to accept moderate clipping. Pragmatic but trains on
   partially-thrashing trajectories — i.e. repeats what the failed model did. Not advised.

No data was collected for training. Awaiting user direction.
