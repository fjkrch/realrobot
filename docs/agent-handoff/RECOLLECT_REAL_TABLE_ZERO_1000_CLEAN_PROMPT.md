# Next Agent Prompt: Recollect 1000 Clean Real-Table Zero-Pose Episodes

Date written: 2026-06-19

Use this as the prompt for the next agent.

```text
Work in /home/chyanin/Desktop/realrobot.

Simulation only. Do not move the real robot. Do not SSH. Do not open CAN. Do
not run real replay or mirror scripts.

User goal: stop spending GPU time on the current bad SmolVLA model and recollect
clean Isaac simulation data for the real-table/floor zero-pose scene.

Current model result:
- New 1500-episode / 15000-step training run produced checkpoints 003000,
  006000, 009000, 012000, 015000, and last -> 015000.
- Completed fresh eval reports so far:
  - 003000, 100 steps: 0/4, wrong-object lifts 0
  - 003000, 150 steps: 0/4, wrong-object lifts 0
  - 006000, 100 steps: 0/4, wrong-object lifts 0
  - 006000, 150 steps: 0/4, wrong-object lifts 0
  - 009000, 100 steps: 0/4, wrong-object lifts 0
- Do not call this model better. It has no target lifts in current eval.

Immediate task:
1. Audit the current collection path before collecting more data.
2. Fix the collection contract if needed.
3. Recollect exactly 1000 clean successful episodes.
4. Write a clear collection/audit report.
5. Do not train until the 1000-episode dataset passes all checks.

Important data concern:
- Existing retained collection manifests show `limit_exceeded: true` on many
  kept episodes and negative `min_tcp_table_clearance_m` even when
  `gripper_table_collision: false`.
- Treat this as suspicious. Before collecting, determine whether these fields
  are real safety violations or bad diagnostics.
- If they are real violations, fix collection and reject those episodes.
- If they are diagnostics with misleading names, document the proof and add
  correct checks that directly measure finger/gripper/table contact and tabletop
  penetration.

New dataset target:
- Recommended root:
  `synthetic_smolvla/datasets/openarm_real_table_zero_v2_clean1000_balanced_smalljitter_v1`
- Recommended repo id:
  `local/openarm_real_table_zero_v2_clean1000_balanced_smalljitter_v1`
- Collect exactly 1000 retained episodes, not just 1000 source attempts.
- Target mix must be exactly balanced across all four objects:
  - 250 orange_ball episodes
  - 250 red_cube episodes
  - 250 green_cube episodes
  - 250 blue_cube episodes
- Keep all four objects visible in every episode as distractors/non-targets.

Episode timing/control contract:
- Each retained episode must contain 100 VLA thinking/control steps.
- Between each VLA thinking/control step, advance physics with exactly 5
  simulation substeps/upsampling steps before the next observation/action pair.
- Suggested phase schedule, if using `collect_dense_isaac_dataset.py`:
  `--approach-steps 28 --descend-steps 24 --close-steps 16 --lift-steps 24 --hold-steps 8 --substeps 5`.
- One dataset frame should correspond to one VLA thinking/control step:
  observation image, observed 8D state, and commanded 8D action.
- The episode is not successful just because the gripper touches or nudges the
  object. It must complete the requested clean 5 cm lift.

Scene/object rule:
- Do not swap object identities or lanes.
- Keep the same object-task mapping:
  - orange_ball means "pick up the orange ball"
  - red_cube means "pick up the red cube"
  - green_cube means "pick up the green cube"
  - blue_cube means "pick up the blue cube"
- Keep all objects close to their configured starting lanes. Only move them a
  little bit with small jitter.
- Recommended small jitter: x +/- 0.005 m, y +/- 0.003 m.
- Enforce lane/order preservation after jitter. Orange must stay in the orange
  lane, red in red, green in green, blue in blue. No object may cross into
  another object's lane or become ambiguous in camera view.

Episode validity contract:
Retain an episode only if every item below is satisfied. If not, discard the
episode and recollect another. Do not edit, relabel, patch, or "correct" a bad
episode into a good one.

- `kept` must be true.
- `success_label` must be true.
- requested target rise must be >= 0.04 m.
- lift waypoint must be exactly 0.05 m above grasp waypoint.
- `wrong_object_lifted` must be false.
- no wrong-object rise above lift threshold.
- no object-object collision/contact.
- no object sweep, drag, or slide before valid lift.
- no gripper/finger/table contact or collision at any time.
- no tabletop penetration.
- no gripper/object/table collision side effects beyond the intended target
  grasp that produces a clean lift.
- no wrong-object target/task mismatch.
- gripper command must not close past -3 deg.
- do not use -2 deg unless the user explicitly approves a documented fallback.
- use the existing joint-limit contract in `synthetic_smolvla/scripts/sim_contract.py`.
- do not keep episodes that hit unsafe joint limits or require action clipping,
  unless you prove the old `limit_exceeded` flag is not measuring commanded
  action clipping and add a better safety flag.
- state/action shape must be `[8]`.
- camera must be real Isaac robot-view frames at 256x256, matching eval.
- no placeholder/top-down camera unless explicitly marked as a separate debug
  dataset and not used for training.

Required audit outputs:
- Write a new collection report under `synthetic_smolvla/reports`, for example:
  `synthetic_smolvla/reports/openarm_real_table_zero_v2_clean1000_balanced_smalljitter_collect_20260619.md`
- Include:
  - source attempts
  - retained episodes
  - target counts
  - proof that every retained episode has exactly 100 VLA thinking/control steps
  - proof that each VLA step uses exactly 5 simulation substeps before the next
    VLA step
  - target rise min/max/mean
  - max surface sweep before lift
  - object-object min distance
  - gripper/finger/table clearance/contact proof
  - tabletop penetration proof
  - gripper command min/max
  - joint-limit/action-clamp stats
  - state/action/image shape proof
  - sample frame paths
- Also write a machine-readable audit JSON next to the report.

Do not delete retained datasets or checkpoints unless the user explicitly
confirms. You may delete failed partial outputs you created during this new
collection attempt after reporting them.

When the 1000-episode dataset is complete, stop and report. Do not train VLA
until the user confirms the data is correct.
```

## Current Files To Read First

- `docs/agent-handoff/NEXT_PHASE_REAL_TABLE_ZERO_VLA.md`
- `synthetic_smolvla/reports/openarm_real_table_zero_lift5cm_routed_status_current.md`
- `synthetic_smolvla/reports/openarm_real_table_zero_dataset_checkpoint_audit_20260619.md`
- `synthetic_smolvla/scripts/collect_dense_isaac_dataset.py`
- `synthetic_smolvla/scripts/sim_contract.py`
- `synthetic_smolvla/scripts/eval_vla_isaac.py`

## Why This Recollection Is Needed

The model trained from the current surviving clean data is not producing lifts
under fresh fixed orange/red eval. More importantly, the collection metadata has
signals that need to be resolved before another training run:

- retained episodes commonly have `limit_exceeded: true`;
- retained episodes show negative `min_tcp_table_clearance_m`;
- the policy evaluation has zero target rise on completed fresh checks.

The next step is therefore not more fine-tuning. It is a strict recollection of
1000 verified clean balanced episodes with small object jitter and no object
identity swaps.
