# OpenArm Success-Filtered 14k Dataset Audit

Date: 2026-06-17

Dataset root: `synthetic_smolvla/datasets/openarm_success_filtered_14000`

> **Resolved (2026-06-17).** The corrected dataset called for here was built:
> `synthetic_smolvla/datasets/openarm_dense_isaac_camera_v1` (940 dense episodes,
> real Isaac camera frames per step, measured state + commanded action, balanced
> targets, 0 wrong-object). See `dense_isaac_camera_v1_dataset.md`. This 14k
> dataset is kept as the old baseline/evidence, not deleted.

## Verdict

Correct the dataset before doing another long retrain.

The dataset is structurally valid and it contains the success-filtered oracle
episodes it was exported to contain. It is not yet the right dataset for a
robust closed-loop pick-and-lift VLA policy, because the visual observations are
static placeholders and each episode has only five sparse keyframes.

The training loss can go low on this data, but that does not mean the policy has
learned contact-rich pickup behavior in Isaac physics.

## What The Dataset Has

| Check | Result |
|---|---:|
| Episodes in metadata | 6,280 |
| Frames in data parquet | 31,400 |
| Frames per episode | 5 |
| Tasks | 4 |
| FPS | 10 |
| Bad frame sequences | 0 |
| Bad state/action handoff episodes | 0 |
| Source wrong-object labels | 0 |
| Source failed labels inside success manifest | 0 |

Features from `meta/info.json`:

| Feature | Shape | Meaning |
|---|---:|---|
| `observation.images.camera1` | 96 x 96 x 3 | RGB observation |
| `observation.state` | 8 | Seven joints plus gripper |
| `action` | 8 | Seven joint targets plus gripper target |
| `task_index` | 1 | Task ID |

The LeRobot loader also resolves the language instruction. A sample loaded as
`task = "pick up the orange ball"`.

## Target Balance

| Target | Episodes |
|---|---:|
| red_cube | 2,222 |
| blue_cube | 1,769 |
| green_cube | 1,679 |
| orange_ball | 610 |

The orange ball is underrepresented. That matters because it has different
geometry and is already the hardest target class.

## Source Oracle Quality

The success manifest has 6,280 successful target-object lifts from 14,000 source
episodes.

| Source field | Result |
|---|---:|
| Success labels | 6,280 / 6,280 |
| Wrong object lifted | 0 |
| Target rise min | 0.05063 m |
| Target rise max | 0.15421 m |
| Target rise mean | 0.12054 m |
| Episodes with `limit_exceeded` | 6,223 / 6,280 |
| Limit-clamp rate | 99.09% |

The oracle labels are successful, but most successful actions are near joint or
workspace clamp limits. That makes the demonstrations brittle for imitation.

## Main Dataset Gaps

| Gap | Why It Hurts The VLA |
|---|---|
| Five frames per episode | The policy sees only sparse keyframes, not dense approach, contact, close, and lift control. |
| Static image per episode | All 6,280 episodes have one repeated image across all five frames, so the camera does not show arm motion, gripper motion, contact, or object lift. |
| Placeholder renderer | RGB comes from `dataset_export.py`, not Isaac camera tensors. It shows object layout and target emphasis, not the real robot scene. |
| Default SmolVLA horizon | The trained checkpoint used `chunk_size=50` and `n_action_steps=50` on five-frame episodes, so most of the predicted action horizon is padding/extrapolation. |
| Limit-clamped demonstrations | 99.09% of success episodes hit the safety/workspace clamp flag, which leaves little margin for closed-loop error. |
| Target imbalance | Orange ball has only 610 successes versus 2,222 red cube successes. |
| No per-frame contact/rise signal | The LeRobot rows do not contain object height, object pose, contact, or phase labels. |

## Answer

Yes, build a corrected dataset. Do not keep doing long retrains on this exact
dataset and expect the VLA Isaac success rate to improve much.

## Fix Order

1. Export dense physics rollouts instead of five keyframes.
   Record actual joint state, action, RGB, and object pose/rise at each control
   step through approach, descend, close, lift, and hold.

2. Replace placeholder RGB with Isaac camera frames.
   The image should show the robot, gripper, object, contact, and lifted state.

3. Reduce the oracle limit-clamp rate.
   Improve grasp pose, wrist orientation, workspace, or sampling so successful
   demos have margin instead of sitting on clamp boundaries.

4. Balance targets.
   Collect more orange-ball successes or cap the larger classes during training.

5. Match policy horizon to episode structure.
   If the corrected dataset remains keyframe-style, train with
   `chunk_size=5` and `n_action_steps=5`. If the corrected dataset is dense,
   choose the horizon to match the control window.

6. Retrain and evaluate.
   After re-export, run a loader smoke test, train SmolVLA, then evaluate with
   `synthetic_smolvla/scripts/eval_vla_isaac.py`.

