# OpenArm Depth Staged-Init Collection Handoff

This is the task-specific handoff for the next agent. Read it after
`README.md`, `SUMMARY.md`, and `docs/agent-handoff/README.md`.

## Goal

Collect new sim-only OpenArm photo-clean datasets with RGB plus depth camera
observations, using two measured left-arm staged-init trajectories. Each init
trajectory has three stages. Collect from 120 cm down to 107.5 cm.

Do not run real hardware. Do not use Jetson, CAN, SSH, or real replay. The CSVs
are measured pose references only.

## Required Read Order

1. `README.md`
2. `SUMMARY.md`
3. `docs/agent-handoff/README.md`
4. `docs/sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md`
5. `docs/reference/OPENARM_JOINT_LIMITS.md`
6. `synthetic_smolvla/scripts/collect_dense_isaac_dataset.py`
7. `synthetic_smolvla/scripts/upsample_episodes_slew.py`
8. `synthetic_smolvla/scripts/_run_one_per_height.sh`
9. `synthetic_smolvla/scripts/sim_contract.py`

## Scope

Heights:

- 120 cm -> `h120cm`
- 117.5 cm -> `h117p5cm`
- 115 cm -> `h115cm`
- 112.5 cm -> `h112p5cm`
- 110 cm -> `h110cm`
- 107.5 cm -> `h107p5cm`

Tasks:

- `orange_ball`
- `red_cube`
- `blue_cube`

Target count:

- 50 clean successful episodes per task
- per init trajectory
- per height

Total clean target: `6 heights * 2 init trajectories * 3 tasks * 50 = 1800`
episodes.

`h105cm` is not part of this pass unless the user explicitly asks for it again.

## Measured Staged Init Trajectories

Source CSVs:

- `/home/chayanin/Downloads/joint_positions_2.csv`
- `/home/chayanin/Downloads/joint_positions_3.csv`

The columns are:

`joint_1.pos, joint_2.pos, joint_3.pos, joint_4.pos, joint_5.pos, joint_6.pos, joint_7.pos, gripper.pos`

Treat values as degrees unless a code-level check proves otherwise. Validate the
gripper convention before saving. These CSV gripper values look degree-like
because they are around `-23` to `-25`, not meter-like.

### Init A

From `/home/chayanin/Downloads/joint_positions_2.csv`:

Stage A1:

```text
[6.327583798, -0.295068675, 3.311326236, 3.027186031, -10.130691159, 4.469743996, 2.961615215, -0.010928469]
```

Stage A2:

```text
[35.528454108, -3.486181747, 3.311326236, 42.478960664, -10.152548098, -0.426210308, -22.960714267, -0.010928469]
```

Stage A3:

```text
[68.532431778, -2.524476438, 1.016347657, 110.563325195, -11.026825652, 12.425669739, -41.582826171, -23.660136310]
```

### Init B

From `/home/chayanin/Downloads/joint_positions_3.csv`:

Stage B1:

```text
[7.442287680, -2.524476438, 1.016347657, 8.994130339, -5.999729716, 4.469743996, -1.519057250, -23.660136310]
```

Stage B2:

```text
[42.828671685, -3.661037258, -1.825054394, 55.396411527, -17.824333636, 4.273031546, -0.907062962, -23.638279371]
```

Stage B3:

```text
[65.538031155, -6.480582370, -8.207280540, 88.640815524, -7.464144619, -6.415011554, -64.335899518, -25.605403868]
```

## Collection Contract

Prefer a single new contract for this new depth collection so the dataset family
is internally consistent. If no stronger local instruction is found, use:

- 20 Hz control
- 400 saved commands per episode
- expected duration `400 / 20 = 20.0 sec`
- max speed `30 deg/s`
- max saved command delta `1.5 deg`

If a staged-init transition requires more than the available setup phase, retime
the stages rather than creating a large jump. Do not silently violate the slew
contract.

Suggested phase structure:

- staged init A/B: retimed interpolation through stage 1 -> stage 2 -> stage 3
- approach
- descend
- close
- lift
- hold/freeze until exactly 400 saved commands

The first saved command must be at or very near stage 1 of the selected init
trajectory. The pick/lift behavior should start after stage 3.

## Depth Camera Requirement

The OpenArm USD does not provide a camera location. The scene camera is created
by `synthetic_smolvla/scripts/make_scene.py` from YAML camera fields.

For this dataset, request depth from Isaac camera by adding:

```yaml
data_types: ["rgb", "distance_to_image_plane"]
```

The collector may need code changes to persist depth. Store depth in each NPZ
episode with an explicit key such as `observation.images.depth` or `depth`.
Document the exact key in the final report.

Validate:

- RGB exists and is non-static.
- Depth exists.
- Depth dtype is float32 or safely convertible to float32.
- Depth shape is `[T, H, W]` or `[T, H, W, 1]`.
- Depth is nonempty and not constant across the whole episode.

## Output Organization

Do not overwrite old datasets. Create new roots:

- configs: `synthetic_smolvla/configs/generated_height_sweep_photo_clean_v1_depth_stagedinit`
- datasets: `synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit`
- reports: `synthetic_smolvla/reports/photo_clean_v1_depth_stagedinit`

Use subfolders by height, init, and task:

```text
synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit/h120cm/initA/orange_ball
synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit/h120cm/initA/red_cube
synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit/h120cm/initA/blue_cube
synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit/h120cm/initB/orange_ball
synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit/h120cm/initB/red_cube
synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit/h120cm/initB/blue_cube
```

Repeat the same structure for every height.

## Episode Acceptance Checks

Keep only episodes that satisfy all of these:

- target object is lifted
- no wrong-object lift
- no object collision
- no gripper/table collision
- no tabletop penetration
- no object pushed down
- no object swept/slid
- action max delta obeys the selected fps contract
- max speed obeys the selected fps contract
- RGB and depth exist for every saved command
- command count matches the selected contract
- first command is near stage 1 for the selected init trajectory
- commanded path passes safely through stage 2 and stage 3 before pick/lift

## Repository Hygiene

Current repository state is already dirty with many modified and untracked
files. Do not revert or clean files you did not create. Before editing, run:

```bash
git status --short
```

Separate future work cleanly:

1. Documentation-only handoff changes.
2. Collector/code changes for staged init and depth persistence.
3. Generated configs.
4. Collection reports/manifests.
5. Dataset artifacts, only if the project intentionally tracks them.

Do not make one giant mixed commit. Do not push anything without explicit user
approval. If pushing later, use separate commits or branches for the groups
above so review and rollback are clean.

## Final Report

Write:

`synthetic_smolvla/reports/photo_clean_v1_depth_stagedinit/final_report.md`

For each height/init/task, list:

- output path
- number of clean kept episodes
- fps
- command length
- target object
- init trajectory used
- max command delta
- max speed
- RGB shape
- depth shape and key
- expected duration
- any failures or blockers

## Copy-Paste Prompt For The Next Agent

```text
Work in /home/chayanin/Desktop/Robot/realrobot.

Read docs/agent-handoff/OPENARM_DEPTH_STAGEDINIT_COLLECTION_HANDOFF.md first,
then follow its read order.

Collect new sim-only OpenArm photo-clean RGB+depth datasets from 120 cm down to
107.5 cm using the two measured three-stage left-arm init trajectories from:
- /home/chayanin/Downloads/joint_positions_2.csv
- /home/chayanin/Downloads/joint_positions_3.csv

Tasks are orange_ball, red_cube, and blue_cube only. Collect 50 clean successful
episodes per task, per init trajectory, per height. Total target is 1800 clean
episodes.

Use new output roots only:
- synthetic_smolvla/configs/generated_height_sweep_photo_clean_v1_depth_stagedinit
- synthetic_smolvla/datasets/openarm_photo_clean_v1_depth_stagedinit
- synthetic_smolvla/reports/photo_clean_v1_depth_stagedinit

Do not overwrite old datasets. Do not run real hardware, Jetson, CAN, SSH, or
real replay. Add/request Isaac depth camera output using distance_to_image_plane
and persist RGB plus depth in every saved NPZ. Validate action slew, max speed,
command count, clean lift success, and depth quality. Keep repo changes split
cleanly and do not push without explicit approval.
```
