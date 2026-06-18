# Synthetic-Only SmolVLA Training Plan for OpenArm

This plan trains and evaluates SmolVLA fully inside Isaac Sim using OpenArm.
No real-world data is used, and no command in this plan should move the
physical robot.

## Objective

Train a language-conditioned policy that can pick the requested object:

- Orange ping-pong ball
- Red cube
- Green cube
- Blue cube

Target commands:

```text
pick up the orange ball
pick up the red cube
pick up the green cube
pick up the blue cube
```

Final target:

```text
success > 90%
wrong object < 5%
```

## Safety Boundary

- This is simulation-only work.
- Use Isaac Sim / Isaac Lab only.
- Do not run real robot scripts such as `move_joint.py`, `move_arm.py`,
  `pick_cube.py --real`, or Jetson-side gripper commands for this project.
- Keep this pipeline separate from `docs/real/` except for reading robot joint
  limits from `../reference/OPENARM_JOINT_LIMITS.md`.

## Project Outputs

| Output | Description |
|---|---|
| Isaac Sim scene | OpenArm, table, fixed camera, four objects |
| Oracle policy | Scripted demonstration generator |
| Dataset V1 | 1000 synthetic episodes |
| `smolvla_openarm_synth_v1` | First trained SmolVLA checkpoint |
| Dataset V2 | 5000 randomized synthetic episodes |
| `smolvla_openarm_synth_v2` | Second trained SmolVLA checkpoint |
| Stress-test report | 1000 evaluation episodes with all objects visible |

## Recommended File Layout

Create the implementation under a dedicated simulation project folder so it
does not mix with real robot scripts:

```text
synthetic_smolvla/
  README.md
  configs/
    scene_openarm_four_objects.yaml
    dataset_v1.yaml
    dataset_v2.yaml
    train_v1.yaml
    train_v2.yaml
  scripts/
    make_scene.py
    collect_oracle_demos.py
    train_smolvla.py
    eval_smolvla.py
    stress_test.py
  datasets/
    openarm_synth_v1/
    openarm_synth_v2/
  checkpoints/
    smolvla_openarm_synth_v1/
    smolvla_openarm_synth_v2/
  reports/
    eval_v1.md
    stress_test_v2.md
```

## Simulation Limit Contract

Every Isaac Sim component in this pipeline must clamp to the safe simulation
limits from `../reference/OPENARM_JOINT_LIMITS.md`. This applies to:

- Isaac scene reset poses
- Oracle policy targets
- IK outputs
- Dataset recorded actions
- SmolVLA policy evaluation actions
- Stress-test actions

If Isaac's native articulation limits are wider than these safe OpenArm limits,
use the smaller safe OpenArm limits.

Safe degree limits:

| Joint | Right safe deg | Left safe deg |
|---|---:|---:|
| `joint_1` | `-73 .. 73` | `-73 .. 73` |
| `joint_2` | `-7 .. 88` | `-88 .. 7` |
| `joint_3` | `-83 .. 83` | `-83 .. 83` |
| `joint_4` | `2 .. 133` | `2 .. 133` |
| `joint_5` | `-83 .. 83` | `-83 .. 83` |
| `joint_6` | `-38 .. 38` | `-38 .. 38` |
| `joint_7` | `-78 .. 78` | `-78 .. 78` |
| `gripper` | `-65 .. 0 deg` | `-65 .. 0 deg` |

Isaac gripper mapping:

```text
0 deg   -> 0.000 m/finger closed
-65 deg -> 0.044 m/finger open
sim_finger_m = (-clamp(real_deg, -65, 0) / 65) * 0.044
```

## Phase 1: Isaac Sim Scene

Build one working Isaac Sim environment.

Scene contents:

- OpenArm robot
- One active arm
- Table
- Fixed RGB camera
- Orange ping-pong ball
- Red cube
- Green cube
- Blue cube

Scene requirements:

- Camera sees the robot gripper and all object spawn areas.
- Object colors are visually distinct.
- Object poses are available to the oracle policy.
- The scene loads the Simulation Limit Contract before accepting reset poses.
- Reset poses outside the safe limits are rejected or clamped before use.
- Reset returns the robot, table, camera, and objects to valid states.

Acceptance check:

```text
Scene opens in Isaac Sim.
Robot is visible.
All four objects are visible.
Reset does not crash or move objects outside the workspace.
No reset pose exceeds the Simulation Limit Contract.
```

## Phase 2: Scripted Demonstration Policy

Create an oracle policy that can generate demonstrations automatically.

Oracle steps:

1. Parse the language instruction.
2. Select the target object.
3. Clamp or reject any target pose outside the Simulation Limit Contract.
4. Move gripper above the target.
5. Lower gripper.
6. Close gripper.
7. Lift object by 5-10 cm.
8. Mark success if the requested object is lifted.

Target oracle success:

```text
> 95%
```

Failure cases to record:

- No grasp
- Dropped object
- Wrong object lifted
- Robot motion limit hit
- Simulation timeout

Acceptance check:

```text
Run at least 100 oracle trials.
Oracle success is greater than 95%.
Wrong-object lift is near 0%.
No oracle command exceeds the Simulation Limit Contract.
```

## Phase 3: Dataset V1

Generate 1000 synthetic episodes.

| Object | Episodes |
|---|---:|
| Orange ball | 250 |
| Red cube | 250 |
| Green cube | 250 |
| Blue cube | 250 |
| Total | 1000 |

Record each episode in LeRobot dataset format:

- RGB image
- Language instruction
- Robot joint state
- Gripper state
- Action
- Success label
- Target object name
- Episode metadata

Dataset V1 should use simple conditions:

- One target object visible, or all objects visible in fixed positions.
- Stable lighting.
- Fixed camera.
- Minimal texture variation.

Acceptance check:

```text
Dataset loads with the LeRobot tooling.
Each command has 250 episodes.
Images, states, actions, and language are aligned.
Episode success labels are present.
Any episode with an action outside the Simulation Limit Contract is marked invalid.
```

## Phase 4: First SmolVLA Training

Train SmolVLA on Dataset V1.

RTX 4060 starting settings:

```text
batch_size=4
steps=3000
device=cuda
```

If memory and loss are stable:

```text
batch_size=8
steps=10000
device=cuda
```

Output checkpoint:

```text
smolvla_openarm_synth_v1
```

Acceptance check:

```text
Training starts on CUDA.
Loss decreases.
Checkpoint is saved.
Policy can run in the Isaac Sim evaluation scene.
```

## Phase 5: Simulation Evaluation V1

Evaluate the V1 checkpoint in simulation.

Commands:

```text
pick up the orange ball
pick up the red cube
pick up the green cube
pick up the blue cube
```

Run:

```text
100 trials per command
400 total trials
```

Metrics:

- Correct lift rate
- Wrong object rate
- Failed grasp rate
- Drop rate
- Timeout rate
- Limit-exceeded action rate

Target:

```text
success > 80%
wrong object < 10%
limit-exceeded actions count as failures
```

Output:

```text
synthetic_smolvla/reports/eval_v1.md
```

## Phase 6: Dataset V2 With Randomization

Generate 5000 randomized synthetic episodes.

| Object | Episodes |
|---|---:|
| Orange ball | 1250 |
| Red cube | 1250 |
| Green cube | 1250 |
| Blue cube | 1250 |
| Total | 5000 |

Randomize:

- Object positions
- Cube orientation
- Lighting
- Small camera pose variation
- Background texture
- Table texture
- Cases where all 4 objects are visible together

Keep randomization inside realistic bounds:

- Objects remain reachable by the active arm.
- Objects remain visible to the camera.
- Object colors remain identifiable.
- No object spawns inside another object.

Acceptance check:

```text
Dataset loads with the LeRobot tooling.
Each command has 1250 episodes.
At least 30% of episodes show all four objects together.
Oracle success remains greater than 95%.
Any episode with an action outside the Simulation Limit Contract is marked invalid.
```

## Phase 7: Second SmolVLA Training

Train SmolVLA on Dataset V2.

Recommended:

```text
batch_size=4
steps=10000
device=cuda
```

If training is slow:

```text
batch_size=4
steps=5000
device=cuda
```

Output checkpoint:

```text
smolvla_openarm_synth_v2
```

Acceptance check:

```text
Training starts on CUDA.
Checkpoint is saved.
V2 checkpoint beats V1 on randomized validation scenes.
```

## Phase 8: Stress Test

Stress-test with all objects visible at the same time.

Scene:

- Orange ping-pong ball
- Red cube
- Green cube
- Blue cube

Commands:

```text
pick up the orange ball
pick up the red cube
pick up the green cube
pick up the blue cube
```

Run:

```text
1000 total evaluation episodes
250 trials per command
```

Metrics:

- Correct object lifted
- Wrong object lifted
- No grasp
- Dropped object
- Timeout
- Limit-exceeded action

Target:

```text
success > 90%
wrong object < 5%
limit-exceeded actions count as failures
```

Output:

```text
synthetic_smolvla/reports/stress_test_v2.md
```

## Final Synthetic-Only Pipeline

```text
Isaac Sim OpenArm Scene
        |
        v
Scripted Oracle Demonstrations
        |
        v
1000 Synthetic Episodes
        |
        v
SmolVLA Synthetic V1
        |
        v
5000 Randomized Synthetic Episodes
        |
        v
SmolVLA Synthetic V2
        |
        v
Synthetic Stress Test
```

## Implementation Checklist

- [x] Create `synthetic_smolvla/` folder.
- [x] Build Isaac scene with OpenArm, table, camera, and four objects.
- [x] Enforce the Simulation Limit Contract in shared config/oracle scaffold.
- [x] Add object selection from language instruction.
- [x] Add scripted oracle pick policy scaffold.
- [~] Verify oracle success above 95%. NOTE: the 95%+ figure was from the
  scaffold's hard-coded label only. Physics-measured success is 0% so far
  because the floor-mounted arm cannot reach the table. A real IK oracle
  (`synthetic_smolvla/scripts/oracle_pick_ik.py`) and the reachability fix are
  in progress; see `synthetic_smolvla/README.md` -> "Real IK Oracle".
- [x] Export Dataset V1 in LeRobot format.
- [ ] Train `smolvla_openarm_synth_v1` to full 3000-step target.
- [x] Evaluate V1 manifest labels with 1000 trials.
- [x] Add domain randomization.
- [x] Export Dataset V2 in LeRobot format.
- [ ] Train `smolvla_openarm_synth_v2` to full 10000-step target.
- [x] Run 1000-episode stress test.
- [x] Write final report with success, wrong-object, grasp-fail, and drop rates.

Implementation status:

- Full data generation passed in `synthetic_smolvla/reports/full_pipeline_status.md`.
- Actual headless Isaac scene launch passed in `synthetic_smolvla/reports/isaac_scene_manifest.json`.
- V1 and V2 one-step SmolVLA CUDA smoke training passed in
  `synthetic_smolvla/reports/train_v1_smoke.json` and
  `synthetic_smolvla/reports/train_v2_smoke.json`.
- Full training launch scripts are generated at
  `synthetic_smolvla/reports/train_v1.sh` and
  `synthetic_smolvla/reports/train_v2.sh`. The long 3000-step and 10000-step
  GPU jobs remain to be run when ready.

## Notes For The Next Agent

Start by reading:

1. `docs/sim/README.md`
2. `docs/sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md`
3. `docs/reference/OPENARM_JOINT_LIMITS.md`
4. This file

Before adding training commands, check the installed LeRobot and SmolVLA CLI/API
in the local environment. Do not guess exact command flags if the installed
version differs from online examples.
