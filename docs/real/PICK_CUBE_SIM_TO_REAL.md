# Cube Pick — Sim To Real (OpenArm)

A scripted **waypoint** pick for a cube at a **fixed, taught location**. No vision,
no inverse kinematics. You teach a short pose sequence once, watch it in Isaac,
then replay the same joint angles on the real robot.

Driver: [scripts/pick_cube.py](../../scripts/pick_cube.py)
Poses:  [scripts/pick_cube_waypoints.json](../../scripts/pick_cube_waypoints.json)

Sequence: `home → pregrasp → descend → grasp → lift → retreat`
(each waypoint is 7 joint angles + a gripper angle).

## How It Works

`pick_cube.py` uses the same `OpenArmFollower` position control as the other real
scripts. One `send_action` sets all 8 motors (7 joints + gripper); each command
is creep-clamped by `max_relative_target` so poses are approached smoothly. Three
modes:

| Mode | What it does | Touches real robot? |
|---|---|---|
| `--teach` | Torque OFF; move arm by hand, press Enter to capture each pose | Reads only |
| `--sim-only` | Mirror each pose into Isaac Lab only | No |
| (default) `--i-am-at-robot` | Replay taught poses on the real arm | Yes |

Real replay is **refused while `taught` is false** in the JSON (the shipped poses
are placeholders) unless you pass `--force-untaught`.

## 0. Prerequisites

Scripts already deployed to the Jetson at
`/home/arms/hsi-pre-grasp/scripts/`. To redeploy from the laptop:

```bash
cd /home/chyanin/Desktop/realrobot
scp scripts/pick_cube.py scripts/pick_cube_waypoints.json arms@10.10.10.2:/home/arms/hsi-pre-grasp/scripts/
```

For Isaac visualization, start the mirror server on the laptop:

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt        # listens on http://10.10.10.1:8765
```

On the Jetson, for every step:

```bash
ssh arms@10.10.10.2
cd /home/arms/hsi-pre-grasp && source .venv/bin/activate
sudo ./scripts/can_up.sh                   # CAN up; 24 V on; e-stop ready
```

## 1. Teach The Poses By Hand (person at robot)

Torque is disabled so you can move the arm freely. Move to each keyframe, press
Enter to capture. Position the gripper by hand too (open for home/pregrasp/
descend, closed for grasp/lift/retreat).

```bash
python scripts/pick_cube.py --teach
```

This overwrites the `pose` of each waypoint with the real captured angles and
sets `taught: true`. Re-run any time to re-teach. (`s` skips a keyframe, `q`
quits.)

If a captured pose is **outside the joint limits**, teach refuses to save it,
prints which joint exceeded its range, and asks you to move into range and teach
that keyframe again. `taught: true` is only set if you finish all keyframes — if
you quit partway, the file stays untaught and real replay stays blocked.

## 2. Watch It In Isaac (no hardware)

```bash
python scripts/pick_cube.py --sim-only
```

The sim **interpolates smoothly to each pose and holds it before moving to the
next**, so it follows the taught path instead of jumping. Slow it down if needed:

```bash
python scripts/pick_cube.py --sim-only --sim-deg-per-step 1.0 --sim-dt 0.15 --sim-settle-sec 2.0
```

| Flag | Effect | Default |
|---|---|---|
| `--sim-deg-per-step` | degrees per sub-step (smaller = smoother/slower) | 2.0 |
| `--sim-dt` | seconds between sub-steps (larger = slower) | 0.1 |
| `--sim-settle-sec` | hold time at each pose before the next | 1.0 |

Each pose is mirrored to the Isaac viewer so you can confirm the trajectory and
grasp look right before moving the real arm.

## 3. Run It On The Real Robot (person at rig, e-stop ready)

```bash
python scripts/pick_cube.py --i-am-at-robot            # prompts, type PICK
python scripts/pick_cube.py --i-am-at-robot --yes      # no prompt
```

The arm plays the taught sequence, creeping into each pose, mirroring to Isaac as
it goes. It refuses any pose outside the live `joint_limits`. If a pose is not
reached within `settle_timeout_sec` it stops (add `--continue-on-miss` to push
through).

## Tuning (`pick_cube_waypoints.json`)

| Field | Meaning |
|---|---|
| `side` / `port` | `right`/`can0` or `left`/`can1` |
| `max_rel` | degrees per control step (creep speed), default 5 |
| `tolerance_deg` | per-joint "reached" tolerance |
| `hold_sec` | time a pose must stay in tolerance to count as reached |
| `settle_timeout_sec` | max seconds to reach one pose |
| `taught` | set true by `--teach`; gates real replay |
| `waypoints[].dwell` | pause after a pose (e.g. let the grasp settle) |
| `waypoints[].pose` | the 7 joint angles + `gripper`, in degrees |

Gripper reminder: `0` = closed, `-65` = open limit, more negative = more open.

## Safety

- Teach and real replay both require a person at the rig with e-stop ready.
- The placeholder poses are NOT for the real robot — teach first.
- Per-step `max_rel` keeps motion slow; lower it for the first real run.
- This pick is open-loop: it assumes the cube is at the taught spot every time.
