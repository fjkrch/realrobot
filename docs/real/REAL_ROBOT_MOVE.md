# Real Robot Move Commands

This file collects the commands that move the **physical** OpenArm robot.
Everything here sends real CAN motion. This is different from the
`*_MIRROR.md` files, which only move the robot inside Isaac Sim.

Source of truth for these commands: `OPENARM_ROBOT_HANDOFF.md`.

## ⚠️ Safety First (required every time)

- A person must be **physically at the rig** before any motion command.
- **E-stop in reach** before motor power or any motion command.
- **24 V motor power ON**, or motors will not answer CAN.
- These scripts use `connect(calibrate=False)` so they do **not** re-zero motors.
- Do **not** run `lerobot-calibrate` casually (it can re-mark motor zero).

The motion scripts only arm real motion when you pass `--i-am-at-robot`
(and `--yes` to skip the confirmation prompt). Without those flags they will
not move the robot.

## 0. Connect To The Robot (Jetson)

These scripts live on the Jetson, not the laptop.

```bash
# direct LAN (preferred)
ssh arms@10.10.10.2
# Wi-Fi fallback:
# ssh arms@192.168.31.50

cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
```

## 1. Bring Up CAN (prerequisite for motion)

```bash
cd /home/arms/hsi-pre-grasp
sudo ./scripts/can_up.sh
```

Verify motors answer (only with 24 V on and a person at the rig):

```bash
lerobot-setup-can --mode=test --interfaces=can0,can1
# expect: can0 8/8, can1 8/8
```

Usual mapping (verify physically, cables can be swapped):

```text
can0 = right / follower arm
can1 = left  / leader arm
```

## 2. Move The Gripper (simplest real motion)

Gripper range: `0 deg` = closed, `-65 deg` = open limit. Positive values are
refused/clipped.

Open the right gripper a little:

```bash
python scripts/open_gripper_small.py \
  --port can0 \
  --side right \
  --target-deg -10 \
  --tolerance-deg 0.5 \
  --timeout-sec 10 \
  --i-am-at-robot \
  --yes
```

Close the right gripper:

```bash
python scripts/open_gripper_small.py \
  --port can0 \
  --side right \
  --target-deg 0 \
  --tolerance-deg 0.5 \
  --timeout-sec 10 \
  --i-am-at-robot \
  --yes
```

Left gripper: use `--port can1 --side left`.

If the target is not reached the script keeps torque on. Press Ctrl-C to stop,
or add `--disable-on-fail` to release torque even on failure.

## Joint Limits (read live, observed 2026-06-16)

`move_joint.py` / `move_arm.py` always read these from the live config and refuse
out-of-range targets. Values observed on this rig (degrees):

| Joint | Right (can0) | Left (can1) |
|---|---|---|
| joint_1 | -75 .. 75 | -75 .. 75 |
| joint_2 | -9 .. 90 | -90 .. 9 |
| joint_3 | -85 .. 85 | -85 .. 85 |
| joint_4 | 0 .. 135 | 0 .. 135 |
| joint_5 | -85 .. 85 | -85 .. 85 |
| joint_6 | -40 .. 40 | -40 .. 40 |
| joint_7 | -80 .. 80 | -80 .. 80 |
| gripper | -65 .. 0 | -65 .. 0 |

Note `joint_2` is mirrored between arms. These are a reference snapshot; the live
config wins.

## 3. Move A Single Arm Joint To An Angle

Use `scripts/move_joint.py` (sibling of `open_gripper_small.py`, same
`OpenArmFollower` API and safety pattern). It moves ONE joint (`1`..`7`) to a
target angle, clamps to the joint's real `joint_limits`, and creeps to target
using `max_relative_target` so it does not snap. It refuses if the joint name or
its current position cannot be read from the live config.

Copy it to the Jetson once (from the laptop):

```bash
cd /home/chyanin/Desktop/realrobot
scp scripts/move_joint.py arms@10.10.10.2:/home/arms/hsi-pre-grasp/scripts/
```

Small relative nudge first (prompts before moving, shows range + current pos):

```bash
python scripts/move_joint.py --port can0 --side right --joint 1 --delta-deg 3 --i-am-at-robot
```

Absolute "set joint to angle":

```bash
python scripts/move_joint.py --port can0 --side right --joint 1 --target-deg 10 --i-am-at-robot --yes
python scripts/move_joint.py --port can0 --side right --joint 2 --target-deg -15 --i-am-at-robot --yes
python scripts/move_joint.py --port can1 --side left  --joint 4 --target-deg 20 --i-am-at-robot --yes
```

Useful flags: `--max-rel` (deg per step, default 5), `--timeout-sec`,
`--tolerance-deg`, `--disable-on-fail`, `--no-isaac-mirror`.

## 3a. Move All 7 Joints At Once (whole body)

Use `scripts/move_arm.py` — the real-robot version of the Isaac phrase
`"whole body right 0 5 -10 20 0 0 0"`. It drives joint_1..joint_7 together, each
clamped to its real `joint_limits` and capped per step by `--max-rel`.

Copy it to the Jetson once (from the laptop):

```bash
cd /home/chyanin/Desktop/realrobot
scp scripts/move_arm.py arms@10.10.10.2:/home/arms/hsi-pre-grasp/scripts/
```

Run it (port auto: right=can0, left=can1):

```bash
python scripts/move_arm.py "whole body right 0 5 -10 20 0 0 0" --i-am-at-robot        # prompts
python scripts/move_arm.py "whole body right 0 5 -10 20 0 0 0" --i-am-at-robot --yes  # no prompt
python scripts/move_arm.py "whole body left  0 5 -10 20 0 0 0" --i-am-at-robot --yes
```

Both arms at once (14 values) is intentionally NOT supported yet — it needs two
simultaneous CAN connections. Run one side at a time.

## 3b. Move The Whole Arm (arm-to-arm teleop)

To move all joints at once, use teleop: the follower arm copies the leader arm
you move **by hand**. Stage both arms in the same posture first.

```bash
python scripts/teleop_native.py \
  --follower-port can0 --follower-side right \
  --leader-port can1 --max-rel 5 --fps 60
```

Notes:

```text
No re-zero. connect(calibrate=False).
Default mirror flips all joints except joint_4 and gripper.
```

## 4. Stop / Release Torque

```bash
python scripts/disable_torque.py --port can0 --side right
python scripts/disable_torque.py --port can1 --side left
```

If a gripper script is stuck:

```bash
pkill -f open_gripper_small.py
python scripts/disable_torque.py --port can0 --side right
```

## Troubleshooting

### `Handshake failed ... motors did not respond: ['joint_1']`

This is usually transient. `can_up.sh`'s ACK test leaves the bus in
`ERROR-PASSIVE` and resets the interface, so the **first** handshake right after
it often misses. `move_joint.py` and `move_arm.py` now retry the handshake 3×
automatically (`--connect-retries`, `--connect-retry-delay`). If it still fails:

- Confirm the 24 V motor supply is ON and the e-stop is released.
- Test the specific bus: `lerobot-setup-can --mode=test --interfaces=can0`
  (or `can1`). You want `8/8 motors found`. The left arm is on **can1**.
- If a bus shows fewer than 8/8, re-seat the PCAN USB (the rig has a known
  `error -71` hub issue) and rerun `sudo ./scripts/can_up.sh`.

### `Relative goal position magnitude had to be clamped to be safe`

Not an error. That is the per-step `max_relative_target` clamp creeping the
joint toward its target a few degrees at a time. Raise/lower with `--max-rel`.

## Verified Status (2026-06-16)

- Passwordless SSH laptop (`10.10.10.1`) → Jetson (`arms@10.10.10.2`) is set up,
  so scripts are deployed with `scp` (no password prompt).
- `move_arm.py` and `move_joint.py` are deployed on the Jetson under
  `/home/arms/hsi-pre-grasp/scripts/` with the CAN handshake retry fix.
- **Real motion confirmed:** `move_arm.py "whole body right 0 5 -10 20 0 0 0"`
  drove the right arm on the real robot — joint_4 crept 0 → 20°, joint_2 → ~5°,
  joint_3 → -10°, all within ±0.5°, then torque disabled cleanly.
- Known open item: the **left arm (can1)** hit a handshake miss; retry now covers
  the transient case, but verify `lerobot-setup-can --mode=test --interfaces=can1`
  shows 8/8 if it persists.

## What Is NOT Real Motion

These are Isaac-sim only and do **not** move the physical robot:

```bash
python scripts/jetson_isaaclab_command.py "joint right 1 10 deg"
python scripts/jetson_isaaclab_command.py "whole body right 0 5 -10 20 0 0 0"
python scripts/jetson_isaaclab_command.py "pick up the cube"
```

See `../sim/WHOLE_BODY_OPENARM_ISAAC_MIRROR.md` and
`../sim/REAL_JETSON_TO_ISAAC_SIMULATION.md` for the simulation side.
