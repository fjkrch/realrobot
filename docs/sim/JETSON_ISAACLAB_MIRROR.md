# Jetson To Isaac Lab Mirror

This is the safe first step: commands typed on the Jetson are mirrored into
Isaac Lab on the laptop. The physical OpenArm robot is not commanded by this
bridge.

## Laptop: Start The Mirror Server

### Clean Default Pose Server

Use this when you want no cube task, no pick behavior, no episode reset, and
only the OpenArm robot at default pose:

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt
```

This is the best server for real Jetson gripper commands like
`scripts/open_gripper_small.py`. It listens on `http://10.10.10.1:8765`.

### Recommended: Isaac Always Open

Use this when you want Isaac Lab visible all the time:

```bash
cd /home/chyanin/IsaacLab
bash source/hsi_pregrasp_refusal/run_openarm_live_command_server.txt
```

Isaac Sim opens once, stays open, and listens for Jetson commands at
`http://10.10.10.1:8765`.

Live mode holds the simulated arm in a stable ready/default pose and only
changes the gripper when a Jetson gripper command arrives. Training-style task
resets are disabled by default, so the scene should not jump back every few
seconds. To re-enable the original Isaac Lab timeout/drop resets for debugging:

```bash
ALLOW_AUTO_RESETS=1 bash source/hsi_pregrasp_refusal/run_openarm_live_command_server.txt
```

Do not run `scripts/isaaclab_command_server.py` at the same time on port `8765`.
In this mode the Isaac viewer itself is the command server.

### Older: Launch Isaac Per Command

Use the direct LAN laptop address from the handoff notes:

```bash
cd /home/chyanin/Desktop/realrobot
python scripts/isaaclab_command_server.py --host 10.10.10.1 --port 8765
```

For a quick no-Isaac test:

```bash
python scripts/isaaclab_command_server.py --host 10.10.10.1 --port 8765 --dry-run
```

## Jetson: Send Commands

Copy the small client to the Jetson once:

```bash
scp scripts/jetson_isaaclab_command.py arms@10.10.10.2:/home/arms/hsi-pre-grasp/scripts/
```

Then run commands from the Jetson:

```bash
cd /home/arms/hsi-pre-grasp
python scripts/jetson_isaaclab_command.py "pick up the cube"
python scripts/jetson_isaaclab_command.py "open gripper"
python scripts/jetson_isaaclab_command.py "close gripper"
python scripts/jetson_isaaclab_command.py "stop"
```

Check status:

```bash
python scripts/jetson_isaaclab_command.py --status
```

Stop a running Isaac Lab mirror job:

```bash
python scripts/jetson_isaaclab_command.py --stop
```

## Verified Dry-Run

Verified on 2026-06-16:

- Laptop direct LAN: `10.10.10.1`
- Jetson direct LAN: `10.10.10.2`
- Laptop dry-run server answered `/health`
- Client copied to `/home/arms/hsi-pre-grasp/scripts/jetson_isaaclab_command.py`
- Jetson ran `python scripts/jetson_isaaclab_command.py --status`
- Jetson ran `python scripts/jetson_isaaclab_command.py "pick up the cube" --wait`
- Server classified the command as `pick`
- Job completed in dry-run mode
- No Isaac Sim launch and no physical robot motion happened

## Curl Alternative

If the client script is not copied yet:

```bash
curl -X POST http://10.10.10.1:8765/command \
  -H 'Content-Type: application/json' \
  -d '{"command":"pick up the cube"}'
```

## Behavior

- `pick`, `grab`, `grasp`, or `lift` commands run the Isaac Lab scripted
  OpenArm pick-and-lift mirror.
- `open gripper` and `close gripper` mirror gripper commands in Isaac Lab only.
- `stop`, `hold`, `wait`, and unknown commands fail safe to stop/hold.
- If a mirror job is already running, send `stop` before starting another one.
- Logs are written on the laptop under `logs/command_bridge/`.

## Real Gripper Script Numeric Mirror

The Jetson gripper script now mirrors the numeric gripper target too. This
command still moves the real gripper, and also sends
`gripper target 0.000 deg` to Isaac Lab:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python scripts/open_gripper_small.py \
  --port can0 \
  --side right \
  --target-deg 0 \
  --tolerance-deg 0.5 \
  --timeout-sec 4 \
  --hold-sec 0.2 \
  --i-am-at-robot \
  --yes \
  --disable-on-fail
```

Negative targets keep the same degree value. For example, `--target-deg -10`
sends this to Isaac Lab:

```text
gripper target -10.000 deg
```

Isaac maps real OpenArm gripper degrees to simulated finger position:

```text
0 deg   -> 0.000 m/finger closed
-65 deg -> 0.044 m/finger fully open
-10 deg -> about 0.0068 m/finger
```

With the always-open Isaac server, you should see the gripper move in the
already-open viewer each time one of these commands is sent.

Disable mirror for one run:

```bash
python scripts/open_gripper_small.py ... --no-isaac-mirror
```

Verified with no robot motion on 2026-06-16: calling the Jetson mirror helper
sent `gripper target -10.000 deg`, and the laptop dry-run bridge completed it as
`gripper_target`.

## Optional Token

For a simple LAN token:

```bash
# laptop
BRIDGE_TOKEN=mytoken python scripts/isaaclab_command_server.py --host 10.10.10.1 --port 8765

# Jetson
BRIDGE_TOKEN=mytoken python scripts/jetson_isaaclab_command.py "pick up the cube"
```
