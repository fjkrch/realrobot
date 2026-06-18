# Real Jetson Command To Isaac Simulation

Goal: type a command on the Jetson robot computer and mirror that command in
Isaac Lab / Isaac Sim on this laptop first.

This is the safe first stage. It does not move the physical OpenArm robot.

## What Runs Where

| Machine | Role | Address |
|---|---|---|
| Laptop | Runs Isaac Lab / Isaac Sim and the mirror server | `10.10.10.1` |
| Jetson robot | Sends command text to the laptop | `10.10.10.2` |

The Jetson sends text such as `pick up the cube` to the laptop. The laptop
starts the existing Isaac Lab OpenArm word-control demo.

## 1. Start Laptop Mirror Server

### Clean Default Pose Server

Use this when you only want the OpenArm robot in Isaac, with no cube, no pick
task, no episode reset, and no random object movement:

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt
```

Keep this terminal open. Isaac Sim opens once and listens on:

```text
http://10.10.10.1:8765
```

This is the clean server to use with real Jetson gripper commands.

### Best For Visualization: Isaac Always Open

Use this when you want Isaac Lab to stay open and move whenever a Jetson command
arrives:

```bash
cd /home/chyanin/IsaacLab
bash source/hsi_pregrasp_refusal/run_openarm_live_command_server.txt
```

Keep this terminal open. Isaac Sim opens once and stays open. The live Isaac
scene listens on:

```text
http://10.10.10.1:8765
```

Live mode holds the simulated arm in a stable ready/default pose and only
changes the gripper when a Jetson gripper command arrives. The normal Isaac Lab
training timeout/drop resets are disabled by default, so the viewer should not
reset every few seconds. To intentionally test with the original resets:

```bash
ALLOW_AUTO_RESETS=1 bash source/hsi_pregrasp_refusal/run_openarm_live_command_server.txt
```

Then Jetson commands update the already-open simulated robot:

```bash
python scripts/jetson_isaaclab_command.py "gripper target -10.000 deg"
python scripts/jetson_isaaclab_command.py "gripper target 0.000 deg"
python scripts/jetson_isaaclab_command.py "pick up the cube"
python scripts/jetson_isaaclab_command.py "stop"
```

Do not run `scripts/isaaclab_command_server.py` at the same time on port `8765`.
The live Isaac scene is the server in this mode.

### Older Mode: Start Isaac Per Command

On the laptop:

```bash
cd /home/chyanin/Desktop/realrobot
python scripts/isaaclab_command_server.py --host 10.10.10.1 --port 8765
```

Keep this terminal open. This older mode launches a new Isaac process per
command, so it is slower and not as nice for watching repeated motion.

For a safe connection test that does not launch Isaac Sim:

```bash
cd /home/chyanin/Desktop/realrobot
python scripts/isaaclab_command_server.py --host 10.10.10.1 --port 8765 --dry-run
```

## 2. Copy Client To Jetson

From the laptop:

```bash
cd /home/chyanin/Desktop/realrobot
scp scripts/jetson_isaaclab_command.py arms@10.10.10.2:/home/arms/hsi-pre-grasp/scripts/
```

If direct LAN is not connected, use the Wi-Fi robot address instead:

```bash
scp scripts/jetson_isaaclab_command.py arms@192.168.31.50:/home/arms/hsi-pre-grasp/scripts/
```

## 3. Send Commands From Jetson

SSH into the Jetson:

```bash
ssh arms@10.10.10.2
```

Then:

```bash
cd /home/arms/hsi-pre-grasp
python scripts/jetson_isaaclab_command.py "pick up the cube"
```

Other mirror commands:

```bash
python scripts/jetson_isaaclab_command.py "open gripper"
python scripts/jetson_isaaclab_command.py "close gripper"
python scripts/jetson_isaaclab_command.py "stop"
```

Check laptop mirror status:

```bash
python scripts/jetson_isaaclab_command.py --status
```

Stop the current Isaac Lab mirror job:

```bash
python scripts/jetson_isaaclab_command.py --stop
```

## 4. Command Behavior

| Jetson command | Isaac Lab mirror behavior |
|---|---|
| `pick up the cube` | Runs the scripted OpenArm pick-and-lift demo |
| `grab`, `grasp`, `lift` | Same as pick command |
| `open gripper` | Holds pose and opens the simulated gripper |
| `close gripper` | Holds pose and closes the simulated gripper |
| `stop`, `hold`, `wait` | Stops a running mirror job, or starts a short hold if idle |
| Unknown command | Fails safe to stop/hold |

If Isaac Lab is already running a mirror job, the server rejects new motion
commands until you send `stop`.

## 5. Real Gripper Command Mirror

The Jetson script `/home/arms/hsi-pre-grasp/scripts/open_gripper_small.py` now
also sends a best-effort numeric Isaac mirror command before it sends the real
gripper action.

For visual matching, start the always-open Isaac server first:

```bash
cd /home/chyanin/IsaacLab
bash source/hsi_pregrasp_refusal/run_openarm_live_command_server.txt
```

Your real command:

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

Because `0 deg` is closed, this also sends this numeric target to Isaac Lab:

```text
gripper target 0.000 deg
```

Open targets mirror with the same degree value:

```bash
python scripts/open_gripper_small.py \
  --port can0 \
  --side right \
  --target-deg -10 \
  --i-am-at-robot \
  --yes
```

This sends this to Isaac Lab:

```text
gripper target -10.000 deg
```

Isaac maps the real OpenArm gripper range to the simulated finger joint:

```text
Real robot: 0 deg closed, -65 deg fully open
Isaac sim:  0.000 m/finger closed, 0.044 m/finger fully open
Example:   -10 deg -> about 0.0068 m/finger in Isaac
```

If you use the older per-command bridge instead, the laptop mirror server must
already be running:

```bash
cd /home/chyanin/Desktop/realrobot
python scripts/isaaclab_command_server.py --host 10.10.10.1 --port 8765
```

To run the real gripper command without Isaac mirroring:

```bash
python scripts/open_gripper_small.py ... --no-isaac-mirror
```

To use a different laptop/server:

```bash
ISAACLAB_MIRROR_SERVER=http://10.10.10.1:8765 python scripts/open_gripper_small.py ...
```

If the Isaac mirror server is not reachable, the gripper script prints a warning
and continues with the real robot command.

Verified on 2026-06-16 with no robot motion:

```text
Jetson helper sent: gripper target -10.000 deg
Laptop bridge mode: gripper_target
Bridge status: completed
Server mode: dry-run
```

## 6. Logs

Laptop logs are saved here:

```text
/home/chyanin/Desktop/realrobot/logs/command_bridge/
```

Check the newest log:

```bash
cd /home/chyanin/Desktop/realrobot
ls -lt logs/command_bridge | head
tail -80 logs/command_bridge/<log-file-name>
```

## 7. Curl Test From Jetson

If the Python client is not copied yet, test with curl:

```bash
curl -X POST http://10.10.10.1:8765/command \
  -H 'Content-Type: application/json' \
  -d '{"command":"pick up the cube"}'
```

Status:

```bash
curl http://10.10.10.1:8765/status
```

Stop:

```bash
curl -X POST http://10.10.10.1:8765/stop
```

## 8. Safe Test Order

Run in this order:

1. Start laptop server with `--dry-run`.
2. From Jetson, run `python scripts/jetson_isaaclab_command.py --status`.
3. From Jetson, run `python scripts/jetson_isaaclab_command.py "pick up the cube"`.
4. Confirm the laptop writes a log under `logs/command_bridge/`.
5. Restart laptop server without `--dry-run`.
6. From Jetson, send `pick up the cube` and watch Isaac Sim mirror the command.
7. Send `stop` before sending a new command.

## 9. Verified Dry-Run Result

Verified on 2026-06-16:

```text
Laptop bridge: http://10.10.10.1:8765
Jetson client: /home/arms/hsi-pre-grasp/scripts/jetson_isaaclab_command.py
Command sent from Jetson: pick up the cube
Bridge mode: pick
Bridge status: completed
Server mode: dry-run
Robot motion: none
Isaac Sim launch: none
```

This proves the Jetson-to-laptop command path works. The next step is to start
the laptop bridge without `--dry-run` so the same Jetson command launches the
Isaac Lab OpenArm mirror.

## 10. Troubleshooting

If Jetson cannot reach laptop:

```bash
ping 10.10.10.1
curl http://10.10.10.1:8765/health
```

If laptop is not listening on LAN, restart the server with:

```bash
python scripts/isaaclab_command_server.py --host 10.10.10.1 --port 8765
```

If the server says Isaac Lab is already running:

```bash
python scripts/jetson_isaaclab_command.py "stop"
```

If Isaac Lab Python/imports fail, check the local guide:

```text
ISAACLAB_ISAACSIM_LOCAL_GUIDE.md
```

## 11. Optional LAN Token

For a simple token:

```bash
# laptop
BRIDGE_TOKEN=mytoken python scripts/isaaclab_command_server.py --host 10.10.10.1 --port 8765

# Jetson
BRIDGE_TOKEN=mytoken python scripts/jetson_isaaclab_command.py "pick up the cube"
```

## 12. Important Safety Boundary

This bridge is for Isaac simulation first. It does not send CAN commands, does
not enable torque, and does not move the real OpenArm robot.

Real robot control should stay separate until the Isaac mirror behavior is
verified and a person is physically at the rig with E-stop ready.
