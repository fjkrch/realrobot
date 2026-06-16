
This file is for the next agent/operator. It contains connection info, repo paths,
commands, scripts added during this session, and safety notes.

Do not commit this file to any public repo. It contains a password.

## SSH Access

Robot host:

```text
192.168.31.50
```

SSH user:

```text
arms
```

SSH command:

```bash
ssh arms@192.168.31.50
```

Password:

```text
[REDACTED - do not commit robot password]
```

Windows/PowerShell direct command:

```powershell
ssh arms@192.168.31.50
```

If SSH asks whether to trust the host key, type:

```text
yes
```

## Direct LAN Access

Direct Ethernet was configured on 2026-06-16 because both sides were plugged in
but waiting forever for DHCP.

Verified working:

```text
Laptop -> robot LAN ping: OK
Laptop -> robot LAN SSH: OK
```

Laptop wired interface:

```text
enp2s0 = 10.10.10.1/24
```

Robot wired interface:

```text
eth0 = 10.10.10.2/24
```

LAN SSH command:

```bash
ssh arms@10.10.10.2
```

The LAN connection has no default gateway, so Wi-Fi remains the internet/default
route. Quick checks:

```bash
ping 10.10.10.2
ssh arms@10.10.10.2
```

If LAN reverts to DHCP, set it again:

```bash
# laptop
nmcli con modify "Wired connection 1" \
  ipv4.method manual \
  ipv4.addresses 10.10.10.1/24 \
  ipv4.gateway "" \
  ipv4.dns "" \
  ipv4.never-default yes \
  ipv6.method disabled \
  connection.autoconnect yes \
  connection.autoconnect-priority 100
nmcli con up "Wired connection 1"

# robot, from an existing SSH path
sudo nmcli con modify "Wired connection 1" \
  ipv4.method manual \
  ipv4.addresses 10.10.10.2/24 \
  ipv4.gateway "" \
  ipv4.dns "" \
  ipv4.never-default yes \
  ipv6.method disabled \
  connection.autoconnect yes \
  connection.autoconnect-priority 100
sudo nmcli con up "Wired connection 1"
```

## Main Robot Repo

Repo path on robot:

```bash
/home/arms/hsi-pre-grasp
```

Enter repo and activate Python environment:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
```

The repo was cloned from:

```text
https://github.com/ChawinOph/openarm_ws.git
```

Important files to read first:

```bash
README.md
docs/openarm-quest3-lerobot-setup.md
docs/can-cheatsheet.md
scripts/open_gripper_small.py
scripts/disable_torque.py
scripts/camera_mjpeg_stream.py
scripts/teleop_native.py
```

Recommended first read:

```bash
cd /home/arms/hsi-pre-grasp
sed -n '/## Operator command cheat sheet/,/## Safety/p' README.md
sed -n '1,220p' docs/openarm-quest3-lerobot-setup.md
```

## Safety Rules

- Do not command robot motion remotely unless a person is physically at the rig.
- E-stop must be in reach before motor power or any motion command.
- 24 V motor power must be ON for motors to answer CAN queries.
- The repo documentation warns that stock LeRobot calibration can re-mark motor zero.
- Prefer scripts that use `connect(calibrate=False)` unless intentionally calibrating.
- Do not run `lerobot-calibrate` casually.

## Current Installed Software

Python environment:

```bash
/home/arms/hsi-pre-grasp/.venv
```

Known working imports/tools:

```text
lerobot 0.5.1
python-can 4.6.1
opencv/cv2 4.13.0
oculus_reader installed
custom teleoperator plugin import ok
```

Check:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python -c "import can, lerobot, oculus_reader.reader, lerobot_teleoperator_openarm_quest; print('ok')"
which lerobot-record
which lerobot-setup-can
```

## CAN Bus

Hardware:

```text
PEAK PCAN-USB Pro FD
can0 = PCAN channel 0
can1 = PCAN channel 1
```

Usually:

```text
can0 = right/follower arm
can1 = left/leader arm
```

But verify physically because cables can be swapped.

Start CAN-FD:

```bash
cd /home/arms/hsi-pre-grasp
sudo ./scripts/can_up.sh
```

Manual CAN-FD setup:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up

sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up
```

Expected:

```text
mtu 72
can <FD>
state ERROR-ACTIVE
bitrate 1000000
dbitrate 5000000
```

Check:

```bash
ip -details link show can0
ip -details link show can1
```

Motor response test:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
lerobot-setup-can --mode=test --interfaces=can0,can1
```

Known good result from this session:

```text
can0: 8/8 motors found
can1: 8/8 motors found
Total motors found: 16
```

Each arm has 8 motors:

```text
joint_1  = motor ID 0x01, response 0x11
joint_2  = motor ID 0x02, response 0x12
joint_3  = motor ID 0x03, response 0x13
joint_4  = motor ID 0x04, response 0x14
joint_5  = motor ID 0x05, response 0x15
joint_6  = motor ID 0x06, response 0x16
joint_7  = motor ID 0x07, response 0x17
gripper  = motor ID 0x08, response 0x18
```

Known CAN/USB issue:

```text
USB logs showed: disabled by hub (EMI?), error -71
```

If `can0`/`can1` disappear, replug the PCAN adapter. Prefer a short direct USB cable
instead of a hub.

Diagnostic commands:

```bash
lsusb | grep -i -E "peak|pcan|0c72"
ip -br link show
dmesg | grep -i -e can -e pcan -e peak -e usb | tail -80
```

## Camera Stream

Camera hardware seen:

```text
Intel RealSense D435i
```

Readable OpenCV indices seen:

```text
2 and 4
```

The live stream script is:

```bash
/home/arms/hsi-pre-grasp/scripts/camera_mjpeg_stream.py
```

Start low-latency color stream, rotated 180 degrees:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python scripts/camera_mjpeg_stream.py \
  --index 2 --host 0.0.0.0 --port 8091 \
  --width 640 --height 480 --fps 30 \
  --jpeg-quality 70 --rotate-180
```

Open in browser:

```text
http://192.168.31.50:8091/
```

Background start:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
nohup python scripts/camera_mjpeg_stream.py \
  --index 2 --host 0.0.0.0 --port 8091 \
  --width 640 --height 480 --fps 30 \
  --jpeg-quality 70 --rotate-180 \
  > /tmp/robot_camera_mjpeg_8091.log 2>&1 &
echo $! > /tmp/robot_camera_mjpeg_8091.pid
```

Camera status:

```bash
curl http://127.0.0.1:8091/status
cat /tmp/robot_camera_mjpeg_8091.log
```

Stop stream:

```bash
kill $(cat /tmp/robot_camera_mjpeg_8091.pid)
```

or:

```bash
pkill -f camera_mjpeg_stream.py
```

## Gripper Control

Script:

```bash
/home/arms/hsi-pre-grasp/scripts/open_gripper_small.py
```

Purpose:

```text
Move only the gripper motor to a target position.
Uses connect(calibrate=False).
Does not calibrate.
Does not re-mark motor zero.
Polls gripper position and only disables after reaching target.
```

Important gripper range:

```text
0 deg    = closed
-65 deg  = open limit
negative values = more open
positive values like +10 are invalid and will be refused/clipped
```

Open right/can0 gripper a little:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python scripts/open_gripper_small.py \
  --port can0 \
  --side right \
  --target-deg -10 \
  --tolerance-deg 0.5 \
  --timeout-sec 10 \
  --i-am-at-robot
```

Close right/can0 gripper:

```bash
python scripts/open_gripper_small.py \
  --port can0 \
  --side right \
  --target-deg 0 \
  --tolerance-deg 0.5 \
  --timeout-sec 10 \
  --i-am-at-robot
```

Left/can1 gripper example:

```bash
python scripts/open_gripper_small.py \
  --port can1 \
  --side left \
  --target-deg -10 \
  --tolerance-deg 0.5 \
  --timeout-sec 10 \
  --i-am-at-robot
```

If target is not reached, the script keeps torque on by default. Press Ctrl-C to stop
and disable. To disable even on failure, add:

```bash
--disable-on-fail
```

Known tested command from session:

```bash
python scripts/open_gripper_small.py \
  --port can0 \
  --side right \
  --target-deg -2 \
  --tolerance-deg 0.5 \
  --timeout-sec 4 \
  --hold-sec 0.2 \
  --i-am-at-robot \
  --yes \
  --disable-on-fail
```

Observed result:

```text
Current gripper.pos: -9.890 deg
Target gripper.pos: -2.000 deg
Reached target near -2.328 deg
Disconnected, torque off
```

## Disable Torque / Stop Holding

Script:

```bash
/home/arms/hsi-pre-grasp/scripts/disable_torque.py
```

Disable torque on can0:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python scripts/disable_torque.py --port can0 --side right
```

Disable torque on can1:

```bash
python scripts/disable_torque.py --port can1 --side left
```

If a gripper script is stuck:

```bash
pkill -f open_gripper_small.py
python scripts/disable_torque.py --port can0 --side right
```

## Native Arm-To-Arm Teleop

Script:

```bash
/home/arms/hsi-pre-grasp/scripts/teleop_native.py
```

Use only with person at rig and e-stop ready.

Command from README/doc:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python scripts/teleop_native.py \
  --follower-port can0 --follower-side right \
  --leader-port can1 --max-rel 5 --fps 60
```

Notes:

```text
No re-zero.
connect(calibrate=False).
Default mirror flips all joints except joint_4 and gripper.
Stage both arms in same reference posture before starting.
```

## Quest 3 Teleop

Relevant doc:

```bash
docs/openarm-quest3-lerobot-setup.md
```

oculus_reader installed at:

```bash
/home/arms/oculus_reader
```

Check Quest:

```bash
adb devices
```

Start reader:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
python /home/arms/oculus_reader/oculus_reader/reader.py
```

Important:

```text
Quest path still needs button-key verification, frame alignment, clutch logic, and IK.
Do not assume Quest teleop is safe until those are verified.
```

## Repo Changes Made During Session

Modified:

```text
README.md
```

Added scripts:

```text
scripts/camera_mjpeg_stream.py
scripts/disable_torque.py
scripts/open_gripper_small.py
```

Check status on robot:

```bash
cd /home/arms/hsi-pre-grasp
git status --short
```

Expected relevant status:

```text
M README.md
?? scripts/camera_mjpeg_stream.py
?? scripts/disable_torque.py
?? scripts/open_gripper_small.py
```

## Quick Start For Next Agent

1. SSH over direct LAN:

```bash
ssh arms@10.10.10.2
```

Wi-Fi fallback:

```bash
ssh arms@192.168.31.50
```

2. Enter repo:

```bash
cd /home/arms/hsi-pre-grasp
source .venv/bin/activate
```

3. Read operator section:

```bash
sed -n '/## Operator command cheat sheet/,/## Safety/p' README.md
```

4. Check CAN:

```bash
ip -details link show can0
ip -details link show can1
```

5. Check motors only if 24 V is ON and someone is at the rig:

```bash
lerobot-setup-can --mode=test --interfaces=can0,can1
```

6. Stop holding torque if needed:

```bash
python scripts/disable_torque.py --port can0 --side right
python scripts/disable_torque.py --port can1 --side left
```
