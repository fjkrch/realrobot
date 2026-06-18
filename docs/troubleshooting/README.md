# Troubleshooting README

Start here when something fails.

## Symptom Routes

| Symptom | Read |
|---|---|
| Jetson cannot reach laptop mirror | [../sim/REAL_JETSON_TO_ISAAC_SIMULATION.md](../sim/REAL_JETSON_TO_ISAAC_SIMULATION.md), section "Troubleshooting" |
| Port `8765` already in use | [../sim/JETSON_ISAACLAB_MIRROR.md](../sim/JETSON_ISAACLAB_MIRROR.md) and check old mirror servers |
| Isaac imports fail | [../sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md](../sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md), "Python cannot import Isaac Lab modules" |
| Isaac GUI slow or unstable | [../sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md](../sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md), "GUI is slow or unstable" |
| Sim robot resets or jumps | [../sim/DEFAULT_OPENARM_ISAAC_MIRROR.md](../sim/DEFAULT_OPENARM_ISAAC_MIRROR.md), use clean mirror instead of task mirror |
| Joint command refused by limit | [../reference/OPENARM_JOINT_LIMITS.md](../reference/OPENARM_JOINT_LIMITS.md) |
| CAN does not see motors | [../real/REAL_ROBOT_MOVE.md](../real/REAL_ROBOT_MOVE.md), "Troubleshooting" |
| Handshake failed for a joint | [../real/REAL_ROBOT_MOVE.md](../real/REAL_ROBOT_MOVE.md), `Handshake failed ... motors did not respond` |
| Real joint movement clamps each step | [../real/REAL_ROBOT_MOVE.md](../real/REAL_ROBOT_MOVE.md), `Relative goal position magnitude had to be clamped` |
| Cube pick pose out of range | [../real/PICK_CUBE_SIM_TO_REAL.md](../real/PICK_CUBE_SIM_TO_REAL.md), teaching and tuning sections |
| Need robot IPs, CAN, camera, Quest notes | [../real/real_robot_calibration.md](../real/real_robot_calibration.md) |

## Common Commands

Check if laptop mirror port is free:

```bash
ss -ltnp 'sport = :8765'
```

Start clean Isaac mirror:

```bash
cd /home/chyanin/Desktop/realrobot
bash run_default_openarm_mirror.txt
```

Check mirror from Jetson:

```bash
python scripts/jetson_isaaclab_command.py --status
```

Stop real robot torque from Jetson:

```bash
python scripts/disable_torque.py --port can0 --side right
python scripts/disable_torque.py --port can1 --side left
```

## Escalation Rule

If a command may move the real robot and you are not sure it is safe, stop and
read [../real/README.md](../real/README.md) before continuing.
