# Reference README

Use this folder for static facts: limits, paths, addresses, script ownership,
and file inventory.

## Key References

| Need | File |
|---|---|
| Folder inventory | [../../SUMMARY.md](../../SUMMARY.md) |
| Joint limits and sim clamp values | [OPENARM_JOINT_LIMITS.md](OPENARM_JOINT_LIMITS.md) |
| LAN, CAN, camera, Quest notes | [../real/real_robot_calibration.md](../real/real_robot_calibration.md) |
| Private full handoff | `../real/OPENARM_ROBOT_HANDOFF.md` |
| Isaac Lab local paths and environment | [../sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md](../sim/ISAACLAB_ISAACSIM_LOCAL_GUIDE.md) |

## Machine Addresses

| Machine | Address |
|---|---|
| Laptop direct LAN | `10.10.10.1` |
| Jetson direct LAN | `10.10.10.2` |
| Jetson Wi-Fi fallback | `192.168.31.50` |

## Important Paths

| Path | Purpose |
|---|---|
| `/home/chyanin/Desktop/realrobot` | This laptop workspace. |
| `/home/chyanin/IsaacLab` | Isaac Lab installation. |
| `/home/chyanin/IsaacLab/source/hsi_pregrasp_refusal` | Isaac task extension project. |
| `/home/arms/hsi-pre-grasp` | Jetson robot repo. |
| `/home/arms/hsi-pre-grasp/scripts` | Jetson robot scripts. |

## Current Limits

Read [OPENARM_JOINT_LIMITS.md](OPENARM_JOINT_LIMITS.md) before
changing any real or simulated joint clamp logic.
