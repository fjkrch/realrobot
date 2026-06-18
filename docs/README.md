# Docs Folder README

This folder is the organized documentation home for the project.

Root `README.md` and `SUMMARY.md` stay at the top level as the entry points.
Topic docs live here so another agent can go straight to the right area:
simulation, real robot motion, troubleshooting, or reference data.

## Folders

| Folder | Use it when |
|---|---|
| [agent-handoff](agent-handoff/README.md) | A new agent needs orientation and safety rules. |
| [sim](sim/README.md) | The work is Isaac-only or Jetson-to-Isaac mirroring. |
| [../synthetic_smolvla](../synthetic_smolvla/README.md) | The work is synthetic-only SmolVLA data, training, or evaluation scaffolding. |
| [real](real/README.md) | The work may move physical motors. |
| [troubleshooting](troubleshooting/README.md) | Something fails and you need a symptom route. |
| [reference](reference/README.md) | You need limits, paths, addresses, or static facts. |

## Recommended Agent Flow

1. Read [../README.md](../README.md).
2. Read [agent-handoff/README.md](agent-handoff/README.md).
3. Pick one task folder:
   - [sim](sim/README.md)
   - [../synthetic_smolvla](../synthetic_smolvla/README.md)
   - [real](real/README.md)
   - [troubleshooting](troubleshooting/README.md)
   - [reference](reference/README.md)
4. Only then edit scripts or run commands.
