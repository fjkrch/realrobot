# SmolVLA Real Mirror Preflight Audit

Date: 2026-06-17

## Status

Real robot motion was **not run** in this session. The user did not provide the
fresh required real-motion confirmation phrase, so the implementation remained
in sim/dry-run mode only.

## Implemented Guards

- `interactive_vla_isaac.py` defaults to sim-only.
- Dry-run tracing is enabled with `--mirror-dry-run PATH` and never contacts the
  robot.
- Real access requires `--real-confirm "I am at the robot with e-stop ready"`.
- Real motion additionally requires `--mirror-real --prepare-real-start-pose`.
- Preparing the real start pose reads
  `robot.reset_pose_deg.<side>` from
  `synthetic_smolvla/configs/scene_openarm_dense_isaac_camera_v1.yaml`.
- The real helper reads current state, validates live limits, moves with
  `connect(calibrate=False)`, reads back state, and aborts on excessive error.
- Real gripper commands are disabled by default.
- Policy targets are clamped to the sim contract, checked for finite values,
  rate-limited, and split into intermediate targets no larger than
  `--max-joint-delta-deg`.
- A watchdog self-test passed:
  `python3 scripts/openarm_safe_real_mirror.py --self-test-watchdog`.

## Dry-Run Evidence

- Trace:
  `synthetic_smolvla/reports/interactive_vla_mirror_dry_run_red_trace.jsonl`
- Audit:
  `synthetic_smolvla/reports/interactive_vla_mirror_dry_run_red_audit.md`
- Task: `pick up the red cube`
- Steps: 5
- Audit status: PASS
- Max observed inter-command arm delta: 1.744373 deg
- Max allowed audit delta: 3.0 deg
- First target delta from configured start pose: 12.169716 deg
- First target tolerance: 15.0 deg
- Gripper sent to real: false

## Real Checklist Not Yet Executed

These items remain pending until the operator is physically at the robot with
e-stop ready and gives the exact confirmation phrase:

- Robot connection reachable through the guarded helper.
- Current real joint state readable.
- Start pose command sent to the real arm.
- Post-move readback audited against the configured start pose.
- First live mirror target checked against current real state.

## Next Safest Step

Run the documented preflight-only command in
`docs/real/SMOLVLA_SIM_TO_REAL_MIRROR.md`. If connection, readback, or start-pose
motion fails, stop there, keep the mirror disabled, save the traceback/log, and
fix that specific failure before attempting a typed task.
