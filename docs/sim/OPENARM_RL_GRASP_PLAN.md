# OpenArm RL Grasp Plan (learned pick oracle)

Simulation-only plan to replace the flaky scripted IK oracle with an RL policy
that reliably picks the requested object. No real robot is involved.

## Why RL

The scripted oracle pipeline (`synthetic_smolvla/scripts/oracle_pick_ik.py`)
reaches the object but grasps inconsistently: position-only differential IK lets
the wrist drift to a position-dependent orientation. Measured results:

- Fixed object positions: cubes 5/6, ball 0/2.
- Randomized 100/object (400 episodes): **157/400 (39.3%)**; cubes ~42-48%,
  ball 20%, 0 wrong-object lifts.

Hand-tuning the wrist failed (see `synthetic_smolvla/README.md` ->
"Optimization attempts"): pulling objects closer regressed cubes 83%->33%, and
locking a *captured* wrist orientation regressed to 0/12. The orientation has to
be *learned*, not guessed — hence RL.

## Approach

PPO (rsl_rl) on an Isaac Lab `DirectRLEnv`, modelled on the stock
`isaaclab_tasks/direct/franka_cabinet` task. The policy learns a closed-loop
joint controller for the right arm + gripper, rewarded by the requested object's
true height rise in physics, so it discovers the grasp wrist on its own.

Implementation: `synthetic_smolvla/rl/openarm_pick_env.py` (env) and
`synthetic_smolvla/rl/train_rl.py` (trainer).

### Environment

| Aspect | Choice |
|---|---|
| Robot | bimanual OpenArm, base at (0,0,0.40), right arm active (left held at rest) |
| Objects | orange ball + red/green/blue cubes in the reachable pocket (x=0.30, y in [-0.26,-0.08], z=0.55), small per-episode xy jitter |
| Target | random per episode; one-hot in the observation (language/target conditioned) |
| Action (8) | 7 right-arm joint deltas (integrated, clamped to soft limits) + 1 gripper open/close |
| Observation (34) | scaled right-arm joint pos/vel, gripper width, TCP pos, each object's pos relative to TCP, target one-hot |
| Reward | reach `1/(1+d^2)` + near bonus + measured target lift (30x) + grip-when-near + success bonus - wrong-object lift penalty (20x) - action penalty |
| Termination | target lift > 0.08 m (success), target falls off table, or 5 s timeout |

### Reward-scale knobs (`OpenArmPickEnvCfg`)

`reach_reward_scale`, `near_bonus`, `lift_reward_scale`, `success_bonus`,
`wrong_penalty_scale`, `action_penalty_scale`, `grip_reward_scale`. If training
plateaus, raise `lift_reward_scale`/`success_bonus` or lower `action_penalty_scale`.

## Status

- [x] DirectRLEnv implemented (`rl/openarm_pick_env.py`).
- [x] PPO trainer implemented (`rl/train_rl.py`, rsl_rl `OnPolicyRunner`).
- [x] Smoke test PASSED: 16 envs, 2 PPO iterations, checkpoints saved
      (`synthetic_smolvla/reports/logs/rl_smoke.log`).
- [ ] Full training run (2048 envs, ~1500 iterations) to convergence.
- [ ] `play_rl.py`: roll out the checkpoint per target, write a measured
      success report, compare to the IK oracle's 39.3%.
- [ ] Use the trained policy as the SmolVLA demonstration oracle; regenerate
      datasets with honest, RL-quality labels.

## Commands

Smoke test (proves the stack works):

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/rl/train_rl.py --headless --smoke --device cuda:0
```

Full training:

```bash
/home/chyanin/IsaacLab/isaaclab_python.sh \
  synthetic_smolvla/rl/train_rl.py --headless --num-envs 2048 --max-iterations 1500
tensorboard --logdir synthetic_smolvla/rl/logs/openarm_pick
```

## Safety boundary

Simulation only. Do not run `scripts/move_joint.py`, `scripts/move_arm.py`,
`scripts/pick_cube.py --real`, or Jetson-side gripper commands for this work.
