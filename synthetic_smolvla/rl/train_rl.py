#!/usr/bin/env python3
"""Train (PPO / rsl_rl) the OpenArm target-conditioned pick policy.

This learns the closed-loop grasp the scripted IK oracle could not get reliably.
Simulation-only; no real robot.

Smoke test (instantiates env + runs 2 PPO iterations on a few envs):

    /home/chyanin/IsaacLab/isaaclab_python.sh \
      synthetic_smolvla/rl/train_rl.py --headless --smoke

Full training:

    /home/chyanin/IsaacLab/isaaclab_python.sh \
      synthetic_smolvla/rl/train_rl.py --headless --num-envs 2048 --max-iterations 1500

Checkpoints + tensorboard land in logs/rsl_rl/openarm_pick/<timestamp>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--max-iterations", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--experiment", default="openarm_pick")
    p.add_argument("--smoke", action="store_true", help="tiny run: 16 envs, 2 iterations, to prove the stack works")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.smoke:
        args.num_envs = 16
        args.max_iterations = 2

    # Isaac import paths (mirror scripts/make_scene.py).
    isaaclab_root = Path("/home/chyanin/IsaacLab")
    repo_root = Path(__file__).resolve().parents[2]
    for path in [
        isaaclab_root / "source" / "isaaclab",
        isaaclab_root / "source" / "isaaclab_assets",
        isaaclab_root / "source" / "isaaclab_tasks",
        isaaclab_root / "source" / "isaaclab_rl",
        repo_root,
        Path(__file__).resolve().parent,
    ]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from isaaclab.app import AppLauncher

    print(f"[train-rl] launching Isaac (num_envs={args.num_envs}, iters={args.max_iterations})", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import os
    from datetime import datetime

    import torch  # noqa: F401
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )

    from openarm_pick_env import OpenArmPickEnv, OpenArmPickEnvCfg

    env_cfg = OpenArmPickEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = OpenArmPickEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=8 if args.smoke else 24,
        max_iterations=args.max_iterations,
        save_interval=50,
        experiment_name=args.experiment,
        seed=args.seed,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.2,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.02,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    log_root = os.path.join(str(repo_root), "synthetic_smolvla", "rl", "logs", args.experiment)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ("_smoke" if args.smoke else ""))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train-rl] logging to {log_dir}", file=sys.stderr, flush=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"[train-rl] DONE. checkpoints in {log_dir}", flush=True)
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
