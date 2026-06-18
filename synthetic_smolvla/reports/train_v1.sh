#!/usr/bin/env bash
set -euo pipefail
/home/chyanin/miniconda3/envs/env_isaaclab/bin/python -m lerobot.scripts.lerobot_train --dataset.repo_id local/openarm_synth_v1 --dataset.root /home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_synth_v1 --policy.type smolvla --policy.device cuda --policy.push_to_hub false --policy.repo_id local/smolvla_openarm_synth_v1 --output_dir /home/chyanin/Desktop/realrobot/synthetic_smolvla/checkpoints/smolvla_openarm_synth_v1 --batch_size 4 --steps 3000 --save_freq 3000 --log_freq 10 --num_workers 0 --wandb.enable false
