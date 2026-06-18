# Isaac Lab / Isaac Sim Local Guide

This guide is for running Isaac Sim and Isaac Lab on this computer.

## Local Paths

| Item | Path / Name |
|---|---|
| Isaac Lab root | `/home/chyanin/IsaacLab` |
| Isaac Lab launcher | `/home/chyanin/IsaacLab/isaaclab.sh` |
| Conda environment | `env_isaaclab` |
| Main logs folder | `/home/chyanin/IsaacLab/logs` |
| Built-in examples | `/home/chyanin/IsaacLab/scripts` |
| Easy Python shortcut | `/home/chyanin/IsaacLab/isaaclab_python.sh` |
| OpenArm mirror shortcut | `/home/chyanin/IsaacLab/run_openarm_mirror.sh` |

If those two shortcuts are missing, recreate them from the repo:

```bash
cd /home/chyanin/Desktop/realrobot
bash scripts/install_isaaclab_shortcuts.sh
```

## Quick Health Check

Check that the conda environment works:

```bash
cd /home/chyanin/IsaacLab
conda run -n env_isaaclab python -c "import sys; print(sys.executable)"
```

Check that Isaac Lab can launch Python:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh -c "print('Isaac Lab Python OK')"
```

Check the launcher options:

```bash
cd /home/chyanin/IsaacLab
TERM=xterm ./isaaclab.sh --help
```

## Open Isaac Sim UI

Start Isaac Sim directly:

```bash
cd /home/chyanin/IsaacLab
TERM=xterm ./isaaclab.sh -s
```

If you want to launch it through the conda environment:

```bash
cd /home/chyanin/IsaacLab
TERM=xterm conda run -n env_isaaclab ./isaaclab.sh -s
```

## Run Isaac Lab Python Scripts

Use the shortcut instead of plain `python` when a script needs Isaac Sim / Isaac Lab modules:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh path/to/script.py
```

From this repo, the same wrapper is:

```bash
cd /home/chyanin/Desktop/realrobot
scripts/isaaclab_python.sh path/to/script.py
```

The wrapper automatically changes into `/home/chyanin/IsaacLab`, uses
`env_isaaclab`, and runs:

```bash
conda run -n env_isaaclab ./isaaclab.sh -p ...
```

To print the command without launching Isaac:

```bash
ISAACLAB_DRY_RUN=1 ./isaaclab_python.sh path/to/script.py
```

For headless runs, most Isaac Lab scripts accept `--headless`:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh path/to/script.py --headless
```

Repo wrapper equivalent:

```bash
scripts/isaaclab_python.sh path/to/script.py --headless
```

## Built-In Demo Scripts

Run a basic empty simulation:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/tutorials/00_sim/create_empty.py
```

Run a basic scene demo:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/tutorials/02_scene/create_scene.py
```

Run an arm demo:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/demos/arms.py
```

Run a camera/sensor demo:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/demos/sensors/cameras.py
```

## List Available Tasks

List registered Isaac Lab environments:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/environments/list_envs.py
```

Run a random-action agent on a task:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/environments/random_agent.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 16
```

Run the same kind of test headless:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/environments/random_agent.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 16 \
  --headless
```

## Train And Play RL Policies

RSL-RL training example:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 64 \
  --headless
```

RSL-RL play example:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Cartpole-v0
```

Other learning backends are under:

```text
/home/chyanin/IsaacLab/scripts/reinforcement_learning/
```

Common folders:

| Backend | Folder |
|---|---|
| RSL-RL | `scripts/reinforcement_learning/rsl_rl` |
| RL-Games | `scripts/reinforcement_learning/rl_games` |
| SKRL | `scripts/reinforcement_learning/skrl` |
| Stable-Baselines3 | `scripts/reinforcement_learning/sb3` |

## Useful Simulation Flags

Common flags used by many Isaac Lab scripts:

| Flag | Meaning |
|---|---|
| `--headless` | Run without the GUI viewer |
| `--num_envs N` | Number of parallel environments |
| `--task TASK_NAME` | Environment/task ID |
| `--device cuda:0` | Use the NVIDIA GPU |
| `--seed N` | Set random seed |
| `--max_iterations N` | Limit RL training iterations, when supported |

Example:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh scripts/environments/random_agent.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num_envs 8 \
  --device cuda:0
```

## Logs And Outputs

Most training or simulation logs are written under:

```text
/home/chyanin/IsaacLab/logs
```

Some scripts also write under:

```text
/home/chyanin/IsaacLab/outputs
```

To inspect recent log folders:

```bash
cd /home/chyanin/IsaacLab
find logs -maxdepth 3 -type d | sort | tail -50
```

## VS Code

Generate Isaac Lab VS Code settings:

```bash
cd /home/chyanin/IsaacLab
./isaaclab.sh -v
```

Then open:

```bash
code /home/chyanin/IsaacLab
```

## Common Problems

### Terminal says `dumb` or reset warnings

Use:

```bash
export TERM=xterm
```

Or prefix commands:

```bash
TERM=xterm ./isaaclab_python.sh scripts/tutorials/00_sim/create_empty.py
```

### Python cannot import Isaac Lab modules

Run the script through `isaaclab.sh -p`:

```bash
cd /home/chyanin/IsaacLab
./isaaclab_python.sh your_script.py
```

### GUI is slow or unstable

Try headless mode:

```bash
--headless
```

Use fewer environments:

```bash
--num_envs 1
```

### GPU check

```bash
nvidia-smi
```

This computer has an NVIDIA RTX GPU available for Isaac Sim. Prefer `cuda:0` for GPU runs.

## Daily Starting Template

Use this when starting a new Isaac Lab session:

```bash
cd /home/chyanin/IsaacLab
export TERM=xterm

conda run -n env_isaaclab python -c "import sys; print(sys.executable)"
./isaaclab.sh --help
```

Then run either the simulator UI:

```bash
./isaaclab.sh -s
```

or an Isaac Lab script:

```bash
./isaaclab_python.sh scripts/tutorials/00_sim/create_empty.py
```
