# Synthetic SmolVLA Pipeline Status

Mode: `full`

| Step | Status | Seconds |
|---|---:|---:|
| scene dry-run | ok | 0.0 |
| oracle acceptance manifest | ok | 1.4 |
| oracle acceptance report | ok | 0.0 |
| dataset v1 export | ok | 29.4 |
| train v1 prepare | ok | 6.2 |
| eval v1 manifest | ok | 0.1 |
| dataset v2 export | ok | 128.3 |
| train v2 prepare | ok | 7.0 |
| stress test report | ok | 0.3 |

## Commands

### scene dry-run

```bash
/home/chyanin/miniconda3/envs/env_isaaclab/bin/python synthetic_smolvla/scripts/make_scene.py --dry-run --manifest synthetic_smolvla/reports/scene_manifest.json
```
### oracle acceptance manifest

```bash
conda run -n env_isaaclab python synthetic_smolvla/scripts/collect_oracle_demos.py --episodes 100 --output synthetic_smolvla/reports/oracle_acceptance_manifest.jsonl
```
### oracle acceptance report

```bash
/home/chyanin/miniconda3/envs/env_isaaclab/bin/python synthetic_smolvla/scripts/eval_smolvla.py --manifest synthetic_smolvla/reports/oracle_acceptance_manifest.jsonl --output synthetic_smolvla/reports/oracle_acceptance.md
```
### dataset v1 export

```bash
conda run -n env_isaaclab python synthetic_smolvla/scripts/collect_oracle_demos.py --dataset-config synthetic_smolvla/configs/dataset_v1.yaml --episodes 1000 --export-lerobot --overwrite
```
### train v1 prepare

```bash
conda run -n env_isaaclab python synthetic_smolvla/scripts/train_smolvla.py --train-config synthetic_smolvla/configs/train_v1.yaml --output synthetic_smolvla/reports/train_v1_preflight.json --command-output synthetic_smolvla/reports/train_v1.sh --overwrite-output-dir
```
### eval v1 manifest

```bash
/home/chyanin/miniconda3/envs/env_isaaclab/bin/python synthetic_smolvla/scripts/eval_smolvla.py --manifest synthetic_smolvla/datasets/openarm_synth_v1/oracle_manifest.jsonl --output synthetic_smolvla/reports/eval_v1.md
```
### dataset v2 export

```bash
conda run -n env_isaaclab python synthetic_smolvla/scripts/collect_oracle_demos.py --dataset-config synthetic_smolvla/configs/dataset_v2.yaml --episodes 5000 --export-lerobot --overwrite
```
### train v2 prepare

```bash
conda run -n env_isaaclab python synthetic_smolvla/scripts/train_smolvla.py --train-config synthetic_smolvla/configs/train_v2.yaml --output synthetic_smolvla/reports/train_v2_preflight.json --command-output synthetic_smolvla/reports/train_v2.sh --overwrite-output-dir
```
### stress test report

```bash
/home/chyanin/miniconda3/envs/env_isaaclab/bin/python synthetic_smolvla/scripts/stress_test.py --episodes 1000 --output synthetic_smolvla/reports/stress_test_v2.md
```
