#!/usr/bin/env bash
# Collect ONE clean success at each requested robot height on the photo-clean scene.
# Runs strictly one Isaac process at a time (8GB GPU) and upsamples each kept
# episode to <=2 deg/command. Diagnostic / approval run: one success per height.
set -uo pipefail
cd /home/chayanin/Desktop/Robot/realrobot

PYTHON=/home/chayanin/Downloads/miniforge3/envs/env_isaaclab/bin/python
HEIGHTS=("125 125cm 26000" "122.5 122p5cm 27000" "120 120cm 28000" "117.5 117p5cm 29000" "115 115cm 30000")
CFGDIR=synthetic_smolvla/configs/generated_height_sweep_photo_clean_v1
OUTBASE=synthetic_smolvla/datasets/openarm_photo_clean_v1_one_per_height
REPBASE=synthetic_smolvla/reports/photo_clean_v1_one_per_height
mkdir -p "$REPBASE"

"$PYTHON" synthetic_smolvla/scripts/collect_height_sweep_successes.py \
  --base-config synthetic_smolvla/configs/scene_openarm_real_photo_left_centered_clean_v1.yaml \
  --heights-cm 125,122.5,120,117.5,115 \
  --height-config-dir "$CFGDIR" \
  --report-dir "$REPBASE/generated_height_configs" \
  --target-quotas 1,0,0,0 \
  --dry-run --no-merge >/dev/null

for entry in "${HEIGHTS[@]}"; do
  read -r H TAG SEED <<< "$entry"
  echo "==== HEIGHT $H cm (tag $TAG seed $SEED) ===="
  CFG="$CFGDIR/scene_openarm_real_photo_left_centered_clean_v1_h${TAG}.yaml"
  DS="$OUTBASE/h${TAG}"
  REP="$REPBASE/h${TAG}"
  mkdir -p "$REP"
  rm -rf "$DS"
  # defensive: ensure no orphaned Isaac process is holding the 8GB GPU
  pkill -9 -f "collect_dense_isaac_dataset.py" 2>/dev/null || true
  sleep 3
  scripts/isaaclab_python.sh synthetic_smolvla/scripts/collect_dense_isaac_dataset.py \
    --config "$CFG" \
    --dataset-root "$DS" \
    --repo-id "local/openarm_photo_clean_v1_h${TAG}" \
    --manifest "$REP/manifest.jsonl" \
    --report "$REP/collect.md" \
    --sample-frame-dir "$REP/samples" \
    --num-envs 4 --rounds 8 --seed "$SEED" \
    --target-weights 1,1,1,1 --max-keep 1 \
    --fps 10 --substeps 20 \
    --record-zero-to-init --zero-init-steps 20 --zero-start-gripper-deg 0 --init-gripper-deg -50 \
    --approach-steps 20 --descend-steps 20 --close-steps 16 --lift-steps 24 --hold-steps 0 \
    --above-offset-m 0.12 --lift-offset-m 0.08 --lift-threshold-m 0.05 \
    --grasp-z-offset-m 0.01 \
    --grasp-close-deg -13.0 --open-gripper-deg -50.0 --max-gripper-close-deg -13.0 \
    --gripper-close-range-deg -17 -13 \
    --max-action-step-deg 0 --action-clip-tol-deg 180.0 --early-stop-on-lift \
    --jitter-x-m 0.005 --jitter-y-m 0.003 \
    --overwrite 2>&1 | grep -v "PhysX error\|gpu.foundation\|omni.fabric" | tail -4
  # upsample the kept episode(s) for this height
  if [ -d "$DS/episodes" ]; then
    "$PYTHON" synthetic_smolvla/scripts/upsample_episodes_slew.py \
      --input-root "$DS" --output-root "${DS}_upsampled" --max-step-deg 2.0 --fps 10 --overwrite 2>&1 | tail -2
  fi
  echo "---- done $H cm ----"
done
echo "ALL_HEIGHTS_DONE"
