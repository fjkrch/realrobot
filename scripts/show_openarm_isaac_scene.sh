#!/usr/bin/env bash
set -euo pipefail

# GUI inspection helper. RTX is the normal Isaac Sim path; PXR remains available
# as ISAAC_VIEWER_RENDERER=pxr if a future driver update regresses RTX startup.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STEPS="${1:-200000}"
if [ "$#" -gt 0 ]; then
  shift
fi
VIEWER_CAMERA="${ISAAC_VIEWER_CAMERA:-wide}"
VIEWER_RENDERER="${ISAAC_VIEWER_RENDERER:-rtx}"

exec scripts/isaaclab_python.sh synthetic_smolvla/scripts/make_scene.py \
  --config synthetic_smolvla/configs/scene_openarm_real_table_zero_train_reachable_left_v1.yaml \
  --steps "${STEPS}" \
  --device "${ISAACLAB_DEVICE:-cuda:0}" \
  --viewer-renderer "${VIEWER_RENDERER}" \
  --viewer-camera "${VIEWER_CAMERA}" \
  --rendering-mode "${ISAAC_RENDERING_MODE:-performance}" \
  --kit-args "${ISAAC_KIT_ARGS:---/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/app/renderer/resolution/width=960 --/app/renderer/resolution/height=540 --/app/window/width=1100 --/app/window/height=760}" \
  "$@"
