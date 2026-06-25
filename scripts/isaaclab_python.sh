#!/usr/bin/env bash
set -euo pipefail

# Thin repo-side wrapper for Isaac Lab Python scripts.
# Equivalent to:
#   cd "$ISAACLAB_ROOT"
#   CONDA_PREFIX="$ISAACLAB_CONDA_PREFIX" ./isaaclab.sh -p ...

if [ -z "${ISAACLAB_ROOT:-}" ]; then
  if [ -d "/home/chayanin/Downloads/IsaacLab" ]; then
    ISAACLAB_ROOT="/home/chayanin/Downloads/IsaacLab"
  else
    ISAACLAB_ROOT="/home/chyanin/IsaacLab"
  fi
fi
ISAACLAB_CONDA_ENV="${ISAACLAB_CONDA_ENV:-env_isaaclab}"
if [ -z "${CONDA_ROOT:-}" ]; then
  if [ -d "/home/chayanin/Downloads/miniforge3" ]; then
    CONDA_ROOT="/home/chayanin/Downloads/miniforge3"
  else
    CONDA_ROOT="/home/chyanin/miniconda3"
  fi
fi
ISAACLAB_CONDA_PREFIX="${ISAACLAB_CONDA_PREFIX:-${CONDA_ROOT}/envs/${ISAACLAB_CONDA_ENV}}"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 path/to/script.py [args...]" >&2
  exit 2
fi

args=("$@")
if [ -f "${args[0]}" ]; then
  args[0]="$(realpath "${args[0]}")"
fi

if [ -z "${TERM:-}" ] || [ "${TERM}" = "dumb" ]; then
  export TERM=xterm
fi
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Isaac Sim camera rendering is RTX/Vulkan based. On hybrid laptops the display
# server may default OpenGL to the Intel GPU while Vulkan/CUDA use NVIDIA, which
# breaks non-RTX Hydra tests and can also confuse renderer selection. Enable
# PRIME offload automatically when the NVIDIA provider is visible; set
# ISAACLAB_NVIDIA_OFFLOAD=0 to disable.
ISAACLAB_NVIDIA_OFFLOAD="${ISAACLAB_NVIDIA_OFFLOAD:-auto}"
if [ "${ISAACLAB_NVIDIA_OFFLOAD}" != "0" ]; then
  if [ "${ISAACLAB_NVIDIA_OFFLOAD}" = "1" ] || xrandr --listproviders 2>/dev/null | grep -q "NVIDIA-G0"; then
    export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
    export __NV_PRIME_RENDER_OFFLOAD_PROVIDER="${__NV_PRIME_RENDER_OFFLOAD_PROVIDER:-NVIDIA-G0}"
    export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
    export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
  fi
fi

if [ ! -x "${ISAACLAB_CONDA_PREFIX}/bin/python" ]; then
  echo "Missing Python for IsaacLab conda env: ${ISAACLAB_CONDA_PREFIX}/bin/python" >&2
  exit 2
fi

cd "${ISAACLAB_ROOT}"
if [ "${ISAACLAB_DRY_RUN:-0}" = "1" ]; then
  printf 'cd %q && ' "${ISAACLAB_ROOT}"
  printf '%q ' env \
    "CONDA_PREFIX=${ISAACLAB_CONDA_PREFIX}" \
    "CONDA_DEFAULT_ENV=${ISAACLAB_CONDA_ENV}" \
    "PATH=${ISAACLAB_CONDA_PREFIX}/bin:${PATH}" \
    ./isaaclab.sh -p "${args[@]}"
  printf '\n'
  exit 0
fi

export CONDA_PREFIX="${ISAACLAB_CONDA_PREFIX}"
export CONDA_DEFAULT_ENV="${ISAACLAB_CONDA_ENV}"
export PATH="${ISAACLAB_CONDA_PREFIX}/bin:${PATH}"
exec ./isaaclab.sh -p "${args[@]}"
