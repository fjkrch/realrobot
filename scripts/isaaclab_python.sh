#!/usr/bin/env bash
set -euo pipefail

# Thin repo-side wrapper for Isaac Lab Python scripts.
# Equivalent to:
#   cd /home/chyanin/IsaacLab
#   CONDA_PREFIX=/home/chyanin/miniconda3/envs/env_isaaclab ./isaaclab.sh -p ...

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/chyanin/IsaacLab}"
ISAACLAB_CONDA_ENV="${ISAACLAB_CONDA_ENV:-env_isaaclab}"
CONDA_ROOT="${CONDA_ROOT:-/home/chyanin/miniconda3}"
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
