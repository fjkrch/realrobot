#!/usr/bin/env bash
set -euo pipefail

# Install convenience shortcuts into the Isaac Lab root on this laptop.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/chyanin/IsaacLab}"

mkdir -p "${ISAACLAB_ROOT}"

ln -sf "${repo_root}/scripts/isaaclab_python.sh" "${ISAACLAB_ROOT}/isaaclab_python.sh"
ln -sf "${repo_root}/run_default_openarm_mirror.txt" "${ISAACLAB_ROOT}/run_openarm_mirror.sh"

echo "Installed Isaac Lab shortcuts:"
echo "  ${ISAACLAB_ROOT}/isaaclab_python.sh -> ${repo_root}/scripts/isaaclab_python.sh"
echo "  ${ISAACLAB_ROOT}/run_openarm_mirror.sh -> ${repo_root}/run_default_openarm_mirror.txt"

