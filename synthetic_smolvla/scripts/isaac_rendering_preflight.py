#!/usr/bin/env python3
"""Report whether this workstation is ready for Isaac Lab RTX camera collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "synthetic_smolvla/configs/scene_openarm_real_table_zero_train_reachable_left_v1.yaml"
BAD_USER_MDL_CACHE = Path.home() / ".local/share/ov/data/exts/v2/omni.kit.usd.mdl-c5413828e5409d2b"
VALIDATED_DRIVER_MAJOR = "580"
KNOWN_SCENEDB_CRASH_DRIVER = "595.71.05"


def _run_text(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError as exc:
        return 127, str(exc)
    return int(proc.returncode), proc.stdout.strip()


def _gpu_info() -> list[dict[str, object]]:
    code, out = _run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if code != 0 or not out:
        return []
    gpus: list[dict[str, object]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, driver, memory_mib = parts
        try:
            memory = int(memory_mib)
        except ValueError:
            memory = 0
        gpus.append({"name": name, "driver": driver, "memory_mib": memory})
    return gpus


def _hybrid_provider() -> bool:
    code, out = _run_text(["xrandr", "--listproviders"])
    return code == 0 and "NVIDIA-G0" in out


def _driver_package_notes() -> list[str]:
    notes: list[str] = []
    for package in ("nvidia-driver-580", "nvidia-driver-580-open"):
        code, out = _run_text(["apt-cache", "policy", package])
        if code == 0 and "Candidate:" in out:
            for line in out.splitlines():
                if "Candidate:" in line:
                    notes.append(f"{package} candidate: {line.split('Candidate:', 1)[1].strip()}")
                    break
    return notes


def _secure_boot_enabled() -> bool:
    code, out = _run_text(["mokutil", "--sb-state"])
    return code == 0 and "SecureBoot enabled" in out


def _nvidia_key_rejected() -> bool:
    code, out = _run_text(["journalctl", "-k", "-b", "--no-pager"])
    return code == 0 and "Loading of module with unavailable key is rejected" in out


def _nvidia_module_signer() -> str | None:
    code, out = _run_text(["modinfo", "nvidia"])
    if code != 0:
        return None
    for line in out.splitlines():
        if line.startswith("signer:"):
            return line.split(":", 1)[1].strip()
    return None


def _signed_module_package_notes() -> list[str]:
    code, kernel = _run_text(["uname", "-r"])
    if code != 0 or not kernel:
        return []
    packages = [
        f"linux-modules-nvidia-580-open-{kernel}",
        "linux-modules-nvidia-580-open-generic-hwe-24.04",
    ]
    notes: list[str] = []
    for package in packages:
        code, out = _run_text(["apt-cache", "policy", package])
        if code != 0 or "Candidate:" not in out:
            continue
        for line in out.splitlines():
            if "Candidate:" in line:
                notes.append(f"{package} candidate: {line.split('Candidate:', 1)[1].strip()}")
                break
    return notes


def _smoke_command(args: argparse.Namespace) -> list[str]:
    return [
        *shlex.split(args.python),
        str(REPO_ROOT / "synthetic_smolvla/scripts/make_scene.py"),
        "--config",
        str(REPO_ROOT / args.config),
        "--headless",
        "--steps",
        str(args.steps),
        "--save-camera-rgb",
        str(REPO_ROOT / args.output),
        "--rendering-mode",
        "performance",
        "--kit-args",
        "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/rtx/materialDb/syncLoads=true --/rtx/hydra/materialSyncLoads=true",
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="scripts/isaaclab_python.sh")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default="synthetic_smolvla/reports/renderer_fix/preflight_rtx_camera.ppm")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--run-smoke", action="store_true", help="run the official Isaac Lab RTX camera smoke test")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    gpus = _gpu_info()
    issues: list[str] = []
    notes: list[str] = []
    secure_boot = _secure_boot_enabled()
    nvidia_key_rejected = _nvidia_key_rejected()

    if not gpus:
        issues.append("nvidia-smi did not report an NVIDIA GPU")
        if secure_boot and nvidia_key_rejected:
            signer = _nvidia_module_signer()
            detail = f" signed by '{signer}'" if signer else ""
            issues.append(f"Secure Boot is rejecting the NVIDIA kernel module{detail}; install signed modules or enroll the MOK")
            notes.extend(_signed_module_package_notes())
            notes.append(
                "recommended Secure Boot repair: sudo apt install "
                "linux-modules-nvidia-580-open-generic-hwe-24.04 nvidia-driver-580-open nvidia-dkms-580-open-"
            )
    else:
        gpu = gpus[0]
        if int(gpu.get("memory_mib", 0)) < 16_000:
            issues.append(
                f"GPU VRAM is {gpu.get('memory_mib')} MiB; Isaac Sim docs list 16GB VRAM as the minimum RTX GPU memory"
            )
        driver = str(gpu.get("driver", ""))
        if driver == KNOWN_SCENEDB_CRASH_DRIVER:
            issues.append(
                f"driver is {driver}; this branch has reproduced Isaac Sim 5.1 RTX scenedb startup crashes locally"
            )
        elif driver.split(".", 1)[0] != VALIDATED_DRIVER_MAJOR:
            notes.append(
                f"driver is {driver}; Isaac Sim 5.1 docs list Linux driver 580.65.06 for production-branch users"
            )
    if _hybrid_provider():
        notes.append("NVIDIA-G0 PRIME provider detected; scripts/isaaclab_python.sh will enable NVIDIA render offload by default")
    if BAD_USER_MDL_CACHE.exists():
        issues.append(f"broken cached Kit MDL extension still exists: {BAD_USER_MDL_CACHE}")
    notes.extend(_driver_package_notes())
    notes.append("repair path: install the Ubuntu 580 driver package, reboot, then rerun this preflight with --run-smoke")

    smoke: dict[str, object] | None = None
    cmd = _smoke_command(args)
    if args.run_smoke:
        Path(REPO_ROOT / args.output).parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        smoke = {"command": cmd, "returncode": proc.returncode, "output_exists": (REPO_ROOT / args.output).exists()}
        if proc.returncode != 0:
            issues.append(f"official Isaac Lab RTX camera smoke failed with return code {proc.returncode}")

    report = {
        "ok": not issues,
        "gpus": gpus,
        "issues": issues,
        "notes": notes,
        "official_camera_smoke_command": cmd,
        "smoke": smoke,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
