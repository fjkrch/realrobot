#!/usr/bin/env python3
"""Receive Jetson text commands and mirror them into Isaac Lab.

This server is intentionally simulation-first. It accepts a small HTTP request
from the Jetson, then launches the existing Isaac Lab OpenArm word-control demo
on the laptop. It does not command the physical robot.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


DEFAULT_ISAACLAB_ROOT = Path("/home/chyanin/IsaacLab")
DEFAULT_RUNNER = DEFAULT_ISAACLAB_ROOT / "source/hsi_pregrasp_refusal/run_openarm_word_control_demo.txt"
STOP_COMMANDS = {"stop", "hold", "wait", "pause", "stay", "do nothing", "no move", "dont move", "don't move"}


def normalize_command(command: str) -> str:
    """Return a compact command string safe to pass as an environment value."""
    normalized = " ".join(str(command).replace("\x00", "").strip().split())
    if not normalized:
        return "stop"
    if len(normalized) > 200:
        raise ValueError("Command is too long; keep it under 200 characters.")
    return normalized


def command_mode(command: str) -> str:
    """Mirror the Isaac word-control routing so responses are predictable."""
    text = normalize_command(command).lower()
    if text in STOP_COMMANDS:
        return "stop"
    if gripper_target_deg(command) is not None:
        return "gripper_target"
    if "open" in text and "gripper" in text:
        return "open_gripper"
    if "close" in text and "gripper" in text:
        return "close_gripper"
    if any(word in text for word in ("pick", "grab", "grasp", "lift")):
        return "pick"
    return "stop"


def gripper_target_deg(command: str) -> float | None:
    """Extract a numeric gripper degree target from command text."""
    text = normalize_command(command).lower()
    if "gripper" not in text:
        return None
    tokens = text.replace("=", " ").replace(",", " ").split()
    for idx, token in enumerate(tokens):
        if token in {"target", "target-deg", "target_deg", "deg", "degree", "degrees"}:
            search = tokens[idx + 1 :] if token != "deg" else tokens[:idx]
            for candidate in search:
                try:
                    return float(candidate)
                except ValueError:
                    continue
    for token in tokens:
        try:
            return float(token)
        except ValueError:
            continue
    return None


@dataclass
class Job:
    """One Isaac Lab mirror command."""

    id: str
    command: str
    mode: str
    status: str
    created_at: float
    started_at: float | None = None
    ended_at: float | None = None
    pid: int | None = None
    returncode: int | None = None
    log_path: str = ""
    message: str = ""
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False, compare=False)

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("process", None)
        return data


class BridgeState:
    """Thread-safe state for the command bridge."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.current_job_id: str | None = None
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    def _active_job_locked(self) -> Job | None:
        if self.current_job_id is None:
            return None
        job = self.jobs.get(self.current_job_id)
        if job is not None and job.status in {"queued", "running"}:
            return job
        return None

    def submit(self, command: str) -> tuple[int, dict[str, Any]]:
        try:
            normalized = normalize_command(command)
        except ValueError as exc:
            return 400, {"ok": False, "error": str(exc)}

        mode = command_mode(normalized)
        if mode == "stop":
            stopped = self.stop_current(reason=f"stop command received: {normalized!r}")
            if stopped["stopped"]:
                return 200, {"ok": True, "mode": mode, **stopped}

        with self.lock:
            running = self._active_job_locked()
            if running is not None:
                return 409, {
                    "ok": False,
                    "error": "Isaac Lab mirror is already running. Send command 'stop' first.",
                    "running_job": running.public(),
                }

            job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
            log_path = Path(self.args.log_dir) / f"{job_id}_{mode}.log"
            job = Job(
                id=job_id,
                command=normalized,
                mode=mode,
                status="queued",
                created_at=time.time(),
                log_path=str(log_path),
            )
            self.jobs[job.id] = job
            self.current_job_id = job.id

        thread = threading.Thread(target=self._run_job, args=(job.id,), daemon=True)
        thread.start()
        return 202, {"ok": True, "job": job.public()}

    def stop_current(self, *, reason: str) -> dict[str, Any]:
        with self.lock:
            job = self._active_job_locked()
            if job is None:
                return {"stopped": False, "message": "No running Isaac Lab mirror job."}
            process = job.process
            job.message = reason

        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=8)

        with self.lock:
            job.status = "stopped"
            job.ended_at = time.time()
            job.returncode = None if process is None else process.returncode
            self.current_job_id = None
            return {"stopped": True, "job": job.public()}

    def status(self) -> dict[str, Any]:
        with self.lock:
            current = self.jobs.get(self.current_job_id) if self.current_job_id else None
            recent = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)[:10]
            return {
                "ok": True,
                "current_job": current.public() if current else None,
                "recent_jobs": [job.public() for job in recent],
                "dry_run": bool(self.args.dry_run),
                "runner": str(self.args.runner),
                "isaaclab_root": str(self.args.isaaclab_root),
            }

    def job(self, job_id: str) -> tuple[int, dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return 404, {"ok": False, "error": f"Unknown job id: {job_id}"}
            return 200, {"ok": True, "job": job.public()}

    def _run_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job.status == "stopped":
                return
            job.status = "running"
            job.started_at = time.time()

        log_path = Path(job.log_path)
        env = os.environ.copy()
        env.update(
            {
                "COMMAND": job.command,
                "NUM_EPISODES": str(self.args.num_episodes),
                "MAX_STEPS": str(self.args.hold_steps if job.mode != "pick" else self.args.max_steps),
                "OPENARM_SETUP": self.args.openarm_setup,
                "CONTROL_ARM": self.args.control_arm,
                "OBJECT_X": str(self.args.object_x),
                "OBJECT_Y": self.args.object_y,
                "OBJECT_XY_NOISE": str(self.args.object_xy_noise),
                "PEDESTAL_HEIGHT": str(self.args.pedestal_height),
                "PEDESTAL_SIZE_XY": str(self.args.pedestal_size_xy),
                "POSITION_THRESHOLD": str(self.args.position_threshold),
                "PYTHONUNBUFFERED": "1",
            }
        )
        target_deg = gripper_target_deg(job.command)
        if target_deg is not None:
            env["GRIPPER_TARGET_DEG"] = f"{target_deg:.6f}"

        try:
            with log_path.open("wb") as log:
                header = (
                    f"[bridge] command={job.command!r} mode={job.mode} dry_run={self.args.dry_run}\n"
                    f"[bridge] runner={self.args.runner}\n"
                )
                log.write(header.encode("utf-8"))
                log.flush()
                if self.args.dry_run:
                    time.sleep(float(self.args.dry_run_delay))
                    returncode = 0
                else:
                    process = subprocess.Popen(
                        ["bash", str(self.args.runner)],
                        cwd=str(self.args.isaaclab_root),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    with self.lock:
                        job.process = process
                        job.pid = process.pid
                    returncode = process.wait()
        except Exception as exc:  # pragma: no cover - defensive runtime path
            with self.lock:
                job.status = "failed"
                job.message = str(exc)
                job.ended_at = time.time()
                self.current_job_id = None
            return

        with self.lock:
            if job.status != "stopped":
                job.returncode = returncode
                job.status = "completed" if returncode == 0 else "failed"
                job.ended_at = time.time()
                job.message = f"Isaac Lab mirror exited with return code {returncode}."
                self.current_job_id = None


def json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(state: BridgeState):
    class CommandHandler(BaseHTTPRequestHandler):
        server_version = "IsaacLabCommandBridge/1.0"

        def _authorized(self) -> bool:
            token = state.args.token or os.environ.get("BRIDGE_TOKEN", "")
            if not token:
                return True
            auth = self.headers.get("Authorization", "")
            header = self.headers.get("X-Bridge-Token", "")
            return header == token or auth == f"Bearer {token}"

        def _reject_unauthorized(self) -> bool:
            if self._authorized():
                return False
            json_response(self, 401, {"ok": False, "error": "Missing or invalid bridge token."})
            return True

        def _read_payload(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(content_length) if content_length else b""
            if not raw:
                return {}
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return json.loads(raw.decode("utf-8"))
            parsed = parse_qs(raw.decode("utf-8"))
            return {key: values[-1] for key, values in parsed.items()}

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            if self._reject_unauthorized():
                return
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/health"}:
                json_response(self, 200, {"ok": True, "service": "isaaclab-command-bridge"})
                return
            if parsed.path == "/status":
                json_response(self, 200, state.status())
                return
            if parsed.path.startswith("/job/"):
                code, payload = state.job(parsed.path.rsplit("/", 1)[-1])
                json_response(self, code, payload)
                return
            if parsed.path == "/command":
                command = parse_qs(parsed.query).get("command", ["stop"])[-1]
                code, payload = state.submit(command)
                json_response(self, code, payload)
                return
            json_response(self, 404, {"ok": False, "error": f"Unknown route: {parsed.path}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook
            if self._reject_unauthorized():
                return
            parsed = urlparse(self.path)
            if parsed.path == "/stop":
                payload = state.stop_current(reason="HTTP /stop requested")
                json_response(self, 200, {"ok": True, **payload})
                return
            if parsed.path == "/command":
                try:
                    payload = self._read_payload()
                except Exception as exc:
                    json_response(self, 400, {"ok": False, "error": f"Invalid request body: {exc}"})
                    return
                command = str(payload.get("command", "stop"))
                code, response = state.submit(command)
                json_response(self, code, response)
                return
            json_response(self, 404, {"ok": False, "error": f"Unknown route: {parsed.path}"})

        def log_message(self, fmt: str, *args: object) -> None:
            if not state.args.quiet:
                super().log_message(fmt, *args)

    return CommandHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mirror Jetson text commands into the Isaac Lab OpenArm demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Use 10.10.10.1 for Jetson direct LAN.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default="", help="Optional token required in X-Bridge-Token or Bearer auth.")
    parser.add_argument("--isaaclab-root", type=Path, default=DEFAULT_ISAACLAB_ROOT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/command_bridge"))
    parser.add_argument("--dry-run", action="store_true", help="Do not launch Isaac Lab; only create bridge jobs/logs.")
    parser.add_argument("--dry-run-delay", type=float, default=0.1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=700, help="Max Isaac steps for pick/grasp commands.")
    parser.add_argument("--hold-steps", type=int, default=120, help="Max Isaac steps for stop/open/close mirror commands.")
    parser.add_argument("--openarm-setup", choices=["bimanual", "unimanual"], default="bimanual")
    parser.add_argument("--control-arm", choices=["right", "left"], default="right")
    parser.add_argument("--object-x", type=float, default=0.29)
    parser.add_argument("--object-y", default="auto")
    parser.add_argument("--object-xy-noise", type=float, default=0.015)
    parser.add_argument("--pedestal-height", type=float, default=0.13)
    parser.add_argument("--pedestal-size-xy", type=float, default=0.24)
    parser.add_argument("--position-threshold", type=float, default=0.03)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    state = BridgeState(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Isaac Lab command bridge listening on http://{args.host}:{args.port}", flush=True)
    print(f"Runner: {args.runner}", flush=True)
    print("Physical robot is not commanded by this bridge.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        state.stop_current(reason="server shutdown")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
