#!/usr/bin/env python3
"""Send a Jetson text command to the laptop Isaac Lab mirror server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_SERVER = "http://10.10.10.1:8765"


def request_json(server: str, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None, token: str = ""):
    base = server.rstrip("/")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(f"{base}{path}", data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("X-Bridge-Token", token)
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"ok": False, "error": body}
        return exc.code, payload
    except URLError as exc:
        return 0, {"ok": False, "error": f"Could not reach bridge server: {exc}"}


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def maybe_wait(server: str, job_id: str, *, token: str, poll_seconds: float) -> int:
    while True:
        status_code, payload = request_json(server, f"/job/{job_id}", token=token)
        print_json(payload)
        job = payload.get("job", {})
        if status_code != 200 or job.get("status") in {"completed", "failed", "stopped"}:
            return 0 if job.get("status") == "completed" else 1
        time.sleep(poll_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send commands from Jetson to the laptop Isaac Lab mirror.")
    parser.add_argument("command", nargs="*", help='Command text, for example: "pick up the cube" or "open gripper".')
    parser.add_argument("--server", default=os.environ.get("ISAACLAB_MIRROR_SERVER", DEFAULT_SERVER))
    parser.add_argument("--token", default=os.environ.get("BRIDGE_TOKEN", ""))
    parser.add_argument("--status", action="store_true", help="Show mirror server status instead of sending a command.")
    parser.add_argument("--stop", action="store_true", help="Stop the current Isaac Lab mirror job.")
    parser.add_argument("--wait", action="store_true", help="Poll until the submitted mirror job exits.")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.status:
        status_code, payload = request_json(args.server, "/status", token=args.token)
        print_json(payload)
        return 0 if status_code == 200 else 1

    if args.stop:
        status_code, payload = request_json(args.server, "/stop", method="POST", token=args.token)
        print_json(payload)
        return 0 if status_code == 200 else 1

    command = " ".join(args.command).strip()
    if not command:
        print("Command is required unless --status or --stop is used.", file=sys.stderr)
        return 2

    status_code, payload = request_json(args.server, "/command", method="POST", payload={"command": command}, token=args.token)
    print_json(payload)
    job_id = payload.get("job", {}).get("id")
    if args.wait and status_code in {200, 202} and job_id:
        return maybe_wait(args.server, job_id, token=args.token, poll_seconds=args.poll_seconds)
    return 0 if status_code in {200, 202} and payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
