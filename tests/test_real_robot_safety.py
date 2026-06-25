"""Static safety checks for scripts that can move the real OpenArm.

These tests intentionally avoid importing the robot scripts. Importing them may
require hardware-only dependencies, and the safety property we care about here
is visible in source: real-motion paths must require an explicit
``--i-am-at-robot`` acknowledgement before touching CAN.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

REAL_MOTION_ENTRYPOINTS = [
    ("scripts/move_joint.py", "main"),
    ("scripts/move_arm.py", "main"),
    ("scripts/pick_cube.py", "run_real"),
    ("scripts/replay_openarm_saved_episode_real.py", "main"),
]

REAL_CONFIRM_ENTRYPOINTS = [
    ("scripts/openarm_safe_real_mirror.py", "main"),
]


def _source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name!r}")


def _first_call_line(node: ast.AST, function_name: str) -> int | None:
    lines = [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == function_name
    ]
    return min(lines) if lines else None


def _first_i_am_at_robot_guard_line(node: ast.AST) -> int | None:
    lines = []
    for child in ast.walk(node):
        if not isinstance(child, ast.If):
            continue
        test = ast.unparse(child.test)
        if "args.i_am_at_robot" in test and ("not " in test or " is False" in test):
            lines.append(child.lineno)
    return min(lines) if lines else None


def _first_real_confirm_guard_line(node: ast.AST) -> int | None:
    lines = []
    for child in ast.walk(node):
        if not isinstance(child, ast.If):
            continue
        test = ast.unparse(child.test)
        if "args.real_confirm" in test and "REQUIRED_REAL_CONFIRMATION" in test:
            lines.append(child.lineno)
    return min(lines) if lines else None


def _first_confirm_real_hardware_guard_line(node: ast.AST) -> int | None:
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.If):
            test = ast.unparse(child.test)
            if "args.confirm_real_hardware" in test and ("not " in test or " is False" in test):
                lines.append(child.lineno)
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "validate_real_flags"
        ):
            lines.append(child.lineno)
    return min(lines) if lines else None


def test_real_motion_scripts_require_operator_acknowledgement_flag() -> None:
    for relative_path, _entrypoint in REAL_MOTION_ENTRYPOINTS + REAL_CONFIRM_ENTRYPOINTS:
        text = _source(relative_path)

        assert (
            "--i-am-at-robot" in text
            or "--real-confirm" in text
            or "--confirm-real-hardware" in text
        ), relative_path
        assert (
            "args.i_am_at_robot" in text
            or "args.real_confirm" in text
            or "args.confirm_real_hardware" in text
        ), relative_path
        assert "Refusing" in text, relative_path


def test_real_motion_paths_check_acknowledgement_before_can_access() -> None:
    for relative_path, entrypoint in REAL_MOTION_ENTRYPOINTS:
        tree = ast.parse(_source(relative_path), filename=relative_path)
        function = _function(tree, entrypoint)

        guard_line = (
            _first_i_am_at_robot_guard_line(function)
            or _first_real_confirm_guard_line(function)
            or _first_confirm_real_hardware_guard_line(function)
        )
        can_line = _first_call_line(function, "require_can_interface")

        assert guard_line is not None, f"{relative_path}:{entrypoint} missing operator confirmation guard"
        assert can_line is not None, f"{relative_path}:{entrypoint} missing CAN interface check"
        assert guard_line < can_line, (
            f"{relative_path}:{entrypoint} must check operator confirmation before CAN access"
        )


def test_safe_real_mirror_checks_confirmation_before_session_connect() -> None:
    for relative_path, entrypoint in REAL_CONFIRM_ENTRYPOINTS:
        tree = ast.parse(_source(relative_path), filename=relative_path)
        function = _function(tree, entrypoint)

        guard_line = _first_real_confirm_guard_line(function)
        connect_lines = [
            child.lineno
            for child in ast.walk(function)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "connect"
        ]
        assert guard_line is not None, f"{relative_path}:{entrypoint} missing --real-confirm guard"
        assert connect_lines, f"{relative_path}:{entrypoint} missing session.connect call"
        assert guard_line < min(connect_lines), (
            f"{relative_path}:{entrypoint} must check --real-confirm before session.connect"
        )


def test_sim_command_client_stays_sim_only() -> None:
    text = _source("scripts/jetson_isaaclab_command.py")

    assert "OpenArmFollower" not in text
    assert "lerobot.robots" not in text
    assert "--i-am-at-robot" not in text


def test_interactive_vla_real_mirror_requires_confirmation_and_preflight() -> None:
    text = _source("synthetic_smolvla/scripts/interactive_vla_isaac.py")

    assert "--mirror-real" in text
    assert "--real-confirm" in text
    assert "--prepare-real-start-pose" in text
    assert "REQUIRED_REAL_CONFIRMATION" in text
    assert "--mirror-real requires --prepare-real-start-pose" in text
