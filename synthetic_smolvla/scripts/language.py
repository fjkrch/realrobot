#!/usr/bin/env python3
"""Language command parsing for the four-object synthetic pick task."""

from __future__ import annotations


INSTRUCTION_BY_OBJECT = {
    "orange_ball": "pick up the orange ball",
    "red_cube": "pick up the red cube",
    "green_cube": "pick up the green cube",
    "blue_cube": "pick up the blue cube",
}

OBJECT_ALIASES = {
    "orange_ball": ("orange ball", "orange ping-pong ball", "orange ping pong ball", "orange", "ball"),
    "red_cube": ("red cube", "red"),
    "green_cube": ("green cube", "green"),
    "blue_cube": ("blue cube", "blue"),
}


class LanguageError(ValueError):
    """Raised when an instruction cannot be mapped to exactly one object."""


def instruction_for_object(object_name: str) -> str:
    try:
        return INSTRUCTION_BY_OBJECT[object_name]
    except KeyError as exc:
        raise LanguageError(f"Unknown object {object_name!r}.") from exc


def parse_target_object(instruction: str) -> str:
    text = " ".join(str(instruction).lower().replace("-", " ").split())
    matches = [
        object_name
        for object_name, aliases in OBJECT_ALIASES.items()
        if any(alias in text for alias in aliases)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise LanguageError(
            "Instruction does not name one of: orange ball, red cube, green cube, blue cube."
        )
    raise LanguageError(f"Instruction is ambiguous across objects: {matches}.")


def all_instructions() -> tuple[str, ...]:
    return tuple(INSTRUCTION_BY_OBJECT[name] for name in sorted(INSTRUCTION_BY_OBJECT))
