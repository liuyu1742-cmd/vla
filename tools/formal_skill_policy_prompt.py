"""Canonical policy-language prompts for the formal organizing::toy skill."""

from __future__ import annotations


CANONICAL_POLICY_INSTRUCTION = "put away the toy in the cabinet"


def build_policy_instruction(instruction: str, canonical_phase: str) -> str:
    task = " ".join(str(instruction).strip().split())
    phase = " ".join(str(canonical_phase).strip().lower().split())
    if not task or not phase:
        raise ValueError("instruction and canonical phase must be non-empty")
    policy_task = task if task.isascii() else CANONICAL_POLICY_INSTRUCTION
    return f"{policy_task}; the current skill phase is {phase}"


__all__ = ["CANONICAL_POLICY_INSTRUCTION", "build_policy_instruction"]
