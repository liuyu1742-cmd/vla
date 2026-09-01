"""Train the formal OpenVLA adapter with shuffled data and canonical policy text."""

from __future__ import annotations

import tools.finetune_formal_skill_openvla_resumable as _trainer
from tools.formal_skill_policy_prompt import (
    CANONICAL_POLICY_INSTRUCTION,
    build_policy_instruction,
)


_original_summary = _trainer._IMPLEMENTATION.planned_training_summary


def _planned_training_summary(*args, **kwargs):
    summary = _original_summary(*args, **kwargs)
    summary.update(
        {
            "policy_instruction": CANONICAL_POLICY_INSTRUCTION,
            "policy_prompt_language": "English",
            "contract_instruction_preserved_separately": True,
        }
    )
    return summary


_trainer._IMPLEMENTATION.build_instruction = build_policy_instruction
_trainer._IMPLEMENTATION.planned_training_summary = _planned_training_summary
main = _trainer.main


if __name__ == "__main__":
    raise SystemExit(main())
