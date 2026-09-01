"""Formal rollout with measured grasp-close and release-open timing locks."""

from __future__ import annotations

import tools.formal_skill_rollout as _rollout
from tools.formal_skill_execution_guard import PickPlaceExecutionGuard


_execution_guard = PickPlaceExecutionGuard(
    grasp_hold_decisions=18,
    release_hold_decisions=18,
    base_guard=_rollout._implementation.guarded_action,
)
_rollout._implementation.guarded_action = _execution_guard.apply
main = _rollout.main


if __name__ == "__main__":
    raise SystemExit(main())
