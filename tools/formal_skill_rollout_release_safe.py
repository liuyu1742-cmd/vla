"""Formal rollout with a measured 36-simulator-step stationary release hold."""

from __future__ import annotations

import tools.formal_skill_rollout as _rollout
from tools.formal_skill_release_guard import ReleaseHoldGuard


_release_guard = ReleaseHoldGuard(
    hold_decisions=18,
    base_guard=_rollout._implementation.guarded_action,
)
_rollout._implementation.guarded_action = _release_guard.apply
main = _rollout.main


if __name__ == "__main__":
    raise SystemExit(main())
