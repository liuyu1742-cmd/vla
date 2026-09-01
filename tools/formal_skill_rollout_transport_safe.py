"""Formal rollout with calibrated contact, lift, transport, and release timing."""

from __future__ import annotations

import tools.formal_skill_rollout as _rollout
from tools.formal_skill_calibrated_phase import next_calibrated_phase
from tools.formal_skill_transport_guard import CalibratedPickPlaceGuard


_transport_guard = CalibratedPickPlaceGuard(
    grasp_hold_decisions=18,
    lift_decisions=27,
    release_hold_decisions=18,
    base_guard=_rollout._implementation.guarded_action,
)
_rollout._implementation.next_canonical_phase = next_calibrated_phase
_rollout._implementation.guarded_action = _transport_guard.apply
main = _rollout.main


if __name__ == "__main__":
    raise SystemExit(main())
