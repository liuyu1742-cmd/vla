"""Formal rollout using only parameters measured from successful demonstrations."""

from __future__ import annotations

import tools.formal_skill_rollout as _rollout
from tools.formal_skill_transport_guard import CalibratedPickPlaceGuard
from tools.formal_skill_verified_phase import VerifiedPhaseController


_phase_controller = VerifiedPhaseController(max_grasp_decisions=20)
_transport_guard = CalibratedPickPlaceGuard(
    grasp_hold_decisions=18,
    lift_decisions=27,
    release_hold_decisions=18,
    base_guard=_rollout._implementation.guarded_action,
)
_rollout._implementation.next_canonical_phase = _phase_controller
_rollout._implementation.guarded_action = _transport_guard.apply
main = _rollout.main


if __name__ == "__main__":
    raise SystemExit(main())
