"""Hybrid-v2 plus auditable state-aware recovery for locate stagnation."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from tools.formal_skill_final_transport import activate_final_transport


_SOURCE = Path(__file__).resolve().parent / "formal_skill_hybrid_rollout.py"
PATCHED_SOURCE = _SOURCE.read_text(encoding="utf-8")
_REPLACEMENTS = (
    (
        "from tools.formal_skill_final_placement import activate_final_placement\n",
        "from tools.formal_skill_final_placement import activate_final_placement\n"
        "from tools.formal_skill_locate_recovery import LocateStagnationRecovery\n",
    ),
    (
        "    final_inside = False\n"
        "    executed_decisions = 0\n",
        "    final_inside = False\n"
        "    locate_recovery = LocateStagnationRecovery(\n"
        "        window_decisions=20, minimum_progress=0.01, model_weight=0.15\n"
        "    )\n"
        "    locate_recovery_started_at = None\n"
        "    executed_decisions = 0\n",
    ),
    (
        "        front, center, retreat = _safe_cabinet_waypoints(raw)\n\n"
        "        def state()",
        "        front, center, retreat = _safe_cabinet_waypoints(raw)\n"
        "        recovery_oracle = SafeCabinetPickPlaceOracle(\n"
        "            front,\n"
        "            center,\n"
        "            retreat,\n"
        "            world_to_origin=controller.world_to_origin_frame,\n"
        "        )\n\n"
        "        def state()",
    ),
    (
        "            if final_decision is None:\n"
        "                action = execution_guard.apply(raw_action, phase)\n"
        "                execution_mode = (\n"
        '                    "openvla_guarded"\n'
        "                    if np.allclose(action, guarded_action(raw_action, phase))\n"
        '                    else "calibrated_pick_timing"\n'
        "                )\n"
        "                controller_phase = phase\n"
        "            else:\n",
        "            recovery_active = (\n"
        "                locate_recovery.observe(phase, distance_before)\n"
        "                if final_decision is None\n"
        "                else False\n"
        "            )\n"
        "            recovery_expert = (\n"
        "                recovery_oracle.decide(snapshot) if recovery_active else None\n"
        "            )\n"
        "            if recovery_active and locate_recovery_started_at is None:\n"
        "                locate_recovery_started_at = decision_step\n"
        "            if final_decision is None:\n"
        "                if recovery_expert is not None:\n"
        "                    recovery_decision = locate_recovery.select_action(\n"
        "                        raw_action,\n"
        "                        recovery_expert.action,\n"
        "                        force_expert=recovery_expert.force_expert,\n"
        "                    )\n"
        "                    action = recovery_decision.action.copy()\n"
        "                    execution_mode = (\n"
        '                        "state_aware_locate_recovery_"\n'
        "                        + recovery_decision.mode\n"
        "                    )\n"
        "                    controller_phase = recovery_expert.phase\n"
        "                else:\n"
        "                    action = execution_guard.apply(raw_action, phase)\n"
        "                    execution_mode = (\n"
        '                        "openvla_guarded"\n'
        "                        if np.allclose(\n"
        "                            action, guarded_action(raw_action, phase)\n"
        "                        )\n"
        '                        else "calibrated_pick_timing"\n'
        "                    )\n"
        "                    controller_phase = phase\n"
        "            else:\n",
    ),
    (
        '        "final_servo_started_at": final_servo_started_at,\n'
        '        "phase_counts": dict(phase_counts),\n',
        '        "final_servo_started_at": final_servo_started_at,\n'
        '        "locate_recovery_started_at": locate_recovery_started_at,\n'
        '        "phase_counts": dict(phase_counts),\n',
    ),
)

PATCH_COUNT = 0
for old, new in _REPLACEMENTS:
    count = PATCHED_SOURCE.count(old)
    if count != 1:
        raise ImportError(
            f"formal hybrid-v3 source substitution drifted: expected 1, got {count}"
        )
    PATCHED_SOURCE = PATCHED_SOURCE.replace(old, new)
    PATCH_COUNT += 1

_MODULE_NAME = "tools._formal_skill_hybrid_rollout_v3"
_IMPLEMENTATION = types.ModuleType(_MODULE_NAME)
_IMPLEMENTATION.__file__ = str(_SOURCE)
_IMPLEMENTATION.__package__ = "tools"
sys.modules.setdefault(_MODULE_NAME, _IMPLEMENTATION)
exec(compile(PATCHED_SOURCE, str(_SOURCE), "exec"), _IMPLEMENTATION.__dict__)
_IMPLEMENTATION.activate_final_placement = activate_final_transport
main = _IMPLEMENTATION.main


if __name__ == "__main__":
    raise SystemExit(main())
