"""Hybrid rollout with the full verified front-align-to-cabinet final path."""

from __future__ import annotations

import tools.formal_skill_hybrid_rollout as _hybrid
from tools.formal_skill_final_transport import activate_final_transport


_hybrid.activate_final_placement = activate_final_transport
main = _hybrid.main


if __name__ == "__main__":
    raise SystemExit(main())
