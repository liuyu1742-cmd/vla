"""Warning-free command-line entry point for generating a formal Skill IR."""

from __future__ import annotations

from .skill_ir import main


if __name__ == "__main__":
    raise SystemExit(main())
