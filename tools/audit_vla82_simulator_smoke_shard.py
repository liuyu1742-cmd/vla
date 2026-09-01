"""Run one independent index shard of the VLA82 simulator smoke audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "robosuite"))
    sys.path.insert(0, str(ROOT / "third_party" / "robocasa"))
    from tools import audit_vla82_simulator_smoke as implementation

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    mappings = plan["mappings"]
    if not 1 <= arguments.start_index <= arguments.end_index <= len(mappings):
        raise ValueError("invalid inclusive shard range")
    selected = mappings[arguments.start_index - 1 : arguments.end_index]
    arguments.output.mkdir(parents=True, exist_ok=True)
    shard_plan = {
        **plan,
        "source_scope": {
            **plan["source_scope"],
            "task_count": len({mapping["task"] for mapping in selected}),
            "object_count": len(selected),
        },
        "mappings": selected,
        "shard": {
            "start_index": arguments.start_index,
            "end_index": arguments.end_index,
            "expected_mapping_count": len(selected),
            "full_plan": str(arguments.plan.resolve()),
        },
    }
    shard_plan_path = arguments.output / "shard_plan.json"
    shard_plan_path.write_text(
        json.dumps(shard_plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = implementation.run_smoke(
        shard_plan_path,
        arguments.output,
        attempts=arguments.attempts,
        resume=True,
        limit=None,
    )
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
