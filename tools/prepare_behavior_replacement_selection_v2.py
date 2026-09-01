"""Create the 44-row replacement selection, excluding the extra apple row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.prepare_behavior_replacement_selection import SPECS, build_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = list(SPECS)
    try:
        SPECS[:] = [row for row in original if row[1] != "苹果"]
        rows = build_selection(args.manifest)
    finally:
        SPECS[:] = original
    if len(rows) != 44:
        raise ValueError(f"expected 44 replacement rows, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"objects": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
