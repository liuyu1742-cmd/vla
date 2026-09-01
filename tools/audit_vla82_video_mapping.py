"""Audit VLA82 operation text against the configured RoboCasa task topology."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "configs" / "vla82_full_simulation" / "operation_specs.json"
MAPPINGS = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"
OUTPUT = ROOT / "outputs" / "midterm_testing_vla82" / "vla82_video_mapping_audit.json"

TASK_TO_ROUTE = {
    "PickPlaceCabinetToCounter": ("cabinet", "counter"),
    "PickPlaceCounterToCabinet": ("counter", "cabinet"),
    "PickPlaceCounterToDrawer": ("counter", "drawer"),
    "PickPlaceDrawerToCounter": ("drawer", "counter"),
    "PickPlaceCounterToSink": ("counter", "sink"),
    "PickPlaceSinkToCounter": ("sink", "counter"),
}


def route_from_operation(text: str) -> tuple[str, str] | None:
    aliases = {"台面": "counter", "橱柜": "cabinet", "柜内": "cabinet", "抽屉": "drawer", "水槽": "sink"}
    if "→" in text:
        left, right = text.split("→", 1)
        source = next((value for key, value in aliases.items() if key in left), None)
        target = next((value for key, value in aliases.items() if key in right), None)
        return (source, target) if source and target else None
    if "放入水槽" in text:
        return ("counter", "sink")
    return None


def build_audit() -> dict[str, object]:
    specs_payload = json.loads(SPECS.read_text(encoding="utf-8"))
    specs = specs_payload.get("operations", []) if isinstance(specs_payload, dict) else specs_payload
    mappings = {row["selection_id"]: row for row in json.loads(MAPPINGS.read_text(encoding="utf-8"))["mappings"]}
    rows = []
    for spec in specs:
        configured = TASK_TO_ROUTE.get(mappings[spec["selection_id"]]["task_class"])
        video_route = route_from_operation(spec["operation_text"])
        status = "MATCH" if video_route and configured == video_route else "REVIEW" if video_route else "UNEXPRESSIBLE"
        rows.append({
            "selection_id": spec["selection_id"], "object_name": spec["object_name"],
            "operation_text": spec["operation_text"], "task_class": mappings[spec["selection_id"]]["task_class"],
            "video_route": video_route, "configured_route": configured, "status": status,
            "source_differences": spec.get("source_differences", []),
        })
    return {"schema_version": "vla82-video-mapping-audit-v1", "rows": rows}


def main() -> int:
    payload = build_audit()
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {key: sum(row["status"] == key for row in payload["rows"]) for key in ("MATCH", "REVIEW", "UNEXPRESSIBLE")}
    print(json.dumps({"output": str(OUTPUT.resolve()), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
