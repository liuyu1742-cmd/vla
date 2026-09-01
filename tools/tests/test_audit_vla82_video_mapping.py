from tools.audit_vla82_video_mapping import build_audit, route_from_operation


def test_operation_route_parser_handles_arrow_and_sink_insert() -> None:
    assert route_from_operation("橱柜→台面放置") == ("cabinet", "counter")
    assert route_from_operation("抓取—放入水槽") == ("counter", "sink")


def test_pan_mapping_matches_real_video_operation() -> None:
    rows = {row["selection_id"]: row for row in build_audit()["rows"]}
    assert rows["VLA82-015"]["status"] == "MATCH"
    assert rows["VLA82-015"]["configured_route"] == ("counter", "sink")


def test_expressible_real_video_routes_match_runtime_tasks() -> None:
    rows = {row["selection_id"]: row for row in build_audit()["rows"]}
    for selection_id in ("VLA82-019", "VLA82-020", "VLA82-043", "VLA82-044"):
        assert rows[selection_id]["status"] == "MATCH"
