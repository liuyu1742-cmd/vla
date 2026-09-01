from __future__ import annotations

from pathlib import Path

from tools.run_vla82_full_simulation import audit_counts, audit_is_complete, reusable_audit_pass


def test_partial_audit_is_not_complete_but_preserves_real_counts() -> None:
    results = [
        {
            "exact": True,
            "loadable": True,
            "visible": True,
            "complete_robot": True,
            "proxy": False,
        }
    ]

    assert audit_counts(results) == {
        "exact": 1,
        "loadable": 1,
            "visible": 1,
            "complete_robot": 1,
            "runtime_identity": 0,
        "proxies": 0,
    }
    assert audit_is_complete(results) is False


def test_resume_never_reuses_fail_and_requires_complete_pass_evidence(tmp_path: Path) -> None:
    preview = tmp_path / "preview.png"
    evidence = tmp_path / "asset_evidence.json"
    preview.write_bytes(b"png")
    evidence.write_text("{}", encoding="utf-8")
    passed = {
        "status": "PASS", "exact": True, "loadable": True, "visible": True,
        "complete_robot": True, "proxy": False, "preview": str(preview),
        "asset_evidence": str(evidence),
        "audit_camera": {"camera_name": "robot0_agentview_left", "segmentation_pixels": 4},
        "visibility_pixels": 4,
        "runtime_asset_identity": {"selection_id": "VLA82-002", "asset_key": "VLA82-002", "asset_kind": "custom_same_class", "matches": True, "loaded_mjcf_path": "C:/assets/VLA82-002/model.xml", "expected_mjcf_path": "C:/assets/VLA82-002/model.xml"},
    }

    assert reusable_audit_pass(passed, render_requested=True) is True
    assert reusable_audit_pass({**passed, "status": "FAIL"}, render_requested=True) is False
    assert reusable_audit_pass({**passed, "asset_evidence": ""}, render_requested=True) is False
    assert reusable_audit_pass({key: value for key, value in passed.items() if key != "runtime_asset_identity"}, render_requested=True) is False


def test_fixture_resume_requires_source_frame_hash_and_operable_joint(tmp_path: Path) -> None:
    frame = tmp_path / "fixture_source.png"
    frame.write_bytes(b"fixture-source-frame")
    import hashlib
    fixture_evidence = tmp_path / "fixture_evidence.json"
    fixture_evidence.write_text(
        __import__("json").dumps(
            {
                "fixture_binding": {"geom": "door_g0", "joint": "door_joint", "joint_required": "true"},
                "joint_verified": True,
                "source_frame": str(frame),
                "source_frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    item = {
        "status": "PASS", "exact": True, "loadable": True, "visible": True,
        "complete_robot": True, "proxy": False, "asset_evidence": str(fixture_evidence),
        "preview": str(frame), "audit_camera": {"segmentation_pixels": 4}, "visibility_pixels": 4,
        "runtime_asset_identity": {"selection_id": "VLA82-007", "asset_key": "VLA82-007", "asset_kind": "fixture_part", "matches": True, "loaded_mjcf_path": "", "expected_mjcf_path": ""},
    }

    assert reusable_audit_pass(item, render_requested=True) is True
    fixture_evidence.write_text('{"fixture_binding": {"geom": "door_g0", "joint": ""}}', encoding="utf-8")
    assert reusable_audit_pass(item, render_requested=True) is False


def test_audit_complete_rejects_duplicate_or_wrong_fixed_ids() -> None:
    record = {"selection_id": "VLA82-001", "exact": True, "loadable": True, "visible": True, "complete_robot": True, "proxy": False}
    assert audit_is_complete([record] * 60) is False
