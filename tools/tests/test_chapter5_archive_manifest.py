from tools.chapter5_archive_manifest import build_manifest


def test_manifest_has_fourteen_tasks_and_keeps_network_out_of_vla_rows():
    manifest = build_manifest([])
    assert len(manifest["tasks"]) == 14
    assert len(manifest["objects"]) == 112
    assert all(
        row["archive_kind"] != "vla_training" or row["source_evidence"]
        for row in manifest["objects"]
    )
    assert sum(row["archive_kind"] == "network_interface" for row in manifest["objects"]) == 24
