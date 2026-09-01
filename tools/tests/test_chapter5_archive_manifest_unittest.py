import unittest


class Chapter5ArchiveManifestTests(unittest.TestCase):
    def test_manifest_has_fourteen_tasks_and_keeps_network_out_of_vla_rows(self):
        from tools.chapter5_archive_manifest import build_manifest

        manifest = build_manifest([])
        self.assertEqual(len(manifest["tasks"]), 14)
        self.assertEqual(len(manifest["objects"]), 112)
        self.assertTrue(all(
            row["archive_kind"] != "vla_training" or row["source_evidence"]
            for row in manifest["objects"]
        ))
        self.assertEqual(sum(row["archive_kind"] == "network_interface" for row in manifest["objects"]), 24)
