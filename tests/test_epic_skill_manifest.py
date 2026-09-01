import unittest


class EpicSkillManifestTests(unittest.TestCase):
    def test_rejects_phase_without_target(self):
        from tools.epic_skill_manifest import validate_manifest

        with self.assertRaisesRegex(ValueError, "target"):
            validate_manifest({"clip_id": "P02_123_130", "phases": [{"verb": "pick", "object": "cup"}]})
