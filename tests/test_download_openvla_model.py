import unittest
from pathlib import Path


class DownloadOpenVlaModelTests(unittest.TestCase):
    def test_default_target_is_project_models_directory(self):
        from tools.download_openvla_model import default_target

        self.assertEqual(default_target().name, "openvla-7b")
        self.assertEqual(default_target().parent.name, "models")

    def test_command_metadata_records_mirror_and_model(self):
        from tools.download_openvla_model import build_download_metadata, model_files

        metadata = build_download_metadata(Path("C:/models/openvla-7b"), "https://hf-mirror.com")

        self.assertEqual(metadata["repo_id"], "openvla/openvla-7b")
        self.assertEqual(metadata["endpoint"], "https://hf-mirror.com")
        self.assertEqual(Path(metadata["target"]), Path("C:/models/openvla-7b"))
        self.assertIn("model-00001-of-00003.safetensors", model_files())
