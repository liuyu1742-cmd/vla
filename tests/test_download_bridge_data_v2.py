import unittest


class BridgeDataDownloadTests(unittest.TestCase):
    def test_official_split_contains_all_expected_tfrecord_names(self):
        from tools.download_bridge_data_v2 import filenames

        names = filenames()

        self.assertEqual(len(names), 1158)
        self.assertEqual(names[0], "bridge_dataset-train.tfrecord-00000-of-01024")
        self.assertIn("bridge_dataset-val.tfrecord-00127-of-00128", names)
        self.assertIn("dataset_info.json", names)
