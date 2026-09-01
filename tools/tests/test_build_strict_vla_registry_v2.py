import unittest


class StrictRegistryV2Tests(unittest.TestCase):
    def test_legacy_network_interface_is_exempt(self):
        from tools.build_strict_vla_registry_v2 import validate_record

        record = {"archive_kind": "network_interface", "object_name": "温湿度传感器"}
        self.assertEqual(validate_record(record), [])


if __name__ == "__main__":
    unittest.main()
