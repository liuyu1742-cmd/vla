import unittest


class StrictRegistryTests(unittest.TestCase):
    def test_vla_entry_requires_instruction_rgb_and_actions(self):
        from tools.build_strict_vla_120_registry import validate_vla_entry

        valid = {
            "instruction": "Pick up the plate and place it in the cabinet.",
            "rgb_files": ["rgb_robot0_agentview_left.mp4"],
            "action_file": "actions.parquet",
            "source_episode": 42,
        }
        self.assertEqual(validate_vla_entry(valid), [])

        invalid = dict(valid, rgb_files=[])
        self.assertEqual(validate_vla_entry(invalid), ["missing_rgb"])

    def test_network_entry_is_explicitly_exempt_from_vla_pairing(self):
        from tools.build_strict_vla_120_registry import validate_entry

        network = {"source_type": "network_control", "object": "智能插座"}
        self.assertEqual(validate_entry(network), [])


if __name__ == "__main__":
    unittest.main()
