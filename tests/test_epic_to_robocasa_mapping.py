import unittest


class EpicToRoboCasaMappingTests(unittest.TestCase):
    def test_maps_cup_pick_place_to_existing_robocasa_task(self):
        from tools.epic_to_robocasa_mapping import map_pick_place

        mapping = map_pick_place("cup")

        self.assertEqual(mapping["task"], "PickPlaceCounterToCabinet")
        self.assertEqual(mapping["object_group"], "glass_cup")

    def test_rejects_unknown_object(self):
        from tools.epic_to_robocasa_mapping import map_pick_place

        with self.assertRaisesRegex(ValueError, "unsupported"):
            map_pick_place("unicorn")
