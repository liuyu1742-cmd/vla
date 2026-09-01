"""Focused tests for the presentation-only replay camera helpers."""

from __future__ import annotations

import unittest

import numpy as np

from tools.replay_vla82_strict_pass_video import (
    camera_position_in_parent_frame,
    presentation_occluder_names,
    presentation_world_camera_position,
)


class ReplayCameraTest(unittest.TestCase):
    def test_world_locked_camera_position_tracks_a_moving_parent(self) -> None:
        """A replay camera remains fixed in world space as the base moves."""
        local = camera_position_in_parent_frame(
            world_position=np.array((3.0, 5.0, 7.0)),
            parent_position=np.array((1.0, 2.0, 3.0)),
            parent_rotation=np.eye(3),
        )
        np.testing.assert_allclose(local, np.array((2.0, 3.0, 4.0)))

    def test_vla004_masks_only_its_open_cabinet_door_in_presentation(self) -> None:
        """Only the visual occluders for the VLA82-004 replay are selected."""
        self.assertEqual(
            presentation_occluder_names("VLA82-004"),
            (
                "cab_1_right_group_left_door_g0",
                "cab_1_right_group_left_door_g1",
                "cab_1_right_group_right_door_g0",
                "cab_1_right_group_right_door_g1",
            ),
        )
        self.assertEqual(presentation_occluder_names("VLA82-017"), ())

    def test_vla018_uses_a_front_world_camera_and_masks_open_door_panels(self) -> None:
        self.assertEqual(
            presentation_occluder_names("VLA82-018"),
            (
                "cab_2_left_group_left_door_g0",
                "cab_2_left_group_left_door_g1",
                "cab_2_left_group_right_door_g0",
                "cab_2_left_group_right_door_g1",
            ),
        )
        np.testing.assert_allclose(
            presentation_world_camera_position(
                "VLA82-018", default=(0.576, -4.848, 1.55),
            ),
            (1.40, -3.20, 2.20),
        )


if __name__ == "__main__":
    unittest.main()
