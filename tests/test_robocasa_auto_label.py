import unittest

import numpy as np

from tools.robocasa_auto_label import (
    bbox_iou,
    bbox_from_mask,
    bbox_to_yolo,
    mask_from_geom_ids,
    split_for_index,
)


class RoboCasaAutoLabelTest(unittest.TestCase):
    def test_mask_and_tight_bbox_from_visual_geom_ids(self):
        segmentation = np.zeros((6, 8, 2), dtype=np.int32)
        segmentation[1:4, 2:6, 0] = 5
        segmentation[1:4, 2:6, 1] = 17

        mask = mask_from_geom_ids(segmentation, {17, 19})

        np.testing.assert_array_equal(mask, segmentation[..., 1] == 17)
        self.assertEqual(bbox_from_mask(mask, min_visible_pixels=4), (2, 1, 6, 4))

    def test_empty_mask_has_no_bbox(self):
        mask = np.zeros((4, 5), dtype=bool)
        self.assertIsNone(bbox_from_mask(mask, min_visible_pixels=1))

    def test_small_mask_is_rejected(self):
        mask = np.zeros((4, 5), dtype=bool)
        mask[1, 2] = True
        self.assertIsNone(bbox_from_mask(mask, min_visible_pixels=2))

    def test_bbox_converts_to_normalized_yolo_row(self):
        row = bbox_to_yolo(class_id=4, bbox=(2, 1, 6, 5), image_width=8, image_height=8)
        self.assertEqual(row, (4, 0.5, 0.375, 0.5, 0.5))

    def test_invalid_segmentation_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "two channels"):
            mask_from_geom_ids(np.zeros((4, 5), dtype=np.int32), {1})

    def test_seed_level_split_keeps_last_seed_for_validation(self):
        self.assertEqual([split_for_index(i, 5, 0.2) for i in range(5)], [
            "train", "train", "train", "train", "val"
        ])

    def test_bbox_iou_distinguishes_target_from_false_positive(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(bbox_iou((0, 0, 2, 2), (3, 3, 5, 5)), 0.0)


if __name__ == "__main__":
    unittest.main()
