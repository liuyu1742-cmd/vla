"""Regression tests for OpenVLA visual-token label alignment."""

from __future__ import annotations

import unittest

import torch

from tools.water_cup_multimodal_labels import expand_labels_for_visual_tokens


class WaterCupMultimodalLabelsTests(unittest.TestCase):
    def test_inserts_ignored_visual_labels_after_bos(self) -> None:
        labels = torch.tensor([[1, -100, 11, 12, 2]])
        expanded = expand_labels_for_visual_tokens(labels, logits_length=8)

        self.assertEqual(expanded.tolist(), [[1, -100, -100, -100, -100, 11, 12, 2]])
        self.assertEqual(expanded.shape[1], 8)


if __name__ == "__main__":
    unittest.main()
