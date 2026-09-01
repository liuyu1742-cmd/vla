from __future__ import annotations

import json
import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.formal_skill_dagger import (
    choose_executed_action,
    save_dagger_episode,
    validate_training_seed,
)


@dataclass(frozen=True)
class Decision:
    action: np.ndarray
    phase: str
    canonical_action: str
    force_expert: bool = False


class FormalSkillDaggerTests(unittest.TestCase):
    def test_heldout_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_training_seed(101, {101, 102, 103})

    def test_premature_policy_close_is_fully_opened(self) -> None:
        policy = np.array([0.2, 0.1, -0.5, 0.0, 0.0, 0.0, 1.0])
        oracle = Decision(
            np.array([0.3, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]),
            "approach_object",
            "locate",
        )

        mixed = choose_executed_action(
            policy, oracle, beta=0.0, rng=random.Random(7)
        )

        self.assertEqual("policy_gated", mixed.source)
        self.assertTrue(mixed.gripper_gated)
        self.assertEqual(-1.0, float(mixed.executed_action[6]))
        np.testing.assert_allclose(mixed.oracle_label, oracle.action)

    def test_forced_oracle_phase_replaces_complete_policy_action(self) -> None:
        policy = np.ones(7)
        oracle = Decision(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            "release_object",
            "place",
            force_expert=True,
        )

        mixed = choose_executed_action(
            policy, oracle, beta=0.0, rng=random.Random(7)
        )

        self.assertEqual("oracle_forced", mixed.source)
        np.testing.assert_allclose(mixed.executed_action, oracle.action)

    def test_saved_training_actions_are_oracle_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
            oracle = np.array(
                [
                    [1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            policy = np.zeros((2, 7), dtype=np.float32)
            executed = policy.copy()
            report = {
                "relation_key": "organizing::toy",
                "instruction": "整理并归位玩具",
                "round": 2,
                "beta": 0.5,
                "success": False,
            }

            episode_path, report_path, sidecar_path = save_dagger_episode(
                root,
                seed=42,
                heldout_seeds={101, 102, 103},
                frames=frames,
                oracle_actions=oracle,
                policy_actions=policy,
                executed_actions=executed,
                phases=["approach_object", "close_gripper"],
                canonical_phases=["locate", "grasp"],
                eef_positions=np.zeros((2, 3)),
                object_positions=np.ones((2, 3)),
                grasped=[False, True],
                report=report,
            )

            with np.load(episode_path, allow_pickle=False) as episode:
                np.testing.assert_allclose(episode["actions"], oracle)
                np.testing.assert_allclose(episode["oracle_actions"], oracle)
                np.testing.assert_allclose(episode["policy_actions"], policy)
                np.testing.assert_allclose(episode["executed_actions"], executed)
                self.assertEqual("organizing::toy", str(episode["relation_key"].item()))
                self.assertEqual((2,), episode["canonical_phases"].shape)
            self.assertTrue(report_path.is_file())
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual("dagger_recovery", sidecar["source"])
            self.assertEqual(2, sidecar["samples"])
            self.assertEqual(str(episode_path.resolve()), sidecar["episode"])


if __name__ == "__main__":
    unittest.main()
