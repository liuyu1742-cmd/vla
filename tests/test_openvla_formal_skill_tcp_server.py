from __future__ import annotations

import unittest

from tools.openvla_formal_skill_tcp_server import conditioned_instruction


class FormalSkillServerTests(unittest.TestCase):
    def test_request_is_conditioned_on_canonical_phase(self) -> None:
        text = conditioned_instruction(
            {
                "instruction": "pick up the toy and place it in the cabinet",
                "canonical_phase": "place",
            }
        )
        self.assertEqual(
            text,
            "pick up the toy and place it in the cabinet; the current skill phase is place",
        )

    def test_missing_phase_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical_phase"):
            conditioned_instruction({"instruction": "organize the toy"})

    def test_unknown_phase_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            conditioned_instruction(
                {"instruction": "organize the toy", "canonical_phase": "dance"}
            )


if __name__ == "__main__":
    unittest.main()
