import unittest

from tools.formal_skill_policy_prompt import build_policy_instruction


class FormalSkillPolicyPromptTests(unittest.TestCase):
    def test_preserves_clean_english_task_instruction(self):
        self.assertEqual(
            build_policy_instruction(
                "pick up the toy and place it in the cabinet", "place"
            ),
            "pick up the toy and place it in the cabinet; "
            "the current skill phase is place",
        )

    def test_replaces_mojibake_with_canonical_openvla_instruction(self):
        prompt = build_policy_instruction(
            "鏁寸悊浠诲姟锛氭暣鐞嗗苟褰掍綅鐜╁叿锛坱oy锛夈€?", "locate"
        )

        self.assertEqual(
            prompt,
            "put away the toy in the cabinet; the current skill phase is locate",
        )
        self.assertTrue(prompt.isascii())

    def test_replaces_chinese_contract_text_without_changing_phase(self):
        self.assertEqual(
            build_policy_instruction("整理并归位玩具", "grasp"),
            "put away the toy in the cabinet; the current skill phase is grasp",
        )

    def test_rejects_missing_or_invalid_phase(self):
        with self.assertRaises(ValueError):
            build_policy_instruction("", "locate")
        with self.assertRaises(ValueError):
            build_policy_instruction("organize toy", "")


if __name__ == "__main__":
    unittest.main()
