import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.vla82_pure_closed_loop import (
    REPRESENTATIVE_IDS,
    adapt_policy_action,
    build_manifest,
    filter_selected_representatives,
    select_representatives,
    task_class_module_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs" / "midterm_testing_vla82" / "simulator_mapping_plan.json"


class VLA82PureClosedLoopTests(unittest.TestCase):
    def test_single_representative_can_be_selected_for_repair_rerun(self):
        mappings = json.loads(PLAN.read_text(encoding="utf-8"))["mappings"]
        selected = select_representatives(mappings)

        repaired = filter_selected_representatives(selected, ["VLA82-030"])

        self.assertEqual([item["selection_id"] for item in repaired], ["VLA82-030"])

    def test_close_drawer_resolver_includes_atomic_drawer_module(self):
        self.assertIn(
            "robocasa.environments.kitchen.atomic.kitchen_drawer",
            task_class_module_candidates("CloseDrawer"),
        )

    def test_policy_action_is_converted_from_metric_delta_to_normalized_control(self):
        converted = adapt_policy_action(
            [0.01, -0.02, 0.10, 0.25, -0.50, 1.0, -1.0],
            translation_gain=20.0,
            rotation_gain=2.0,
        )

        self.assertEqual(converted, [0.2, -0.4, 1.0, 0.5, -1.0, 1.0, -1.0])

    def test_openvla_server_script_can_be_invoked_directly(self):
        completed = subprocess.run(
            [sys.executable, "tools/openvla_tcp_server.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--model-dir", completed.stdout)

    def test_representatives_cover_all_eight_categories_once(self):
        mappings = json.loads(PLAN.read_text(encoding="utf-8"))["mappings"]
        selected = select_representatives(mappings)

        self.assertEqual([item["selection_id"] for item in selected], REPRESENTATIVE_IDS)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({item["source_table"] for item in selected}), 8)
        self.assertTrue(all(item["mapping_mode"] != "functional_proxy" for item in selected[:-2]))

    def test_manifest_records_actual_pure_closed_loop_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            video = evidence / "rollout.mp4"
            first = evidence / "first.png"
            last = evidence / "last.png"
            for path in (video, first, last):
                path.write_bytes(b"evidence")
            result = {
                "selection_id": "VLA82-002",
                "source_table": "5-8-1",
                "success": False,
                "actual_closed_loop": True,
                "pure_autonomous_vla": True,
                "simulator_success_predicate_checked": True,
                "executed_decision_steps": 4,
                "executed_sim_steps": 8,
                "video": str(video),
                "first_frame": str(first),
                "last_frame": str(last),
            }

            manifest = build_manifest([result], expected_count=1)

        self.assertEqual(manifest["completed_count"], 1)
        self.assertEqual(manifest["success_count"], 0)
        self.assertTrue(manifest["evidence_complete"])
        self.assertEqual(manifest["evaluation_kind"], "pure_openvla_closed_loop")


if __name__ == "__main__":
    unittest.main()
