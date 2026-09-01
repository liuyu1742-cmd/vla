import unittest


class OpenVlaSimulatorAdapterTests(unittest.TestCase):
    def test_seven_dimensional_action_is_preserved_inside_bounds(self):
        from tools.openvla_simulator_adapter import adapt_action

        action = adapt_action([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0], [-1.0] * 7, [1.0] * 7)

        self.assertEqual(action, [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0])

    def test_out_of_range_action_is_clipped_before_simulator_execution(self):
        from tools.openvla_simulator_adapter import adapt_action

        action = adapt_action([4.0, -2.0, 0.0, 0.0, 0.0, 0.0, 3.0], [-1.0] * 7, [1.0] * 7)

        self.assertEqual(action, [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_action_dimension_mismatch_is_rejected(self):
        from tools.openvla_simulator_adapter import adapt_action

        with self.assertRaisesRegex(ValueError, "dimension"):
            adapt_action([0.0] * 6, [-1.0] * 7, [1.0] * 7)

    def test_seven_dimensional_action_maps_to_stationary_robocasa_control_dict(self):
        from tools.openvla_simulator_adapter import to_robocasa_action

        mapped = to_robocasa_action([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        self.assertTrue((abs(mapped["action.end_effector_position"] - [0.1, 0.2, 0.3]) < 1e-6).all())
        self.assertTrue((abs(mapped["action.end_effector_rotation"] - [0.4, 0.5, 0.6]) < 1e-6).all())
        self.assertTrue((abs(mapped["action.gripper_close"] - [0.7]) < 1e-6).all())
        self.assertEqual(mapped["action.base_motion"].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(mapped["action.control_mode"].tolist(), [0.0])
