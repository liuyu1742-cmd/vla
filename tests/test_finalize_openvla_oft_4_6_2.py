from tools.finalize_openvla_oft_4_6_2 import mean_one_decimal_half_up


def test_reference_average_uses_conventional_half_up_rounding():
    assert mean_one_decimal_half_up([76.5, 72.1, 69.4, 68.2]) == 71.6
