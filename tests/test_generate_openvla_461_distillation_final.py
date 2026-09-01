from tools.generate_openvla_461_distillation_final import format_rate, method_rows


def test_format_rate_includes_fraction_and_percent():
    assert format_rate(5, 5) == "5/5 (100.0%)"


def test_method_rows_keep_the_four_requested_methods():
    rows = method_rows()
    assert [row["method"] for row in rows] == [
        "Full Fine-tuning",
        "LoRA (rank=32)",
        "Last-Layer-Only",
        "Frozen-Vision",
    ]
