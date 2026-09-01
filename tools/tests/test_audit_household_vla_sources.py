from tools.audit_household_vla_sources import classify_record


def test_classify_record_requires_rgb_text_and_actions():
    record = classify_record("x", "put cup away", True, True, "direct", "home")
    assert record["eligible"] is True
    assert (
        classify_record("x", "put cup away", True, False, "missing", "home")[
            "eligible"
        ]
        is False
    )
