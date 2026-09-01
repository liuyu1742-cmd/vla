from tools.build_midterm_layered_evidence import evaluate_layered_acceptance


def test_layered_midterm_acceptance_passes_without_claiming_60_closed_loops():
    result = evaluate_layered_acceptance(
        category_success=8,
        category_expected=8,
        interface_pass=60,
        interface_expected=60,
        inference_pass=600,
        inference_expected=600,
        pure_representative_success=True,
    )
    assert result["midterm_layered_acceptance_passed"] is True
    assert result["all_60_pure_vla_closed_loop_claimed"] is False


def test_layered_midterm_acceptance_fails_when_a_required_layer_is_incomplete():
    result = evaluate_layered_acceptance(
        category_success=7,
        category_expected=8,
        interface_pass=60,
        interface_expected=60,
        inference_pass=600,
        inference_expected=600,
        pure_representative_success=True,
    )
    assert result["midterm_layered_acceptance_passed"] is False
