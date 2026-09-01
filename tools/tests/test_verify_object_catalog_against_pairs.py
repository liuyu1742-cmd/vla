from tools.verify_object_catalog_against_pairs import verify_object_row


def test_verify_object_row_requires_exact_source_phrase_and_paired_task():
    paired = {0: {"instruction": "Turn on the radio receiver on the table.", "pair_count": 20}}
    assert verify_object_row({"source_task_index": 0, "source_phrase": "radio receiver"}, paired) is True
    assert verify_object_row({"source_task_index": 0, "source_phrase": "television"}, paired) is False
    assert verify_object_row({"source_task_index": 1, "source_phrase": "radio receiver"}, paired) is False
