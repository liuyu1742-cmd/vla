from pathlib import Path

from tools.openvla_4l_robocasa_eval import parse_seeds
from tools.openvla_4l_tcp_server import processor_root


def test_lora_uses_shared_base_processor():
    assert processor_root(
        "lora_r32", Path("models/adapter"), Path("models/base")
    ) == Path("models/base")


def test_full_checkpoint_uses_its_own_processor():
    assert processor_root(
        "full", Path("models/full"), Path("models/base")
    ) == Path("models/full")


def test_parse_seeds_is_exact_and_nonempty():
    assert parse_seeds("0,1,2,3,5") == [0, 1, 2, 3, 5]
    try:
        parse_seeds("")
    except ValueError as error:
        assert "at least one seed" in str(error)
    else:
        raise AssertionError("empty seeds must fail")

