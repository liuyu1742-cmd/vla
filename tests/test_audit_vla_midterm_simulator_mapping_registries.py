from tools.audit_vla_midterm_simulator_mapping import OBJECT_REGISTRIES


def test_audit_uses_the_default_nonempty_robocasa_registries() -> None:
    assert OBJECT_REGISTRIES == ("objaverse", "lightwheel")
