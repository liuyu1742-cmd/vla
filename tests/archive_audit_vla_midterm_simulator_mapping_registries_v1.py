from tools.audit_vla_midterm_simulator_mapping import OBJECT_REGISTRIES


def test_audit_uses_all_robocasa_vla_object_registries() -> None:
    assert OBJECT_REGISTRIES == ("objaverse", "lightwheel", "aigen")
