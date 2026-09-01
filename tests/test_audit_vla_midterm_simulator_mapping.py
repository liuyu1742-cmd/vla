from tools.audit_vla_midterm_simulator_mapping import install_target_object_group


class _FakeTask:
    def _get_obj_cfgs(self):
        return [
            {"name": "obj", "obj_groups": "all"},
            {"name": "distractor", "obj_groups": "all"},
        ]


def test_install_target_object_group_only_overrides_target() -> None:
    original = install_target_object_group(_FakeTask, "apple")
    try:
        configs = _FakeTask()._get_obj_cfgs()
        assert configs[0]["obj_groups"] == "apple"
        assert configs[1]["obj_groups"] == "all"
    finally:
        _FakeTask._get_obj_cfgs = original


def test_install_target_object_group_requires_nonempty_group() -> None:
    try:
        install_target_object_group(_FakeTask, "  ")
    except ValueError as exc:
        assert "object_group" in str(exc)
    else:
        raise AssertionError("empty object group must be rejected")
