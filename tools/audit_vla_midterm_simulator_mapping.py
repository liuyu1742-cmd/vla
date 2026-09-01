"""Use RoboCasa's default nonempty object registries for the VLA audit."""

from tools import audit_vla_midterm_simulator_mapping_v2 as implementation


OBJECT_REGISTRIES = ("objaverse", "lightwheel")
implementation.OBJECT_REGISTRIES = OBJECT_REGISTRIES
implementation.legacy._one_reset = implementation._one_reset

install_target_object_group = implementation.install_target_object_group
audit_relation = implementation.audit_relation
run_audit = implementation.run_audit
build_parser = implementation.build_parser
main = implementation.main


if __name__ == "__main__":
    raise SystemExit(main())
