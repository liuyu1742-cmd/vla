"""Regression contract for the calibrated storage-box candidate."""


def test_storage_box_relation_selects_the_calibrated_container_oracle():
    from tools.openvla_household_object_rollout import recovery_oracle_class
    from tools.pick_place_oracle.safe_cabinet import SafeCabinetPickPlaceOracle
    from tools.pick_place_oracle.storage_box import StorageBoxPickPlaceOracle

    assert recovery_oracle_class("organizing::storage_box") is StorageBoxPickPlaceOracle
    assert recovery_oracle_class(None) is SafeCabinetPickPlaceOracle
