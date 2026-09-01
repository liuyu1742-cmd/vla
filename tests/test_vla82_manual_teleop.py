from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import vla82_manual_teleop as teleop


def test_resolve_selection_returns_requested_runtime_spec() -> None:
    spec = SimpleNamespace(selection_id="VLA82-017")
    assert teleop.resolve_selection("vla82-017", [spec]) is spec


def test_resolve_selection_rejects_unknown_identifier() -> None:
    with pytest.raises(ValueError, match="unknown VLA82 selection"):
        teleop.resolve_selection("VLA82-999", [])


def test_episode_metadata_marks_capture_manual_and_unverified() -> None:
    metadata = teleop.build_episode_metadata(
        selection_id="VLA82-017",
        seed=820017,
        task_class="PickPlaceCounterToCabinet",
        camera="robot0_agentview_left",
        action_count=2,
        request=SimpleNamespace(
            manipulated_objects=("bottle",),
            source_fixture="counter",
            target_fixture="cabinet",
            target_relation="inside",
        ),
        fingerprint={"sha256": "scene-hash"},
        friction_evidence={"material": "plastic"},
    )

    assert metadata["status"] == "manual_unverified"
    assert metadata["action_count"] == 2
    assert metadata["scene_sha256"] == "scene-hash"
    assert metadata["target_relation"] == "inside"


def test_save_episode_writes_native_actions_and_metadata(tmp_path) -> None:
    output = teleop.save_episode(
        tmp_path / "episode",
        [np.zeros(12, dtype=np.float32)],
        [0.25],
        {"status": "manual_unverified"},
        "<mujoco/>",
    )

    assert (output / "actions.npz").is_file()
    assert (output / "model.xml").read_text(encoding="utf-8") == "<mujoco/>"
    assert json.loads((output / "episode.json").read_text(encoding="utf-8"))["status"] == "manual_unverified"
    with np.load(output / "actions.npz") as saved:
        assert saved["actions"].shape == (1, 12)
        assert np.allclose(saved["timestamps"], [0.25])


def test_save_episode_rejects_empty_action_capture(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one action"):
        teleop.save_episode(tmp_path / "episode", [], [], {}, "<mujoco/>")


def test_parser_requires_selection_and_accepts_record_options() -> None:
    parser = teleop.build_parser()
    args = parser.parse_args(["--selection-id", "VLA82-017", "--record-video"])
    assert args.selection_id == "VLA82-017"
    assert args.record_video is True


def test_parser_does_not_claim_uninstalled_selection_ids_are_available() -> None:
    assert "VLA82-001 through VLA82-082" not in teleop.build_parser().format_help()


def test_environment_kwargs_use_robocasa_onscreen_switch_without_duplicate_renderer_flag() -> None:
    from tools.vla82_full_sim.environment import robocasa_environment_kwargs

    request = SimpleNamespace(
        selection_id="VLA82-017",
        camera_names=("robot0_agentview_left", "robot0_eye_in_hand"),
        primary_asset=SimpleNamespace(kind="fixture_part"),
    )
    kwargs = robocasa_environment_kwargs(request, 820017, has_renderer=True, renderer="mjviewer")

    assert kwargs["render_onscreen"] is True
    assert "has_renderer" not in kwargs
    assert kwargs["renderer"] == "mjviewer"
    assert kwargs["render_offscreen"] is True
