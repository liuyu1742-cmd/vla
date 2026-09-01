import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.eval_libero_diffusion_policy import (
    PolicyState,
    SUITE_HORIZONS,
    _configure_libero,
    _default_observation_adapter,
    evaluate_suite,
    load_policy_checkpoint,
    policy_action,
)
from tools.libero_diffusion_contracts import ActionNormalizer
from tools.libero_diffusion_model import DiffusionPolicyConfig


def test_configure_libero_uses_vendored_checkout(monkeypatch):
    import tools.eval_libero_diffusion_policy as evaluation

    monkeypatch.setattr(
        evaluation.sys,
        "path",
        ["foreign", str(evaluation.LIBERO_ROOT), "tail"],
    )
    monkeypatch.setenv("LIBERO_CONFIG_PATH", "C:/ambient/libero")

    _configure_libero()

    assert evaluation.sys.path[0] == str(evaluation.LIBERO_ROOT)
    assert evaluation.sys.path.count(str(evaluation.LIBERO_ROOT)) == 1
    assert Path(evaluation.os.environ["LIBERO_CONFIG_PATH"]) == (
        evaluation.REPO_ROOT / "outputs" / "libero_config"
    )


def test_configure_libero_rejects_foreign_preimport(monkeypatch, tmp_path: Path):
    import types
    import tools.eval_libero_diffusion_policy as evaluation

    foreign = types.SimpleNamespace(__file__=str(tmp_path / "site-packages" / "libero" / "__init__.py"))
    monkeypatch.setitem(evaluation.sys.modules, "libero", foreign)

    with pytest.raises(RuntimeError, match="foreign libero module already imported"):
        _configure_libero()


def test_configure_libero_accepts_vendored_namespace_package(monkeypatch):
    import types
    import tools.eval_libero_diffusion_policy as evaluation

    namespace = types.SimpleNamespace(
        __file__=None,
        __path__=[str(evaluation.LIBERO_ROOT / "libero")],
    )
    monkeypatch.setitem(evaluation.sys.modules, "libero", namespace)
    monkeypatch.delitem(evaluation.sys.modules, "libero.libero", raising=False)

    _configure_libero()


def test_configure_libero_rejects_cached_foreign_config(monkeypatch, tmp_path: Path):
    import types
    import tools.eval_libero_diffusion_policy as evaluation

    namespace = types.SimpleNamespace(
        __file__=None,
        __path__=[str(evaluation.LIBERO_ROOT / "libero")],
    )
    inner = types.SimpleNamespace(
        __file__=str(evaluation.LIBERO_ROOT / "libero" / "libero" / "__init__.py"),
        config_file=str(tmp_path / "foreign" / "config.yaml"),
    )
    monkeypatch.setitem(evaluation.sys.modules, "libero", namespace)
    monkeypatch.setitem(evaluation.sys.modules, "libero.libero", inner)

    with pytest.raises(RuntimeError, match="cached foreign LIBERO config"):
        _configure_libero()


def test_configure_libero_fails_closed_on_missing_checkout(monkeypatch, tmp_path: Path):
    import tools.eval_libero_diffusion_policy as evaluation

    monkeypatch.setattr(evaluation, "LIBERO_ROOT", tmp_path / "missing-libero")

    with pytest.raises(FileNotFoundError, match="vendored LIBERO package"):
        _configure_libero()


def test_configure_libero_fails_closed_on_missing_config(monkeypatch, tmp_path: Path):
    import tools.eval_libero_diffusion_policy as evaluation

    monkeypatch.setattr(evaluation, "LIBERO_CONFIG_DIR", tmp_path / "missing-config")

    with pytest.raises(FileNotFoundError, match="validated LIBERO config"):
        _configure_libero()


def test_configure_libero_routes_numba_cache_to_writable_workspace(
    monkeypatch, tmp_path: Path
):
    import tools.eval_libero_diffusion_policy as evaluation

    cache_dir = tmp_path / "numba-cache"
    monkeypatch.setattr(evaluation, "NUMBA_CACHE_DIR", cache_dir)
    monkeypatch.setenv("NUMBA_CACHE_DIR", "C:/read-only/site-packages")

    _configure_libero()

    assert evaluation.os.environ["NUMBA_CACHE_DIR"] == str(cache_dir)
    assert cache_dir.is_dir()


class FakeChunkPolicy:
    def __init__(self) -> None:
        self.calls = []

    def sample_actions(self, condition, **_kwargs):
        self.calls.append(condition)
        actions = torch.zeros((1, 8, 7), dtype=torch.float32)
        actions[:, 0::2, -1] = 1.0
        actions[:, 1::2, -1] = -1.0
        return actions


class FakeEnvironment:
    def __init__(self, *, done_at: int | None = None, error_at: int | None = None):
        self.done_at = done_at
        self.error_at = error_at
        self.actions = []
        self.initial_state = None

    @staticmethod
    def _observation() -> dict[str, np.ndarray]:
        return {
            "image": np.full((3, 4, 3), 127, dtype=np.uint8),
            "state": np.asarray([0.1, 0.2], dtype=np.float32),
        }

    def reset(self):
        return self._observation()

    def set_init_state(self, initial_state):
        self.initial_state = initial_state
        return self._observation()

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        if self.error_at is not None and len(self.actions) == self.error_at:
            raise RuntimeError("simulator disconnected")
        done = self.done_at == len(self.actions)
        return self._observation(), 999.0, done, {"ignored": True}


class FakeSuite:
    n_tasks = 1

    def get_task(self, task_id):
        assert task_id == 0
        return type("Task", (), {"language": "put the mug away"})()

    def get_task_init_states(self, task_id):
        assert task_id == 0
        return ["official-init-0", "official-init-1"]


def _normalizer() -> ActionNormalizer:
    return ActionNormalizer.fit(
        np.asarray([[0.0] * 7, [1.0] * 7], dtype=np.float32)
    )


def _adapter(observation):
    return observation["image"], observation["state"]


def test_policy_history_gripper_contract_and_eight_action_queue():
    policy = FakeChunkPolicy()
    state = PolicyState(obs_horizon=2)
    normalizer = ActionNormalizer.fit(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    visual = np.arange(4, dtype=np.float32)
    proprio = np.arange(2, dtype=np.float32)

    emitted = [
        policy_action(
            policy,
            state,
            visual_feature=visual,
            proprio=proprio,
            instruction="open the drawer",
            normalizer=normalizer,
            device=torch.device("cpu"),
        )
        for _ in range(8)
    ]

    assert len(policy.calls) == 1
    torch.testing.assert_close(
        policy.calls[0]["visual"][0, 0],
        policy.calls[0]["visual"][0, 1],
    )
    assert emitted[0][-1] == -1.0
    assert emitted[1][-1] == 1.0

    policy_action(
        policy,
        state,
        visual_feature=visual,
        proprio=proprio,
        instruction="open the drawer",
        normalizer=normalizer,
        device=torch.device("cpu"),
    )
    assert len(policy.calls) == 2


def test_suite_horizons_cover_the_four_supported_libero_suites():
    assert SUITE_HORIZONS == {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }


def test_default_adapter_rotates_image_and_builds_libero_proprio_without_external_helpers():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rotated, state = _default_observation_adapter(
        {
            "agentview_image": image,
            "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0]),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.asarray([0.2, 0.3]),
        }
    )
    np.testing.assert_array_equal(rotated, image[::-1, ::-1])
    np.testing.assert_allclose(state, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.2, 0.3])


def test_default_adapter_returns_contiguous_image_for_torch_encoder():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    rotated, _ = _default_observation_adapter(
        {"agentview_image": image, "state": np.zeros(8, dtype=np.float32)}
    )

    assert rotated.flags.c_contiguous


def test_evaluate_suite_applies_settle_horizon_and_uses_only_done(tmp_path: Path):
    environment = FakeEnvironment(done_at=None)
    policy = FakeChunkPolicy()
    summary = evaluate_suite(
        suite_name="libero_spatial",
        task_suite=FakeSuite(),
        policy=policy,
        normalizer=_normalizer(),
        encoder=lambda images: np.zeros((len(images), 4), dtype=np.float32),
        checkpoint_sha256="checkpoint-sha",
        seed=7,
        output_dir=tmp_path,
        trials_per_task=1,
        task_ids=[0],
        env_factory=lambda _task: environment,
        observation_adapter=_adapter,
        horizon=2,
        settle_steps=1,
        write_video=False,
    )

    assert len(environment.actions) == 3
    assert len(policy.calls) == 1
    assert summary["successes"] == 0
    assert summary["success_rate"] == 0.0
    episode = json.loads((tmp_path / "episodes" / "task_000_trial_000.json").read_text())
    assert episode["success"] is False
    assert episode["steps"] == 2
    assert episode["initial_state_index"] == 0
    assert episode["checkpoint_sha256"] == "checkpoint-sha"
    assert episode["action_summary"]["policy_queries"] == 1
    assert {path.name for path in (tmp_path / "episodes").glob("*.png")} == {
        "task_000_trial_000_first.png",
        "task_000_trial_000_middle.png",
        "task_000_trial_000_final.png",
    }


def test_done_is_the_only_success_signal_even_when_reward_is_large(tmp_path: Path):
    environment = FakeEnvironment(done_at=2)
    summary = evaluate_suite(
        suite_name="libero_goal",
        task_suite=FakeSuite(),
        policy=FakeChunkPolicy(),
        normalizer=_normalizer(),
        encoder=lambda images: np.zeros((len(images), 4), dtype=np.float32),
        checkpoint_sha256="sha",
        seed=3,
        output_dir=tmp_path,
        trials_per_task=1,
        task_ids=[0],
        env_factory=lambda _task: environment,
        observation_adapter=_adapter,
        horizon=4,
        settle_steps=0,
        write_video=False,
    )

    assert summary["successes"] == 1
    episode = json.loads((tmp_path / "episodes" / "task_000_trial_000.json").read_text())
    assert episode["success"] is True
    assert episode["steps"] == 2


def test_episode_error_is_persisted_as_unsuccessful(tmp_path: Path):
    summary = evaluate_suite(
        suite_name="libero_object",
        task_suite=FakeSuite(),
        policy=FakeChunkPolicy(),
        normalizer=_normalizer(),
        encoder=lambda images: np.zeros((len(images), 4), dtype=np.float32),
        checkpoint_sha256="sha",
        seed=3,
        output_dir=tmp_path,
        trials_per_task=1,
        task_ids=[0],
        env_factory=lambda _task: FakeEnvironment(error_at=1),
        observation_adapter=_adapter,
        horizon=2,
        settle_steps=0,
        write_video=False,
    )

    assert summary["successes"] == 0
    episode = json.loads((tmp_path / "episodes" / "task_000_trial_000.json").read_text())
    assert episode["success"] is False
    assert "simulator disconnected" in episode["error"]


def test_initialization_error_is_also_persisted_as_unsuccessful(tmp_path: Path):
    class BrokenInitializationEnvironment(FakeEnvironment):
        def set_init_state(self, _initial_state):
            raise RuntimeError("initial state rejected")

    summary = evaluate_suite(
        suite_name="libero_object",
        task_suite=FakeSuite(),
        policy=FakeChunkPolicy(),
        normalizer=_normalizer(),
        encoder=lambda images: np.zeros((len(images), 4), dtype=np.float32),
        checkpoint_sha256="sha",
        seed=3,
        output_dir=tmp_path,
        trials_per_task=1,
        task_ids=[0],
        env_factory=lambda _task: BrokenInitializationEnvironment(),
        observation_adapter=_adapter,
        horizon=2,
        settle_steps=0,
        write_video=False,
    )

    episode = json.loads((tmp_path / "episodes" / "task_000_trial_000.json").read_text())
    assert summary["successes"] == 0
    assert episode["success"] is False
    assert "initial state rejected" in episode["error"]
    assert episode["suite"] == "libero_object"
    assert episode["checkpoint_sha256"] == "sha"
    assert episode["seed"] == 3
    assert episode["task_id"] == 0
    assert episode["trial_index"] == 0
    assert episode["initial_state_index"] == 0


def test_environment_construction_error_persists_every_trial_identity(tmp_path: Path):
    summary = evaluate_suite(
        suite_name="libero_object",
        task_suite=FakeSuite(),
        policy=FakeChunkPolicy(),
        normalizer=_normalizer(),
        encoder=lambda images: np.zeros((len(images), 4), dtype=np.float32),
        checkpoint_sha256="construction-sha",
        seed=17,
        output_dir=tmp_path,
        trials_per_task=2,
        task_ids=[0],
        env_factory=lambda _task: (_ for _ in ()).throw(
            RuntimeError("renderer unavailable")
        ),
        observation_adapter=_adapter,
        horizon=2,
        settle_steps=0,
        write_video=False,
    )

    assert summary["episodes"] == 2
    assert summary["successes"] == 0
    for trial_index in range(2):
        episode = json.loads(
            (tmp_path / "episodes" / f"task_000_trial_{trial_index:03d}.json").read_text()
        )
        assert episode["success"] is False
        assert "renderer unavailable" in episode["error"]
        assert episode["suite"] == "libero_object"
        assert episode["checkpoint_sha256"] == "construction-sha"
        assert episode["seed"] == 17
        assert episode["task_id"] == 0
        assert episode["trial_index"] == trial_index


def test_one_environment_is_reused_per_task_and_closed_once(tmp_path: Path):
    class CloseTrackingEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    environments = []

    def build_environment(_task):
        environment = CloseTrackingEnvironment()
        environments.append(environment)
        return environment

    summary = evaluate_suite(
        suite_name="libero_10",
        task_suite=FakeSuite(),
        policy=FakeChunkPolicy(),
        normalizer=_normalizer(),
        encoder=lambda images: np.zeros((len(images), 4), dtype=np.float32),
        checkpoint_sha256="reuse-sha",
        seed=5,
        output_dir=tmp_path,
        trials_per_task=2,
        task_ids=[0],
        env_factory=build_environment,
        observation_adapter=_adapter,
        horizon=1,
        settle_steps=0,
        write_video=False,
    )

    assert summary["episodes"] == 2
    assert len(environments) == 1
    assert environments[0].close_calls == 1


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("obs_horizon", 3), ("action_horizon", 4)],
)
def test_checkpoint_rejects_nonfixed_policy_horizons(
    tmp_path: Path, field: str, invalid_value: int
):
    config = asdict(DiffusionPolicyConfig())
    config[field] = invalid_value
    checkpoint = tmp_path / f"invalid_{field}.pt"
    torch.save(
        {
            "model": {},
            "model_config": config,
            "normalizer": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match=field):
        load_policy_checkpoint(checkpoint, device="cpu")


def test_resume_requires_matching_checkpoint_seed_task_and_trial(tmp_path: Path):
    first_environment = FakeEnvironment(done_at=1)
    common = dict(
        suite_name="libero_10",
        task_suite=FakeSuite(),
        policy=FakeChunkPolicy(),
        normalizer=_normalizer(),
        encoder=lambda images: np.zeros((len(images), 4), dtype=np.float32),
        checkpoint_sha256="sha-a",
        seed=11,
        output_dir=tmp_path,
        trials_per_task=1,
        task_ids=[0],
        observation_adapter=_adapter,
        horizon=1,
        settle_steps=0,
        write_video=False,
    )
    evaluate_suite(env_factory=lambda _task: first_environment, **common)
    resumed = evaluate_suite(
        env_factory=lambda _task: (_ for _ in ()).throw(AssertionError("must resume")),
        **common,
    )
    assert resumed["resumed_episodes"] == 1

    changed_checkpoint_environment = FakeEnvironment(done_at=None)
    evaluate_suite(
        env_factory=lambda _task: changed_checkpoint_environment,
        checkpoint_sha256="sha-b",
        **{key: value for key, value in common.items() if key != "checkpoint_sha256"},
    )
    assert len(changed_checkpoint_environment.actions) == 1


def test_video_failure_is_recorded_without_changing_done_success(tmp_path: Path):
    def missing_ffmpeg_after_json(video_path, _frames):
        if not video_path.with_suffix(".json").is_file():
            raise RuntimeError("episode json was not written before video")
        raise RuntimeError("ffmpeg missing")

    summary = evaluate_suite(
        suite_name="libero_goal",
        task_suite=FakeSuite(),
        policy=FakeChunkPolicy(),
        normalizer=_normalizer(),
        encoder=lambda images: np.zeros((len(images), 4), dtype=np.float32),
        checkpoint_sha256="sha",
        seed=3,
        output_dir=tmp_path,
        trials_per_task=1,
        task_ids=[0],
        env_factory=lambda _task: FakeEnvironment(done_at=1),
        observation_adapter=_adapter,
        horizon=2,
        settle_steps=0,
        video_writer=missing_ffmpeg_after_json,
    )

    episode = json.loads((tmp_path / "episodes" / "task_000_trial_000.json").read_text())
    assert summary["successes"] == 1
    assert episode["success"] is True
    assert "ffmpeg missing" in episode["media_error"]
