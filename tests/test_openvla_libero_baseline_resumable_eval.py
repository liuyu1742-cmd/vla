import argparse
import json
from pathlib import Path

import numpy as np
import pytest


class FakeTask:
    language = "put the bowl on the stove"


class FakeSuite:
    n_tasks = 1

    def get_task(self, task_id):
        assert task_id == 0
        return FakeTask()

    def get_task_init_states(self, task_id):
        assert task_id == 0
        return ["fixed-state-0", "fixed-state-1"]


class FakeEnv:
    def __init__(self, *, done_on_step=2):
        self.done_on_step = done_on_step
        self.steps = 0
        self.closed = False
        self.initial_states = []

    def reset(self):
        return None

    def set_init_state(self, state):
        self.initial_states.append(state)
        return {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)}

    def step(self, action):
        self.steps += 1
        return {"agentview_image": np.full((4, 4, 3), self.steps, dtype=np.uint8)}, 0.0, self.steps >= self.done_on_step, {}

    def close(self):
        self.closed = True


class FakeDeps:
    def __init__(self, *, env=None, fail_model=False):
        self.env = env or FakeEnv()
        self.fail_model = fail_model
        self.model_loads = 0
        self.saved_images = []
        self.saved_videos = []

    def set_seed(self, seed):
        self.seed = seed

    def load_suite(self, suite):
        assert suite == "libero_goal"
        return FakeSuite()

    def load_model(self, checkpoint):
        self.model_loads += 1
        if self.fail_model:
            raise RuntimeError("model unavailable")
        return object(), object(), {"libero_goal": {}}

    def create_env(self, task):
        return self.env

    def preprocess_image(self, observation):
        return observation["agentview_image"]

    def predict_action(self, model, processor, image, task, unnorm_key):
        return np.array([0.25, 0, 0, 0, 0, 0, 0.75], dtype=float)

    def save_image(self, path, image):
        self.saved_images.append(Path(path).name)

    def save_video(self, path, frames):
        self.saved_videos.append(Path(path).name)


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"known-checkpoint")
    return checkpoint


def test_official_suite_mapping_keeps_four_distinct_local_baselines_and_horizons():
    from tools.openvla_libero_baseline_resumable_eval import BASELINE_CHECKPOINTS, OFFICIAL_HORIZONS

    assert list(BASELINE_CHECKPOINTS) == ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    assert BASELINE_CHECKPOINTS["libero_10"].name == "openvla-7b-finetuned-libero-10-gitcode"
    assert len(set(BASELINE_CHECKPOINTS.values())) == 4
    assert OFFICIAL_HORIZONS == {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def test_eval_writes_atomic_episode_evidence_fixed_states_and_visuals(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    deps_impl = FakeDeps(env=FakeEnv(done_on_step=3))
    report = evaluate_suite(
        suite="libero_goal",
        checkpoint=_checkpoint(tmp_path),
        output_dir=tmp_path / "results",
        task_ids=[0],
        trials_per_task=1,
        seed=17,
        num_steps_wait=1,
        dependencies=EvalDependencies.from_object(deps_impl),
    )

    episode_path = tmp_path / "results" / "libero_goal" / "seed_17" / "episodes" / "task_000" / "trial_000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    assert report["completed_episodes"] == 1
    assert episode["success"] is True
    assert episode["done"] is True
    assert episode["initial_state_index"] == 0
    assert episode["suite"] == "libero_goal"
    assert episode["seed"] == 17
    assert episode["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert episode["uses_expert_recovery"] is False
    assert episode["horizon"] == 300
    assert deps_impl.env.initial_states == ["fixed-state-0"]
    assert deps_impl.env.closed is True
    assert deps_impl.model_loads == 1
    assert set(deps_impl.saved_images) == {"first.png", "middle.png", "last.png"}
    assert deps_impl.saved_videos == ["rollout.mp4"]
    assert not list(episode_path.parent.glob("*.tmp"))


def test_resume_skips_only_matching_completed_episode_without_reloading_model(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    checkpoint = _checkpoint(tmp_path)
    first = FakeDeps(env=FakeEnv(done_on_step=2))
    kwargs = dict(
        suite="libero_goal", checkpoint=checkpoint, output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=1, seed=7, num_steps_wait=0,
    )
    evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(first))
    second = FakeDeps()
    report = evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(second))

    assert report["skipped_episodes"] == 1
    assert second.model_loads == 0


def test_partial_resume_counts_skipped_episode_and_executes_pending_trial(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    checkpoint = _checkpoint(tmp_path)
    base = dict(
        suite="libero_goal", checkpoint=checkpoint, output_dir=tmp_path / "results", task_ids=[0],
        seed=7, num_steps_wait=0,
    )
    evaluate_suite(
        **base, trials_per_task=1, dependencies=EvalDependencies.from_object(FakeDeps(env=FakeEnv(done_on_step=1)))
    )
    pending_deps = FakeDeps(env=FakeEnv(done_on_step=1))
    report = evaluate_suite(
        **base, trials_per_task=2, dependencies=EvalDependencies.from_object(pending_deps)
    )

    assert report["skipped_episodes"] == 1
    assert report["completed_episodes"] == 2
    assert pending_deps.model_loads == 1
    assert pending_deps.env.initial_states == ["fixed-state-1"]


@pytest.mark.parametrize("raw", ["0", "-1", "not-an-integer"])
def test_max_episode_steps_cli_type_rejects_non_positive_values(raw):
    from tools.openvla_libero_baseline_resumable_eval import parse_positive_int

    assert parse_positive_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_int(raw)


def test_max_episode_steps_caps_horizon_and_is_strict_resume_identity(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, ResumeMismatchError, evaluate_suite

    checkpoint = _checkpoint(tmp_path)
    base = dict(
        suite="libero_goal", checkpoint=checkpoint, output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=1, seed=7, num_steps_wait=0,
    )
    report = evaluate_suite(
        **base,
        max_episode_steps=2,
        dependencies=EvalDependencies.from_object(FakeDeps(env=FakeEnv(done_on_step=999))),
    )
    episode = json.loads(Path(report["episode_paths"][0]).read_text(encoding="utf-8"))

    assert episode["decision_steps"] == 2
    assert episode["horizon"] == 2
    assert episode["max_episode_steps"] == 2
    assert episode["effective_horizon"] == 2
    assert report["max_episode_steps"] == 2
    assert report["effective_horizon"] == 2

    with pytest.raises(ResumeMismatchError, match="max_episode_steps"):
        evaluate_suite(
            **base,
            max_episode_steps=3,
            dependencies=EvalDependencies.from_object(FakeDeps()),
        )


def test_resume_accepts_legacy_uncapped_official_horizon_without_rewriting_evidence(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, ResumeMismatchError, evaluate_suite

    class FiveStateSuite(FakeSuite):
        def get_task_init_states(self, task_id):
            assert task_id == 0
            return [f"fixed-state-{index}" for index in range(5)]

    def five_state_deps(*, env=None):
        deps = FakeDeps(env=env)
        deps.load_suite = lambda suite: FiveStateSuite()
        return deps

    checkpoint = _checkpoint(tmp_path)
    kwargs = dict(
        suite="libero_goal", checkpoint=checkpoint, output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=5, seed=7, num_steps_wait=0,
    )
    first = evaluate_suite(
        **kwargs,
        dependencies=EvalDependencies.from_object(five_state_deps(env=FakeEnv(done_on_step=1))),
    )
    paths = [Path(item) for item in first["episode_paths"]]
    legacy_text = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("max_episode_steps")
        payload.pop("effective_horizon")
        legacy_text[path] = json.dumps(payload, sort_keys=True)
        path.write_text(legacy_text[path], encoding="utf-8")

    resumed_deps = five_state_deps()
    resumed = evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(resumed_deps))

    assert resumed["skipped_episodes"] == 5
    assert resumed_deps.model_loads == 0
    assert all(path.read_text(encoding="utf-8") == legacy_text[path] for path in paths)

    with pytest.raises(ResumeMismatchError, match="max_episode_steps"):
        evaluate_suite(
            **kwargs,
            max_episode_steps=2,
            dependencies=EvalDependencies.from_object(five_state_deps()),
        )

    mismatched = json.loads(paths[0].read_text(encoding="utf-8"))
    mismatched["horizon"] = 299
    paths[0].write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ResumeMismatchError, match="effective_horizon"):
        evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(five_state_deps()))


@pytest.mark.parametrize("missing_field", ["max_episode_steps", "effective_horizon"])
def test_resume_rejects_episode_missing_only_one_horizon_identity_field(tmp_path, missing_field):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, ResumeMismatchError, evaluate_suite

    checkpoint = _checkpoint(tmp_path)
    kwargs = dict(
        suite="libero_goal", checkpoint=checkpoint, output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=1, seed=7, num_steps_wait=0,
    )
    report = evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(FakeDeps()))
    path = Path(report["episode_paths"][0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(missing_field)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResumeMismatchError, match=missing_field):
        evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(FakeDeps()))


def test_resume_rejects_episode_with_different_checkpoint_identity(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, ResumeMismatchError, evaluate_suite

    checkpoint = _checkpoint(tmp_path)
    deps = FakeDeps()
    kwargs = dict(suite="libero_goal", checkpoint=checkpoint, output_dir=tmp_path / "results", task_ids=[0], trials_per_task=1, seed=7, num_steps_wait=0)
    evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(deps))
    episode_path = tmp_path / "results" / "libero_goal" / "seed_7" / "episodes" / "task_000" / "trial_000.json"
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "wrong"
    episode_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResumeMismatchError, match="checkpoint_sha256"):
        evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(FakeDeps()))


def test_model_load_failure_is_persisted_for_each_requested_episode(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    report = evaluate_suite(
        suite="libero_goal", checkpoint=_checkpoint(tmp_path), output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=2, seed=7, dependencies=EvalDependencies.from_object(FakeDeps(fail_model=True)),
    )

    assert report["completed_episodes"] == 0
    assert report["error_episodes"] == 2
    assert report["integrity_status"] == "invalid"
    assert report["success_rate"] is None
    assert report["successes"] == 0
    paths = sorted((tmp_path / "results" / "libero_goal" / "seed_7" / "episodes" / "task_000").glob("*.json"))
    assert len(paths) == 2
    assert all("RuntimeError: model unavailable" in json.loads(path.read_text())["error"] for path in paths)


def test_suite_build_failure_persists_default_ten_task_error_rows(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    deps = FakeDeps()
    deps.load_suite = lambda suite: (_ for _ in ()).throw(RuntimeError("LIBERO unavailable"))
    report = evaluate_suite(
        suite="libero_goal", checkpoint=_checkpoint(tmp_path), output_dir=tmp_path / "results",
        dependencies=EvalDependencies.from_object(deps),
    )

    assert report["requested_episodes"] == 10
    assert report["completed_episodes"] == 0
    assert report["error_episodes"] == 10
    assert report["integrity_status"] == "invalid"
    assert report["success_rate"] is None
    assert all(json.loads(path.read_text())["status"] == "error" for path in (tmp_path / "results").rglob("trial_*.json"))


class WarmupDoneEnv(FakeEnv):
    def __init__(self):
        super().__init__(done_on_step=999)
        self.dummy_steps = 0
        self.policy_steps = 0

    def step(self, action):
        self.steps += 1
        is_dummy = float(action[0]) == 0.0
        if is_dummy:
            self.dummy_steps += 1
            done = True
        else:
            self.policy_steps += 1
            done = self.policy_steps == 2
        obs = {"agentview_image": np.full((4, 4, 3), self.steps, dtype=np.uint8)}
        return obs, 0.0, done, {}


def test_warmup_done_is_ignored_until_policy_action_reports_done(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    env = WarmupDoneEnv()
    report = evaluate_suite(
        suite="libero_goal", checkpoint=_checkpoint(tmp_path), output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=1, seed=7, num_steps_wait=3, dependencies=EvalDependencies.from_object(FakeDeps(env=env)),
    )
    episode_path = Path(report["episode_paths"][0])
    episode = json.loads(episode_path.read_text(encoding="utf-8"))

    assert env.dummy_steps == 3
    assert env.policy_steps == 2
    assert episode["decision_steps"] == 2
    assert episode["success"] is True


def test_resume_rejects_changed_wait_protocol(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, ResumeMismatchError, evaluate_suite

    checkpoint = _checkpoint(tmp_path)
    base = dict(suite="libero_goal", checkpoint=checkpoint, output_dir=tmp_path / "results", task_ids=[0], trials_per_task=1, seed=7)
    evaluate_suite(**base, num_steps_wait=0, dependencies=EvalDependencies.from_object(FakeDeps()))
    with pytest.raises(ResumeMismatchError, match="num_steps_wait"):
        evaluate_suite(**base, num_steps_wait=1, dependencies=EvalDependencies.from_object(FakeDeps()))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("schema_version", "old-schema"), ("horizon", 999), ("initial_state_index", 9), ("uses_expert_recovery", True)],
)
def test_resume_rejects_tampered_protocol_identity(tmp_path, field, bad_value):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, ResumeMismatchError, evaluate_suite

    checkpoint = _checkpoint(tmp_path)
    kwargs = dict(suite="libero_goal", checkpoint=checkpoint, output_dir=tmp_path / "results", task_ids=[0], trials_per_task=1, seed=7, num_steps_wait=0)
    report = evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(FakeDeps()))
    path = Path(report["episode_paths"][0])
    episode = json.loads(path.read_text(encoding="utf-8"))
    episode[field] = bad_value
    path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(ResumeMismatchError, match=field):
        evaluate_suite(**kwargs, dependencies=EvalDependencies.from_object(FakeDeps()))


def test_checkpoint_hash_failure_writes_explicit_unavailable_error_rows(tmp_path, monkeypatch):
    import tools.openvla_libero_baseline_resumable_eval as runner

    monkeypatch.setattr(runner, "checkpoint_sha256", lambda checkpoint: (_ for _ in ()).throw(OSError("cannot read weights")))
    report = runner.evaluate_suite(
        suite="libero_goal", checkpoint=_checkpoint(tmp_path), output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=2, dependencies=runner.EvalDependencies.from_object(FakeDeps()),
    )
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report["episode_paths"]]

    assert report["error_episodes"] == 2
    assert report["integrity_status"] == "invalid"
    assert all(row["checkpoint_sha256"] is None for row in rows)
    assert all(row["checkpoint_hash_status"] == "unavailable" for row in rows)
    assert all("cannot read weights" in row["error"] for row in rows)


def test_seed_failure_writes_every_pending_trial_as_error(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    deps = FakeDeps()
    deps.set_seed = lambda seed: (_ for _ in ()).throw(RuntimeError("seed failed"))
    report = evaluate_suite(
        suite="libero_goal", checkpoint=_checkpoint(tmp_path), output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=2, dependencies=EvalDependencies.from_object(deps),
    )

    assert report["error_episodes"] == 2
    assert all("seed failed" in json.loads(Path(path).read_text())["error"] for path in report["episode_paths"])


@pytest.mark.parametrize("failure_point", ["get_task", "get_task_init_states"])
def test_task_metadata_failure_writes_all_affected_trials(tmp_path, failure_point):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    class BrokenSuite(FakeSuite):
        def get_task(self, task_id):
            if failure_point == "get_task":
                raise RuntimeError("task metadata failed")
            return super().get_task(task_id)

        def get_task_init_states(self, task_id):
            if failure_point == "get_task_init_states":
                raise RuntimeError("init states failed")
            return super().get_task_init_states(task_id)

    deps = FakeDeps()
    deps.load_suite = lambda suite: BrokenSuite()
    report = evaluate_suite(
        suite="libero_goal", checkpoint=_checkpoint(tmp_path), output_dir=tmp_path / "results", task_ids=[0],
        trials_per_task=2, dependencies=EvalDependencies.from_object(deps),
    )

    assert report["completed_episodes"] == 0
    assert report["error_episodes"] == 2
    assert report["success_rate"] is None


def test_close_failure_is_persisted_in_episode_and_summary_diagnostics(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    class CloseFailEnv(FakeEnv):
        def close(self):
            raise RuntimeError("close failed")

    report = evaluate_suite(
        suite="libero_goal", checkpoint=_checkpoint(tmp_path), output_dir=tmp_path / "results", task_ids=[0],
        dependencies=EvalDependencies.from_object(FakeDeps(env=CloseFailEnv(done_on_step=1))),
    )
    episode = json.loads(Path(report["episode_paths"][0]).read_text(encoding="utf-8"))

    assert any("close failed" in item for item in episode["diagnostics"])
    assert any("close failed" in item for item in report["diagnostics"])


def test_python_api_rejects_duplicate_task_ids(tmp_path):
    from tools.openvla_libero_baseline_resumable_eval import EvalDependencies, evaluate_suite

    with pytest.raises(ValueError, match="duplicate task IDs"):
        evaluate_suite(
            suite="libero_goal", checkpoint=_checkpoint(tmp_path), output_dir=tmp_path / "results", task_ids=[0, 0],
            dependencies=EvalDependencies.from_object(FakeDeps()),
        )
