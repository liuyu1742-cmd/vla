import json
import sys
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import tools.prepare_libero_diffusion_data as preparation
from tools.prepare_libero_diffusion_data import (
    EpisodeArrays,
    install_tfds_resource_compatibility,
    iter_rlds_episodes,
    iter_selected_episodes,
    _source_commit,
    select_episodes,
    validate_expanded_tfrecords,
    write_feature_cache,
)
from tools.libero_diffusion_contracts import ActionNormalizer
from tools.train_libero_diffusion_policy import FeatureWindowDataset


def fake_episode(suite: str, instruction: str, index: int) -> EpisodeArrays:
    return EpisodeArrays(
        images=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        states=np.zeros((2, 8), dtype=np.float32),
        actions=np.zeros((2, 7), dtype=np.float32),
        instruction=instruction,
        suite=suite,
        episode_index=index,
    )


def test_selection_caps_each_full_instruction_without_cross_suite_collisions():
    episodes = [
        fake_episode("libero_goal", "turn on the stove", index)
        for index in range(12)
    ] + [
        fake_episode("libero_spatial", "turn on the stove", index)
        for index in range(12)
    ]
    selected = select_episodes(episodes, max_per_instruction=10)
    assert len(selected) == 20
    assert [row.episode_index for row in selected[:10]] == list(range(10))


def test_tfrecord_validation_rejects_unexpanded_lfs_pointer(tmp_path: Path):
    shard = tmp_path / "suite" / "1.0.0" / "data.tfrecord-00000-of-00001"
    shard.parent.mkdir(parents=True)
    shard.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        validate_expanded_tfrecords(tmp_path)


def test_windows_tfds_resource_compatibility_exposes_shuffle_symbols(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "resource", raising=False)
    install_tfds_resource_compatibility()
    resource = sys.modules["resource"]
    assert resource.RLIMIT_NOFILE == 7
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (512, 512)
    assert resource.setrlimit(resource.RLIMIT_NOFILE, (256, 512)) is None


def test_tfds_resource_compatibility_does_not_inject_on_non_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "resource", raising=False)
    install_tfds_resource_compatibility()
    assert "resource" not in sys.modules


def test_rlds_iteration_reads_required_fields_and_transforms_gripper(tmp_path: Path):
    class Tensor:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

    class Builder:
        def as_dataset(self, *, split, shuffle_files):
            assert split == "train"
            assert shuffle_files is False
            return [{
                "steps": [
                    {
                        "observation": {
                            "image": Tensor(np.ones((4, 5, 3), dtype=np.uint8)),
                            "state": Tensor(np.arange(8, dtype=np.float32)),
                        },
                        "action": Tensor(np.array([1, 2, 3, 4, 5, 6, -1], dtype=np.float32)),
                        "language_instruction": Tensor(b"open the drawer"),
                    },
                    {
                        "observation": {
                            "image": Tensor(np.zeros((4, 5, 3), dtype=np.uint8)),
                            "state": Tensor(np.arange(8, dtype=np.float32)),
                        },
                        "action": Tensor(np.array([1, 2, 3, 4, 5, 6, 1], dtype=np.float32)),
                        "language_instruction": Tensor(b"open the drawer"),
                    },
                ]
            }]

    class TFDS:
        @staticmethod
        def builder_from_directory(path):
            suite = Path(path).parent.name
            assert suite in {
                "libero_spatial_no_noops",
                "libero_object_no_noops",
                "libero_goal_no_noops",
                "libero_10_no_noops",
            }
            return Builder() if suite == "libero_goal_no_noops" else type(
                "EmptyBuilder", (), {"as_dataset": lambda self, **_: []}
            )()

    episodes = list(iter_rlds_episodes(tmp_path, tfds_module=TFDS()))
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.suite == "libero_goal_no_noops"
    assert episode.instruction == "open the drawer"
    assert episode.images.shape == (2, 4, 5, 3)
    assert episode.actions[:, -1].tolist() == [1.0, 0.0]


def test_feature_cache_writes_exact_episode_boundaries_and_rejects_mismatch(tmp_path: Path):
    episode = fake_episode("libero_goal_no_noops", "close the drawer", 3)

    manifest = write_feature_cache(
        [episode],
        output_dir=tmp_path,
        encoder=lambda images: np.ones((len(images), 4), dtype=np.float32),
        source_commit="abc123",
        config={"image_size": 128, "encoder": "synthetic", "feature_dim": 4},
    )
    cache_path = tmp_path / manifest["episodes"][0]["path"]
    with np.load(cache_path) as cache:
        assert cache["features"].dtype == np.float16
        assert cache["features"].shape == (2, 4)
        assert cache["states"].dtype == np.float32
        assert cache["actions"].dtype == np.float32
        assert cache["instruction"].item() == "close the drawer"
        assert cache["episode_index"].item() == 3
    persisted = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert persisted["episodes"][0]["boundary"] == {"start": 0, "stop": 2}
    assert persisted["episodes"][0]["length"] == 2
    normalizer = ActionNormalizer.fit(episode.actions)
    training_dataset = FeatureWindowDataset(
        tmp_path / "dataset_manifest.json",
        normalizer=normalizer,
        obs_horizon=1,
        action_horizon=1,
    )
    assert training_dataset[0]["visual"].shape == (1, 4)
    with pytest.raises(RuntimeError, match="source_commit"):
        write_feature_cache(
            [episode],
            output_dir=tmp_path,
            encoder=lambda images: np.ones((len(images), 4), dtype=np.float32),
            source_commit="different",
            config={"image_size": 128, "encoder": "synthetic", "feature_dim": 4},
        )


def test_source_commit_is_read_from_dataset_repository(monkeypatch, tmp_path: Path):
    expected = "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
    calls = []

    def dataset_git(command, **kwargs):
        calls.append((command, kwargs))
        return expected + "\n"

    monkeypatch.setattr("tools.prepare_libero_diffusion_data.subprocess.check_output", dataset_git)
    assert _source_commit(tmp_path) == (expected, True)
    assert "-C" in calls[0][0]
    assert str(tmp_path.resolve()) in calls[0][0]
    assert f"safe.directory={tmp_path.resolve()}" in calls[0][0]


def test_source_commit_records_unavailable_dataset_git_without_faking_a_revision(
    monkeypatch, tmp_path: Path
):
    def missing_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr("tools.prepare_libero_diffusion_data.subprocess.check_output", missing_git)
    assert _source_commit(tmp_path) == (None, False)


def test_episode_selection_and_cache_writing_are_streaming(tmp_path: Path):
    events = []

    def episodes():
        for index in range(3):
            events.append(f"yield:{index}")
            yield fake_episode("suite", f"instruction {index}", index)

    def encoder(images):
        events.append("encode")
        return np.ones((len(images), 4), dtype=np.float32)

    selected = iter_selected_episodes(episodes(), max_per_instruction=1)
    assert events == []
    write_feature_cache(
        selected,
        output_dir=tmp_path,
        encoder=encoder,
        source_commit="commit",
        config={"encoder": "synthetic", "feature_dim": 4},
    )
    assert events == [
        "yield:0",
        "encode",
        "yield:1",
        "encode",
        "yield:2",
        "encode",
    ]


def test_main_starts_rlds_child_before_initializing_torch_encoder(
    monkeypatch, tmp_path: Path
):
    """TFDS producer can run while the parent builds its Torch encoder."""

    events = []
    episode = fake_episode("libero_goal_no_noops", "close the drawer", 0)

    class Child:
        returncode = 0

        def communicate(self):
            return ("decoder complete", "")

    def start(dataset_root, staging_dir, max_per_instruction, smoke_episodes, run_dir):
        events.append("stage")
        staging_dir.mkdir(parents=True)
        np.savez_compressed(
            staging_dir / "00000.npz",
            images=episode.images,
            states=episode.states,
            actions=episode.actions,
            instruction=np.asarray(episode.instruction),
            suite=np.asarray(episode.suite),
            episode_index=np.asarray(episode.episode_index),
        )
        (staging_dir / "complete.json").write_text('{"episode_count": 1}')
        return Child()

    monkeypatch.setattr(preparation, "start_rlds_decoder_subprocess", start)
    monkeypatch.setattr(
        preparation, "finish_rlds_decoder_subprocess", lambda *args: events.append("finish")
    )
    monkeypatch.setattr(preparation, "validate_expanded_tfrecords", lambda _root: {})
    monkeypatch.setattr(preparation, "_source_commit", lambda _root: (None, False))
    monkeypatch.setattr(
        preparation,
        "build_imagenet_resnet18_encoder",
        lambda _size, **_kwargs: events.append("encoder")
        or (lambda images: np.zeros((len(images), 512))),
    )
    monkeypatch.setattr(
        preparation,
        "write_feature_cache",
        lambda episodes, **_kwargs: {"episodes": list(episodes)},
    )
    monkeypatch.setattr(preparation, "publish_feature_cache", lambda *args: events.append("publish"))

    assert preparation.main(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "output"),
            "--smoke-episodes",
            "1",
        ]
    ) == 0
    assert events == ["stage", "encoder", "finish", "publish"]


def test_decoder_fatal_stderr_fails_closed(tmp_path: Path):
    class Child:
        returncode = 0

        def wait(self, timeout):
            return self.returncode

    stdout_path = tmp_path / "decoder.stdout.log"
    stderr_path = tmp_path / "decoder.stderr.log"
    stdout_path.write_text("")
    stderr_path.write_text("Windows fatal exception: access violation")

    with pytest.raises(RuntimeError, match="fatal native runtime"):
        preparation.finish_rlds_decoder_subprocess(Child(), stdout_path, stderr_path)


def test_finish_allows_extended_tensorflow_shutdown(tmp_path: Path):
    observed_timeouts = []

    class Child:
        def wait(self, timeout):
            observed_timeouts.append(timeout)
            return 0

    stdout_path = tmp_path / "decoder.stdout.log"
    stderr_path = tmp_path / "decoder.stderr.log"
    stdout_path.write_text("")
    stderr_path.write_text("")

    preparation.finish_rlds_decoder_subprocess(Child(), stdout_path, stderr_path)

    assert preparation.DECODER_NORMAL_SHUTDOWN_TIMEOUT_SECONDS >= 30
    assert preparation.DECODER_SHUTDOWN_TIMEOUT_SECONDS == 5
    assert observed_timeouts == [preparation.DECODER_NORMAL_SHUTDOWN_TIMEOUT_SECONDS]


def test_live_staging_consumes_each_episode_and_deletes_it(tmp_path: Path):
    episode = fake_episode("suite", "instruction", 0)
    np.savez_compressed(
        tmp_path / "00000.npz",
        images=episode.images,
        states=episode.states,
        actions=episode.actions,
        instruction=np.asarray(episode.instruction),
        suite=np.asarray(episode.suite),
        episode_index=np.asarray(episode.episode_index),
    )
    (tmp_path / "complete.json").write_text('{"episode_count": 1}')

    rows = list(preparation.iter_live_staged_episodes(tmp_path, child=object()))

    assert [row.instruction for row in rows] == ["instruction"]
    assert not (tmp_path / "00000.npz").exists()


def test_live_staging_rejects_completion_count_mismatch(tmp_path: Path):
    (tmp_path / "complete.json").write_text('{"episode_count": 1}')

    with pytest.raises(RuntimeError, match="count mismatch"):
        list(preparation.iter_live_staged_episodes(tmp_path, child=object()))


def test_main_rejects_nonfixed_image_size(tmp_path: Path):
    with pytest.raises(SystemExit) as error:
        preparation.main(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-dir",
                str(tmp_path / "output"),
                "--image-size",
                "64",
            ]
        )
    assert error.value.code == 2


def test_decoder_child_hides_cuda_and_captures_native_diagnostics(monkeypatch, tmp_path: Path):
    captured = {}

    class Child:
        pass

    def popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["stdout"] = kwargs["stdout"]
        captured["stderr"] = kwargs["stderr"]
        return Child()

    monkeypatch.setattr(preparation.subprocess, "Popen", popen)
    child = preparation.start_rlds_decoder_subprocess(
        tmp_path, tmp_path / "staging", 10, 1, tmp_path / "run"
    )

    assert isinstance(child, Child)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert captured["stdout"] != preparation.subprocess.PIPE
    assert captured["stderr"] != preparation.subprocess.PIPE
    assert captured["stdout"].name.endswith("decoder.stdout.log")
    assert captured["stderr"].name.endswith("decoder.stderr.log")


def test_publish_replaces_prior_same_config_without_stale_episode_files(tmp_path: Path):
    canonical = tmp_path / "feature_cache"
    (canonical / "episodes").mkdir(parents=True)
    (canonical / "episodes" / "stale.npz").write_bytes(b"stale")
    candidate = tmp_path / "candidate"
    (candidate / "episodes").mkdir(parents=True)
    (candidate / "episodes" / "fresh.npz").write_bytes(b"fresh")
    (candidate / "dataset_manifest.json").write_text("{}")

    preparation.publish_feature_cache(candidate, canonical)

    assert (canonical / "episodes" / "fresh.npz").is_file()
    assert not (canonical / "episodes" / "stale.npz").exists()


def test_staging_producer_waits_for_consumer_ack_with_backlog_one(tmp_path: Path):
    results = []
    producer = threading.Thread(
        target=lambda: results.append(
            preparation.write_rlds_staging(
                [fake_episode("suite", "one", 0), fake_episode("suite", "two", 1)],
                tmp_path,
            )
        )
    )
    producer.start()

    first = tmp_path / "00000.npz"
    deadline = time.monotonic() + 2
    while not first.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert first.exists()
    time.sleep(0.05)  # Deliberately slow consumer: producer must remain blocked.
    assert len(list(tmp_path.glob("*.npz"))) == 1

    first.unlink()
    (tmp_path / "00000.ack").write_text("ack\n")
    second = tmp_path / "00001.npz"
    while not second.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert second.exists()
    assert len(list(tmp_path.glob("*.npz"))) == 1
    second.unlink()
    (tmp_path / "00001.ack").write_text("ack\n")
    producer.join(timeout=2)
    assert not producer.is_alive()
    assert results == [2]


def test_parent_encoder_error_terminates_and_reaps_decoder(monkeypatch, tmp_path: Path):
    events = []

    class Child:
        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            events.append(f"wait:{timeout}")
            return 0

        def kill(self):
            events.append("kill")

    child = Child()
    monkeypatch.setattr(
        preparation,
        "start_rlds_decoder_subprocess",
        lambda *_args: child,
    )
    monkeypatch.setattr(preparation, "validate_expanded_tfrecords", lambda _root: {})
    monkeypatch.setattr(preparation, "_source_commit", lambda _root: (None, False))
    monkeypatch.setattr(
        preparation,
        "build_imagenet_resnet18_encoder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("encoder boom")),
    )
    monkeypatch.setattr(preparation, "cleanup_staging_run", lambda _path: True)

    with pytest.raises(RuntimeError, match="encoder boom"):
        preparation.main(
            ["--dataset-root", str(tmp_path / "dataset"), "--output-dir", str(tmp_path / "out")]
        )
    assert events[0] == "terminate"
    assert events[1].startswith("wait:")


def test_staged_loader_retries_transient_permission_error(monkeypatch, tmp_path: Path):
    episode = fake_episode("suite", "instruction", 7)
    path = tmp_path / "00007.npz"
    np.savez_compressed(
        path,
        images=episode.images,
        states=episode.states,
        actions=episode.actions,
        instruction=np.asarray(episode.instruction),
        suite=np.asarray(episode.suite),
        episode_index=np.asarray(episode.episode_index),
    )
    real_load = np.load
    attempts = []

    def transient_sharing_lock(*args, **kwargs):
        attempts.append(args[0])
        if len(attempts) <= 2:
            raise PermissionError(13, "Permission denied", str(path))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(preparation.np, "load", transient_sharing_lock)

    loaded = preparation._staged_episode(path)

    assert loaded.episode_index == 7
    assert len(attempts) == 3


def test_segment_slices_after_deterministic_suite_selection():
    episodes = [
        fake_episode("libero_object_no_noops", "instruction", index)
        for index in range(12)
    ] + [
        fake_episode("libero_goal_no_noops", "instruction", index)
        for index in range(12)
    ]

    selected = list(
        preparation.iter_selected_segment(
            episodes,
            suite="libero_object_no_noops",
            max_per_instruction=10,
            selected_start=8,
            selected_count=5,
        )
    )

    assert [episode.episode_index for episode in selected] == [8, 9]


def test_feature_cache_manifest_records_segment_metadata(tmp_path: Path):
    episode = fake_episode("libero_object_no_noops", "instruction", 3)

    manifest = write_feature_cache(
        [episode],
        output_dir=tmp_path,
        encoder=lambda images: np.ones((len(images), 4), dtype=np.float32),
        source_commit="commit",
        config={"encoder": "synthetic", "feature_dim": 4},
        segment={
            "suite": "libero_object_no_noops",
            "selected_start": 20,
            "selected_count": 1,
        },
    )

    assert manifest["segment"] == {
        "suite": "libero_object_no_noops",
        "selected_start": 20,
        "selected_count": 1,
    }


@pytest.mark.parametrize(
    "different_segment",
    [
        {
            "suite": "libero_goal_no_noops",
            "selected_start": 20,
            "selected_count": 1,
        },
        {
            "suite": "libero_object_no_noops",
            "selected_start": 40,
            "selected_count": 1,
        },
        {
            "suite": "libero_object_no_noops",
            "selected_start": 20,
            "selected_count": 2,
        },
    ],
)
def test_segment_cache_overwrite_rejects_different_selection(
    tmp_path: Path, different_segment: dict
):
    episode = fake_episode("libero_object_no_noops", "instruction", 3)
    config = {"encoder": "synthetic", "feature_dim": 4}
    original_segment = {
        "suite": "libero_object_no_noops",
        "selected_start": 20,
        "selected_count": 1,
    }
    write_feature_cache(
        [episode],
        output_dir=tmp_path,
        encoder=lambda images: np.ones((len(images), 4), dtype=np.float32),
        source_commit="commit",
        config=config,
        segment=original_segment,
    )
    # Identical selection remains safely resumable.
    write_feature_cache(
        [episode],
        output_dir=tmp_path,
        encoder=lambda images: np.ones((len(images), 4), dtype=np.float32),
        source_commit="commit",
        config=config,
        segment=original_segment,
    )

    with pytest.raises(RuntimeError, match="segment"):
        write_feature_cache(
            [episode],
            output_dir=tmp_path,
            encoder=lambda images: np.ones((len(images), 4), dtype=np.float32),
            source_commit="commit",
            config=config,
            segment=different_segment,
        )


def test_monolithic_cache_overwrite_remains_compatible(tmp_path: Path):
    episode = fake_episode("suite", "instruction", 0)
    arguments = {
        "output_dir": tmp_path,
        "encoder": lambda images: np.ones((len(images), 4), dtype=np.float32),
        "source_commit": "commit",
        "config": {"encoder": "synthetic", "feature_dim": 4},
    }

    write_feature_cache([episode], **arguments)
    manifest = write_feature_cache([episode], **arguments)

    assert "segment" not in manifest


def test_live_staging_rejects_mismatched_segment_sentinel(tmp_path: Path):
    requested = {
        "suite": "libero_object_no_noops",
        "selected_start": 0,
        "selected_count": 2,
    }
    (tmp_path / "complete.json").write_text(
        json.dumps(
            {
                "episode_count": 0,
                "segment": {**requested, "selected_start": 20},
            }
        )
    )

    with pytest.raises(RuntimeError, match="segment sentinel mismatch"):
        list(
            preparation.iter_live_staged_episodes(
                tmp_path, child=object(), expected_segment=requested
            )
        )


def test_live_staging_accepts_identical_segment_sentinel(tmp_path: Path):
    requested = {
        "suite": "libero_object_no_noops",
        "selected_start": 0,
        "selected_count": 2,
    }
    (tmp_path / "complete.json").write_text(
        json.dumps({"episode_count": 0, "segment": requested})
    )

    assert list(
        preparation.iter_live_staged_episodes(
            tmp_path, child=object(), expected_segment=requested
        )
    ) == []
