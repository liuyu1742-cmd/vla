"""Regression tests for resumable BEHAVIOR-1K downloads."""

import importlib.util
import urllib.error
from pathlib import Path


def _load_downloader():
    path = Path(__file__).parents[1] / "tools" / "download_behavior1k_subset.py"
    spec = importlib.util.spec_from_file_location("behavior_downloader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_download_survives_four_transient_tls_failures(tmp_path, monkeypatch):
    downloader = _load_downloader()
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size):
            return b"" if hasattr(self, "done") else setattr(self, "done", True) or b"ok"

    def flaky_urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= 4:
            raise urllib.error.URLError("transient TLS EOF")
        return Response()

    monkeypatch.setattr(downloader.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(downloader.time, "sleep", lambda _seconds: None)

    destination = tmp_path / "episode.parquet"
    downloader.download("https://example.invalid/episode", destination)

    assert attempts == 5
    assert destination.read_bytes() == b"ok"
