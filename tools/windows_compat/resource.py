"""Minimal Windows compatibility shim for TensorFlow Datasets imports."""

RLIMIT_NOFILE = 7


def getrlimit(_resource: int) -> tuple[int, int]:
    """Return a stable no-op descriptor limit on Windows."""
    return (2048, 2048)


def setrlimit(_resource: int, _limits: tuple[int, int]) -> None:
    """Accept TFDS limit changes; Windows has no POSIX rlimit API."""

