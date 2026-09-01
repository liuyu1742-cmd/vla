"""JSON-safe entry point for the DAgger collector.

The first real rollout exposed NumPy scalar values in the detailed report.
This entry point recursively converts those values before delegating to the
tested collector persistence contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools import collect_water_cup_dagger as collector


_save_episode = collector.save_episode


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_episode(*args: Any, **kwargs: Any):
    if "report" not in kwargs:
        raise TypeError("save_episode requires report")
    kwargs["report"] = json_safe(kwargs["report"])
    return _save_episode(*args, **kwargs)


def main() -> int:
    collector.save_episode = save_episode
    return collector.main()


if __name__ == "__main__":
    raise SystemExit(main())
