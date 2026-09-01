"""Action-space contract shared by water-cup training and OpenVLA serving."""

from __future__ import annotations

from typing import Any


ACTION_NORM_KEY = "water_cup_robocasa"
ACTION_DIMENSION = 7


def install_water_cup_action_stats(model: Any) -> None:
    """Make OpenVLA decode the normalized actions used in the local dataset.

    The expert dataset already stores RoboCasa arm actions in the OpenVLA
    normalized range [-1, 1].  Supplying an identity quantile transform avoids
    applying BridgeData V2's unrelated physical-unit statistics at inference.
    """
    model.norm_stats = {
        ACTION_NORM_KEY: {
            "action": {
                "q01": [-1.0] * ACTION_DIMENSION,
                "q99": [1.0] * ACTION_DIMENSION,
                "mask": [True] * ACTION_DIMENSION,
            }
        }
    }
