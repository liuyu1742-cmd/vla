"""Small newline-delimited JSON protocol shared by both Conda environments."""

from __future__ import annotations

import json
from typing import Any


def encode_message(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def decode_message(raw: bytes) -> dict[str, Any]:
    if not raw.strip():
        raise ValueError("empty TCP message")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("TCP message must be a JSON object")
    return parsed
