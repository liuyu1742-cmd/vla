"""Request one 8x7 continuous action chunk from the local OFT service."""

from __future__ import annotations

import socket
from typing import Any

import numpy as np

from tools.openvla_tcp_protocol import decode_message, encode_message


def predict_chunk(
    host: str, port: int, request: dict[str, Any], timeout: float = 120.0
) -> np.ndarray:
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(encode_message(request))
        response = b""
        while not response.endswith(b"\n"):
            packet = connection.recv(65536)
            if not packet:
                raise ConnectionError("OFT service closed the connection")
            response += packet
    payload = decode_message(response)
    if "error" in payload:
        raise RuntimeError(f"OFT service error: {payload['error']}")
    from tools.robocasa_oft_rollout import validate_action_chunk

    return validate_action_chunk(payload.get("action_chunk"))
