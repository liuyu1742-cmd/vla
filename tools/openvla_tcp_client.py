"""Request one OpenVLA action from the persistent local inference service."""

from __future__ import annotations

import socket
from typing import Any

from tools.openvla_tcp_protocol import decode_message, encode_message


def predict(host: str, port: int, request: dict[str, Any], timeout: float = 120.0) -> list[float]:
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(encode_message(request))
        response = b""
        while not response.endswith(b"\n"):
            packet = connection.recv(65536)
            if not packet:
                raise ConnectionError("OpenVLA service closed the connection")
            response += packet
    payload = decode_message(response)
    if "error" in payload:
        raise RuntimeError(f"OpenVLA service error: {payload['error']}")
    action = payload.get("raw_action")
    if not isinstance(action, list) or len(action) != 7:
        raise ValueError("OpenVLA service returned an invalid action")
    return [float(value) for value in action]
