"""Serve a balanced formal adapter with the same canonical English policy prompt."""

from __future__ import annotations

import tools.openvla_formal_skill_tcp_server as _server
from tools.formal_skill_policy_prompt import build_policy_instruction


_server.build_instruction = build_policy_instruction
conditioned_instruction = _server.conditioned_instruction
Predictor = _server.Predictor
Handler = _server.Handler
Server = _server.Server
main = _server.main


if __name__ == "__main__":
    raise SystemExit(main())
