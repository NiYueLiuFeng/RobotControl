"""Minimal JSON-over-WebSocket client for a remote policy server."""

from __future__ import annotations

import json
from typing import Any, Callable

from ur5e_control.inference.vla_client import connect_websocket

ConnectionFactory = Callable[[str, str | None, float], Any]


class JsonWebSocketClient:
    """Synchronous client: one JSON payload per round trip."""

    def __init__(
        self,
        server_url: str,
        api_key: str | None = None,
        timeout_s: float = 5.0,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self.server_url = server_url
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        connect = connection_factory or connect_websocket
        self._connection = connect(server_url, api_key, self.timeout_s)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._connection.send(json.dumps(payload))
        response = json.loads(self._connection.recv())
        if not isinstance(response, dict):
            raise RuntimeError(
                f"policy server returned {type(response).__name__}, expected an object"
            )
        if response.get("ok") is False:
            raise RuntimeError(str(response.get("error", "policy server error")))
        return response

    def close(self) -> None:
        close = getattr(self._connection, "close", None)
        if callable(close):
            close()
