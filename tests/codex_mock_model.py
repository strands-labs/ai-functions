"""Hermetic model backend for Codex integration tests.

The ``codex app-server`` binary is real (it ships with the ``openai-codex``
SDK); only the *model* is mocked: a local HTTP server impersonates the OpenAI
Responses API, and an isolated ``CODEX_HOME`` config routes the app-server's
model provider at it. Tests script the model's SSE outputs and read back the
exact requests the app-server sent, so turn structure, notifications, and
token accounting are all produced by the real protocol implementation.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openai_codex.client import CodexConfig

Json = dict[str, Any]


def sse_body(events: list[Json]) -> str:
    """Frame Responses-API event objects as one SSE body."""
    chunks = [f"event: {event['type']}\ndata: {json.dumps(event)}\n" for event in events]
    return "\n".join(chunks) + "\n"


def assistant_turn(
    text: str,
    *,
    response_id: str = "resp-1",
    input_tokens: int = 100,
    cached_input_tokens: int = 40,
    output_tokens: int = 7,
) -> list[Json]:
    """One complete assistant-message model response with explicit usage."""
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "id": f"msg-{response_id}",
                "content": [{"type": "output_text", "text": text}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "usage": {
                    "input_tokens": input_tokens,
                    "input_tokens_details": {"cached_tokens": cached_input_tokens},
                    "output_tokens": output_tokens,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        },
    ]


class MockModelServer:
    """Threaded HTTP server standing in for the Responses API."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self.requests: list[Json] = []
        self._lock = threading.Lock()
        self._httpd = _Server(("127.0.0.1", 0), _Handler, self)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> MockModelServer:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def enqueue(self, events: list[Json]) -> None:
        """Queue the SSE response for the next model call."""
        self._queue.put(sse_body(events))

    def user_texts(self, request_index: int) -> list[str]:
        """All user-role input_text strings in the recorded request."""
        texts: list[str] = []
        for item in self.requests[request_index].get("input", []):
            if item.get("type") != "message" or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
                continue
            for span in content or []:
                if isinstance(span, dict) and span.get("type") == "input_text":
                    texts.append(str(span.get("text", "")))
        return texts

    def _record(self, body: bytes) -> None:
        with self._lock:
            self.requests.append(json.loads(body.decode("utf-8")))

    def _next(self) -> str:
        return self._queue.get_nowait()


class _Server(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], handler: type[BaseHTTPRequestHandler], mock: MockModelServer) -> None:
        super().__init__(addr, handler)
        self.mock = mock


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            body = json.dumps(
                {"object": "list", "data": [{"id": "mock-model", "object": "model", "created": 0, "owned_by": "t"}]},
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        self.server.mock._record(self.rfile.read(length))  # pyright: ignore[reportPrivateUsage]
        if not self.path.endswith("/responses"):
            self.send_error(404)
            return
        try:
            body = self.server.mock._next()  # pyright: ignore[reportPrivateUsage]
        except queue.Empty:
            self.send_error(500, "no queued model response")
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body.encode())
        self.wfile.flush()


def codex_test_config(tmp_path: Path, model_url: str) -> CodexConfig:
    """An isolated CodexConfig whose model provider is the mock server."""
    home = tmp_path / "codex-home"
    workspace = tmp_path / "workspace"
    home.mkdir(exist_ok=True)
    workspace.mkdir(exist_ok=True)
    (home / "config.toml").write_text(
        f"""
model = "mock-model"
approval_policy = "never"
sandbox_mode = "read-only"

model_provider = "mock_provider"

[model_providers.mock_provider]
name = "Mock provider for ai_functions tests"
base_url = "{model_url}/v1"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
""".lstrip(),
    )
    return CodexConfig(
        cwd=str(workspace),
        env={
            "CODEX_HOME": str(home),
            "CODEX_APP_SERVER_DISABLE_MANAGED_CONFIG": "1",
            "RUST_LOG": "warn",
        },
    )
