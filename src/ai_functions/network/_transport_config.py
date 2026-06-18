"""Shared ``websockets`` transport tuning for the client and endpoint.

Both peers — :class:`~ai_functions.network.client.CoordinatorClient` and
:class:`~ai_functions.network.endpoint.CoordinatorEndpoint` — must construct their
WebSocket with the *same* size and keepalive limits; a ``max_size`` mismatch
between the two ends would itself trigger a close, so the values live in one
place rather than being duplicated at each call site.

This module is private (no public stub) — it is an implementation detail of
the transport layer, not part of the package contract.
"""

from __future__ import annotations

# ``max_size`` defaults to 1 MiB in ``websockets``. Cycle results travel
# cloudpickled + base64 (≈ +33% overhead) and large event payloads (assistant
# messages, thinking-token events) are broadcast as single frames. A coding
# agent routinely crosses 1 MiB, and the receiver closes the connection with
# code 1009 ("message too big") when it does. The cap is raised generously
# rather than removed (``None``) so a runaway frame still fails loudly instead
# of exhausting memory.
MAX_MESSAGE_BYTES: int = 64 * 1024 * 1024

# ``ping_timeout`` defaults to 20s. The keepalive pong is answered by an
# on-loop task; any on-loop stall longer than the timeout (e.g. a large
# synchronous cloudpickle of a cycle result) starves the pong and the peer
# declares the connection dead with code 1011. Keepalive stays enabled — so a
# genuinely dead TCP peer is still detected — but the timeout is widened to
# tolerate brief loop-blocking.
PING_INTERVAL_SECONDS: float = 20.0
PING_TIMEOUT_SECONDS: float = 60.0
