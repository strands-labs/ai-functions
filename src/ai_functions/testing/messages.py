"""Message-history comparison utilities for harness-based tests.

Used to compare ``agent.messages`` against ``reconstruct_messages(events)``. The
normalizer strips the per-message telemetry keys Strands attaches to live messages
(``metadata`` and ``tracking_id``) — no other shape changes — so tests that fail
indicate genuine divergence between the live agent history and the event-log
reconstruction.
"""

from __future__ import annotations

import copy
import difflib
import json

from strands.types.content import Message, Messages

# Per-message keys Strands populates as telemetry on the live ``agent.messages``.
# Both are assigned by the agent automatically and stripped before model calls, so
# they are not part of the conversation the model observes on replay and must be
# removed before comparing against ``reconstruct_messages`` output:
#   - ``metadata``: usage/metrics attached after ``AfterModelCallEvent``.
#   - ``tracking_id``: durable per-message UUID added in strands-agents 1.47
#     (see ``strands.types.content.Message`` / ``_ensure_tracking_id``).
_TELEMETRY_KEYS = ("metadata", "tracking_id")


def normalize_messages(messages: Messages) -> Messages:
    """Return a canonical form suitable for equality comparison.

    The only normalization applied is stripping the per-message telemetry keys
    Strands attaches to the live ``agent.messages`` — ``metadata`` (usage/metrics,
    set after ``AfterModelCallEvent``) and ``tracking_id`` (a durable per-message
    UUID added in strands-agents 1.47). Both are populated by the agent automatically
    and discarded before the next model call, so neither is part of the conversation
    the model observes on replay.

    No other normalization is applied. In particular, consecutive same-role messages
    are NOT merged — Strands does not merge them either. The round-trip property
    ``agent.messages == reconstruct_messages(events)`` is exactly that: equality
    after stripping the telemetry keys.

    Args:
        messages: A Strands-format message list.

    Returns:
        A deep copy with the telemetry keys removed from every message.

    Ensures:
        The returned list is a deep copy; callers may mutate freely.
    """
    result: list[Message] = []
    for msg in copy.deepcopy(list(messages)):
        # These keys are runtime telemetry Strands stashes on each message; strip
        # them so we compare only what the model actually sees on replay.
        for key in _TELEMETRY_KEYS:
            msg.pop(key, None)  # type: ignore[misc]
        result.append(msg)
    return result


def assert_messages_equivalent(actual: Messages, expected: Messages) -> None:
    """Assert two message histories are equal after ``normalize_messages``.

    Args:
        actual: The observed message history.
        expected: The reference message history.

    Raises:
        AssertionError: The normalized histories differ; the message
            includes a unified diff of the JSON-serialized forms.
    """
    norm_actual = normalize_messages(actual)
    norm_expected = normalize_messages(expected)
    if norm_actual == norm_expected:
        return
    diff = "\n".join(
        difflib.unified_diff(
            json.dumps(norm_expected, indent=2, default=str).splitlines(),
            json.dumps(norm_actual, indent=2, default=str).splitlines(),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
    raise AssertionError(f"message histories differ after normalization:\n{diff}")
