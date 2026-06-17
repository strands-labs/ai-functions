"""Message-history comparison utilities for harness-based tests.

Used to compare ``agent.messages`` against ``reconstruct_messages(events)``. The
normalizer strips only the ``metadata`` key that Strands attaches to assistant
messages as post-hoc telemetry — no other shape changes — so tests that fail indicate
genuine divergence between the live agent history and the event-log reconstruction."""

from strands.types.content import Messages


def normalize_messages(messages: Messages) -> Messages:
    """Return a canonical form suitable for equality comparison.

    The only normalization applied is stripping the ``metadata`` key that Strands
    attaches to assistant messages in ``event_loop.event_loop``
    (``message["metadata"] = {"usage": ..., "metrics": ...}``). That field is post-hoc
    telemetry: populated after ``AfterModelCallEvent``, not part of the conversation
    the model observes on replay, and discarded by Strands' own
    ``_normalize_messages`` before the next model call.

    No other normalization is applied. In particular, consecutive same-role messages
    are NOT merged — Strands does not merge them either. The round-trip property
    ``agent.messages == reconstruct_messages(events)`` is exactly that: equality
    after stripping ``metadata``.

    Args:
        messages: A Strands-format message list.

    Returns:
        A deep copy with ``metadata`` keys removed from every message.

    Ensures:
        The returned list is a deep copy; callers may mutate freely.
    """
    ...


def assert_messages_equivalent(actual: Messages, expected: Messages) -> None:
    """Assert two message histories are equal after ``normalize_messages``.

    Args:
        actual: The observed message history.
        expected: The reference message history.

    Raises:
        AssertionError: The normalized histories differ; the message
            includes a unified diff of the JSON-serialized forms.
    """
    ...
