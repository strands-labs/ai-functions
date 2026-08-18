"""The shared coordinator-tool core, and the schema parity it exists to enforce.

``list_threads`` / ``send_message`` are published to agents by more than one
runtime adapter. These tests pin the two properties that made a shared core
necessary: every adapter offers the same argument schema and the same
descriptions, and the body's defensive validation still answers a caller that
reaches it with an unvalidated argument.
"""

import importlib

import pytest
from strands.tools.decorator import tool as strands_tool

from ai_functions.runtime.coordinator_tools_core import (
    LIST_THREADS_DESCRIPTION,
    SEND_MESSAGE_DESCRIPTION,
    SEND_MESSAGE_INPUT_SCHEMA,
    SEND_MESSAGE_MODES,
    ListThreadsResult,
    SendMessageArgs,
    SendMessageMode,
    ThreadSummary,
    send_message,
)

# NOTE: no ``from __future__ import annotations`` in this module. Strands resolves
# the wrapper's annotations to derive the tool schema, so ``SendMessageMode`` has
# to be a real type at decoration time rather than a deferred string.


@strands_tool(name="send_message", description=SEND_MESSAGE_DESCRIPTION)
async def _strands_send_message(
    thread_id: str,
    message: str,
    mode: SendMessageMode = "wait",
) -> str:
    """Stand-in with the same signature the Strands binding uses."""
    return ""


def _strands_send_message_schema() -> dict[str, object]:
    """Return the input schema Strands derives from the binding's signature."""
    schema = _strands_send_message.tool_spec["inputSchema"]
    return schema["json"] if "json" in schema else schema


def _claude_send_message_schema() -> dict[str, object]:
    """Build the Claude SDK binding's ``send_message`` and return its input schema."""
    claude_sdk = pytest.importorskip("claude_agent_sdk")

    @claude_sdk.tool("send_message", SEND_MESSAGE_DESCRIPTION, SEND_MESSAGE_INPUT_SCHEMA)
    async def _send_message(args: dict[str, object]) -> dict[str, object]:
        return {}

    return _send_message.input_schema


def test_send_message_mode_is_an_enum_with_a_default() -> None:
    """``mode`` is constrained to the legal values and is not required."""
    props = SEND_MESSAGE_INPUT_SCHEMA["properties"]
    assert props["mode"]["enum"] == list(SEND_MESSAGE_MODES)
    assert props["mode"]["default"] == "wait"
    # Only the two genuinely required arguments are required.
    assert SEND_MESSAGE_INPUT_SCHEMA["required"] == ["thread_id", "message"]


def test_send_message_schema_is_identical_across_adapters() -> None:
    """Both bindings publish the same argument constraints.

    The Strands binding derives its schema from the wrapper signature and the
    Claude binding hands ``SEND_MESSAGE_INPUT_SCHEMA`` to its SDK. Different
    derivations, one declaration — so they must agree on the parts an agent can
    observe.
    """
    strands = _strands_send_message_schema()
    claude = _claude_send_message_schema()

    strands_props = strands["properties"]
    claude_props = claude["properties"]

    assert set(strands_props) == set(claude_props) == {"thread_id", "message", "mode"}
    assert strands["required"] == claude["required"] == ["thread_id", "message"]
    for name in ("thread_id", "message", "mode"):
        assert strands_props[name]["type"] == claude_props[name]["type"]
    assert strands_props["mode"]["enum"] == claude_props["mode"]["enum"] == list(SEND_MESSAGE_MODES)
    assert strands_props["mode"]["default"] == claude_props["mode"]["default"] == "wait"


def test_tool_descriptions_are_shared_constants() -> None:
    """Descriptions come from the core, so no adapter can describe a tool its own way."""
    pytest.importorskip("claude_agent_sdk")

    # import_module, not ``from ai_functions.claude_code import coordinator_tools``:
    # the package re-exports a *function* of that name, which would shadow the module.
    strands_tools = importlib.import_module("ai_functions.ai_thread.tools")
    claude_tools = importlib.import_module("ai_functions.claude_code.coordinator_tools")

    assert strands_tools.LIST_THREADS_DESCRIPTION is LIST_THREADS_DESCRIPTION
    assert strands_tools.SEND_MESSAGE_DESCRIPTION is SEND_MESSAGE_DESCRIPTION
    assert claude_tools.LIST_THREADS_DESCRIPTION is LIST_THREADS_DESCRIPTION
    assert claude_tools.SEND_MESSAGE_DESCRIPTION is SEND_MESSAGE_DESCRIPTION


def test_every_adapter_dispatches_through_the_guarded_body() -> None:
    """Both adapters call the core bodies, so both inherit the deadlock guard.

    ``send_message(mode="wait")`` refuses a wait that would close a cycle in the
    graph of in-flight waits. That graph lives in the core and is consulted by the
    core body, so an adapter can only get the protection by calling it — and only
    one shared graph means a cycle spanning two runtimes is caught as well.
    """
    pytest.importorskip("claude_agent_sdk")

    strands_tools = importlib.import_module("ai_functions.ai_thread.tools")
    claude_tools = importlib.import_module("ai_functions.claude_code.coordinator_tools")
    core = importlib.import_module("ai_functions.runtime.coordinator_tools_core")

    for adapter in (strands_tools, claude_tools):
        assert adapter._send_message is core.send_message
        assert adapter._list_threads is core.list_threads
    # The waits-for graph has exactly one home.
    assert not hasattr(strands_tools, "_wait_edges")
    assert not hasattr(claude_tools, "_wait_edges")
    assert hasattr(core, "_wait_edges")


async def test_send_message_to_self_is_refused_before_any_lookup() -> None:
    """A self-send is rejected without touching the coordinator."""
    result = await send_message(
        coordinator=None,  # pyright: ignore[reportArgumentType]  # never reached
        self_thread_id="t-self",
        thread_id="t-self",
        message="hi",
        mode="wait",
    )
    assert result == "error: cannot send_message to self"


async def test_send_message_unknown_mode_names_the_legal_modes() -> None:
    """The body validates ``mode`` even when no schema layer did.

    Both shipped adapters validate against the schema before dispatch, so this
    guard is unreachable through them. It is the safety net for a transport that
    does not validate, and it must answer with text a model can act on rather
    than raise.
    """

    from ai_functions.types import InputShape

    class _Info:
        input_shape = InputShape.STR_PROMPT

    class _Handle:
        def run(self, message: str) -> object:
            raise AssertionError("an unknown mode must not run a cycle on the peer")

    class _StubCoordinator:
        """Minimal coordinator: resolves one str_prompt peer, hands back a stub handle."""

        async def get_thread_info(self, thread_id: object) -> object:
            return _Info()

        def get_handle(self, thread_id: object) -> object:
            return _Handle()

    result = await send_message(
        coordinator=_StubCoordinator(),  # pyright: ignore[reportArgumentType]  # structural stub
        self_thread_id="t-self",
        thread_id="t-peer",
        message="hi",
        mode="async",
    )
    assert result.startswith("error: unknown mode")
    for mode in SEND_MESSAGE_MODES:
        assert mode in result


def test_thread_summary_round_trips_as_json() -> None:
    """``list_threads`` results serialise to the documented wire shape."""
    result = ListThreadsResult(
        threads=[
            ThreadSummary(
                thread_id="t-1",
                thread_name="alice",
                status="idle",
                input_shape="str_prompt",
                parent_id=None,
                is_self=True,
            ),
        ],
    )
    payload = ListThreadsResult.model_validate_json(result.model_dump_json())
    assert payload.threads[0].thread_name == "alice"
    assert payload.threads[0].is_self is True
    assert payload.threads[0].parent_id is None


def test_send_message_args_defaults_to_wait() -> None:
    """Omitting ``mode`` yields the documented default."""
    assert SendMessageArgs(thread_id="t-1", message="hi").mode == "wait"
