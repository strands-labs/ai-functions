"""Kiro Agent thread — run a Kiro session, visualize events, print the result.

``KiroAgent`` is a ``Spawnable`` that drives a ``kiro-cli acp`` subprocess over
the Agent Client Protocol (ACP). The ACP agent owns the conversation transcript;
ai_functions observes the ``session/update`` stream and re-emits each element as
a ai_functions event for observability.

This example:

1. Spawns a ``KiroAgent`` thread on a ``LocalWorker``.
2. Subscribes to events on the coordinator and pretty-prints them as
   they arrive.
3. Sends a task as a prompt and awaits the result.
4. Tears the thread down so the ``kiro-cli acp`` subprocess exits cleanly.

Prerequisites:
    - The ``kiro`` extra: ``pip install 'strands-ai-functions[kiro]'``.
    - The ``kiro-cli`` binary on your PATH (or pass ``KiroAgent(executable=...)``
      with its full path). Install it from
      https://kiro.dev/docs/cli/installation/.

Run it from the repository root so the agent's file tools see this project's
files (the ACP session uses the process working directory by default).
"""

from __future__ import annotations

import asyncio

from strands.types.tools import ToolResultContent

from ai_functions.kiro import KiroAgent
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types import (
    CompletedEvent,
    Event,
    FailedEvent,
    MessageAssistantCompleteEvent,
    MessageUserEvent,
    StartedEvent,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def log_event(event: Event) -> None:
    """Pretty-print one ai_functions event to the console."""
    match event:
        case StartedEvent(thread_name=name):
            print(f"  ▶ {name or 'thread'} started")
        case CompletedEvent(thread_name=name):
            print(f"  ✓ {name or 'thread'} completed")
        case FailedEvent(thread_name=name, error=error):
            print(f"  ✗ {name or 'thread'} failed: {error}")
        case MessageUserEvent(text=text):
            preview = text if len(text) <= 120 else text[:117] + "..."
            print(f"  ▷ user: {preview}")
        case MessageAssistantCompleteEvent(content=content):
            texts: list[str] = []
            for block in content:
                text = block.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
            if texts:
                joined = "\n".join(texts)
                preview = joined if len(joined) <= 240 else joined[:237] + "..."
                print(f"  ◁ assistant: {preview}")
            else:
                kinds = [key for block in content for key in block]
                summary = ", ".join(kinds) if kinds else "empty"
                print(f"  ◁ assistant: <{summary}>")
        case ToolCallEvent(tool_name=tool_name, arguments=arguments):
            print(f"    ⚙ tool call: {tool_name}({_format_args(arguments)})")
        case ToolResultEvent(status=status, content=content):
            preview = _format_tool_result(content) or "<empty>"
            error = "[error] " if status == "error" else ""
            print(f"    ⚙ tool result: {error}{preview}")
        case TokenUsageEvent(token_usage=usage):
            total_input = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
            total = total_input + usage.output_tokens
            print(
                f"  Σ tokens: in={usage.input_tokens} "
                f"cache_r={usage.cache_read_tokens} cache_w={usage.cache_write_tokens} "
                f"out={usage.output_tokens} (total={total})",
            )
        case _:
            pass


def _format_args(arguments: dict[str, object]) -> str:
    """Render tool-call arguments compactly for log output."""
    parts: list[str] = []
    for key, value in arguments.items():
        rendered = repr(value)
        if len(rendered) > 60:
            rendered = rendered[:57] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _format_tool_result(content: list[ToolResultContent]) -> str:
    """Render a tool result's content as a compact one-line preview."""
    texts: list[str] = []
    for block in content:
        text = block.get("text")
        if isinstance(text, str):
            texts.append(text)
    if texts:
        joined = " ".join(" ".join(t.split()) for t in texts)
        return joined if len(joined) <= 120 else joined[:117] + "..."
    kinds = [key for block in content for key in block]
    return f"<{', '.join(kinds)}>" if kinds else ""


async def main() -> None:
    # ``auto_approve`` lets the ACP agent run its tools non-interactively.
    # Don't bypass approvals outside trusted environments.
    template = KiroAgent(auto_approve=True, name="kiro_agent_example")

    coord = InMemoryCoordinator()
    coord.on(log_event)

    worker = LocalWorker(coord)
    await worker.register()

    handle = await worker.spawn_locally(template, thread_name="kiro_agent_example")
    try:
        result = await handle.run(
            "List the three files in the current directory whose names end in .md, "
            "and in one short sentence say what the project is about based on README.md.",
        )
        print()
        print("─" * 60)
        print("RESULT")
        print("─" * 60)
        print(result)
    finally:
        await handle.terminate_now()
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
