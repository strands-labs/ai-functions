"""Render parameters and message traces into text/XML for the backward prompt."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from strands.types.content import Message

from ..types.graph import Node
from ._formatting import to_yaml, truncate


def _yaml_safe_value(value: Any) -> Any:  # pyright: ignore[reportExplicitAny]
    """Convert a value to a YAML-safe form (custom objects become str())."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _yaml_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_yaml_safe_value(item) for item in value]
    return str(value)


def render_inputs(nodes: list[Node]) -> str:
    result: dict[str, Any] = {}  # pyright: ignore[reportExplicitAny]
    for node in nodes:
        node_type = node.__class__.__name__.lower()
        result[node.node_id] = {
            "type": node_type,
            "description": getattr(node, "description", ""),
            "value": _yaml_safe_value(node.value),
        }
        if not result[node.node_id]["description"]:
            del result[node.node_id]["description"]
    return to_yaml(result)


def _convert_id(tool_id: str | None, tool_id_to_tool_result_id: dict[str, str]) -> tuple[str | None, str]:
    call_type = "function" if tool_id in tool_id_to_tool_result_id else "tool"
    tool_id = tool_id_to_tool_result_id.get(tool_id, tool_id) if tool_id is not None else tool_id
    return tool_id, call_type


def _collect_tool_results(
    messages: list[Message],
    maybe_truncate: Callable[[Any], Any] = truncate,  # pyright: ignore[reportExplicitAny]
) -> dict[str, dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Build a map from toolUseId to its result data across all messages."""
    results_map: dict[str, dict[str, Any]] = {}  # pyright: ignore[reportExplicitAny]
    for message in messages:
        for block in message.get("content", []):
            if not isinstance(block, dict):
                continue
            if tool_result := block.get("toolResult", None):
                use_id = tool_result.get("toolUseId")
                if use_id is None:
                    continue
                results: list[Any] = []  # pyright: ignore[reportExplicitAny]
                for trc in tool_result.get("content", []):
                    if text := trc.get("text"):
                        results.append(maybe_truncate(text))
                    elif json_result := trc.get("json"):
                        results.append(maybe_truncate(json_result))
                results_map[use_id] = {"status": tool_result.get("status"), "results": results}
    return results_map


def render_messages(
    messages: list[Message] | None,
    tool_id_to_tool_result_id: dict[str, str] | None = None,
    should_truncate: bool = True,
) -> str:
    """Format agent messages into a readable conversation trace string."""
    if not messages:
        return ""

    if tool_id_to_tool_result_id is None:
        tool_id_to_tool_result_id = {}

    maybe_truncate: Callable[[Any], Any] = truncate if should_truncate else (lambda x: x)  # pyright: ignore[reportExplicitAny]

    tool_results_map = _collect_tool_results(messages, maybe_truncate)

    message_list: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
    for i, message in enumerate(messages, 1):
        msg_dict: dict[str, Any] = {"role": message.get("role", "unknown").upper(), "content": []}  # pyright: ignore[reportExplicitAny]
        for block in message.get("content", []):
            if not isinstance(block, dict):
                continue
            if reasoning_text := block.get("reasoningContent", {}).get("text"):
                msg_dict["content"].append({"reasoning": reasoning_text})
            if text := block.get("text", ""):
                msg_dict["content"].append({"text": text})
            if tool_use := block.get("toolUse", None):
                original_id = tool_use.get("toolUseId")
                _, call_type = _convert_id(original_id, tool_id_to_tool_result_id)
                entry: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
                    "type": f"{call_type}_call",
                    "name": tool_use.get("name"),
                    "inputs": maybe_truncate(tool_use.get("input", {})),
                }
                if call_type == "function":
                    entry["id"] = tool_id_to_tool_result_id[original_id]
                if original_id in tool_results_map:
                    result_data = tool_results_map[original_id]
                    entry["status"] = result_data["status"]
                    entry["output"] = result_data["results"]
                msg_dict["content"].append(entry)
        if msg_dict["content"]:
            message_list.append({f"message_{i}": msg_dict})

    return to_xml(message_list)


def _format_tool_inputs(inputs: Any) -> str:  # pyright: ignore[reportExplicitAny]
    """Try to pretty-print JSON tool inputs; fall back to raw string."""
    try:
        parsed = json.loads(inputs) if isinstance(inputs, str) else inputs
        if isinstance(parsed, dict):
            return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    return str(inputs)


def to_xml(message_list: list[dict[str, Any]]) -> str:  # pyright: ignore[reportExplicitAny]
    """Convert the message_list produced by render_messages into an XML string.

    Content is NOT XML-escaped so embedded XML/HTML tags are preserved verbatim.
    """
    parts: list[str] = []
    for entry in message_list:
        key = next(iter(entry))
        msg = entry[key]
        number = key.split("_", 1)[1]
        role = msg["role"]

        parts.append(f'<message step="{number}" role="{role}">')

        for block in msg["content"]:
            if isinstance(block, dict) and "reasoning" in block:
                parts.append(f"<reasoning>{block['reasoning']}</reasoning>")
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif isinstance(block, dict) and "type" in block:
                name = block.get("name", "")
                inputs_raw = block.get("inputs", "")
                status = block.get("status", "")
                output_parts = block.get("output", [])
                output_text = "\n".join(str(o) for o in output_parts)

                if name == "python_executor":
                    try:
                        parsed = json.loads(inputs_raw) if isinstance(inputs_raw, str) else inputs_raw
                        code = parsed.get("code", inputs_raw) if isinstance(parsed, dict) else str(inputs_raw)
                    except (json.JSONDecodeError, TypeError):
                        code = str(inputs_raw)
                    parts.append(f"<execute_code>\n{code}\n</execute_code>")
                    if output_text:
                        parts.append(f"<result>\n{output_text}\n</result>")
                else:
                    formatted_inputs = _format_tool_inputs(inputs_raw)
                    attrs = f' name="{name}"'
                    if status:
                        attrs += f' status="{status}"'
                    inner = f"<inputs>{formatted_inputs}</inputs>"
                    if block.get("id"):
                        inner += f"\n<id>{block['id']}</id>"
                    if output_text:
                        inner += f"\n<output>{output_text}</output>"
                    parts.append(f"<tool_call{attrs}>{inner}</tool_call>")

        parts.append("</message>")

    return "\n".join(parts)
