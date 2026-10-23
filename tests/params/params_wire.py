"""Fixtures shared by the params tests: a recording model, an offline backward stub, and agents.

Everything the model saw is asserted from what the model actually received (``Recording.contexts``),
and everything about the optimizer's graph is asserted from ``build_graph_from_result`` over the
coordinator's own event log, never from state the params package keeps for itself.

Output types, functions and agents are module level because ``compile_ai_method`` and the ``@ai_function``
decorator resolve annotations with ``typing.get_type_hints`` against module globals; a function-local
``Trainable`` annotation cannot be resolved.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, Field
from strands.models import Model

from ai_functions import ai_function
from ai_functions.experimental.methods import MethodAgent, ai_method
from ai_functions.experimental.params import NUMERIC_ADD, Trainable
from ai_functions.optimizer.textgrad import Feedback, Feedbacks
from ai_functions.testing import ScriptedModel, Turn

# ── Output types ──


class Choice(BaseModel):
    """What the navigator returns; named for the tool call the scripted turns make."""

    transition: str = Field(description="The transition to take next.")
    reason: str = Field(description="Why.")


# ── Functions ──


@ai_function[str](structured_output=False)
def research(query: str, instructions: Annotated[str, Trainable()] = "Find relevant facts.") -> str:
    """Instructions: {instructions}

    Research: {query}
    """
    ...


@ai_function[str](structured_output=False)
def write(
    facts: str,
    style: Annotated[str, Trainable(conflict="merge", description="How the report should read.")] = "Plain prose.",
) -> str:
    """Style: {style}

    Write a report from: {facts}
    """
    ...


@ai_function[str](structured_output=False)
def weigh(
    query: str,
    weights: Annotated[list[float], Trainable(reducer=NUMERIC_ADD)] = [1.0, 0.5],  # noqa: B006 (the default is the initial value)
) -> str:
    """Weights: {weights}. Query: {query}."""
    ...


@ai_function[str](structured_output=False)
def plain(query: str) -> str:
    """Just {query}."""
    ...


# ── Agents ──


class Navigator(MethodAgent):
    """One ``Trainable`` capability and one plain one on the same agent."""

    name = "navigator"

    @ai_method(Choice, description="Choose the next transition")
    def choose(
        self,
        state: str,
        playbook: Annotated[str, Trainable()] = "Prefer the shortest path.",
    ) -> Choice:
        """Playbook: {playbook}

        You are in {state}. Choose.
        """
        ...

    @ai_method(Choice, description="Choose with nothing trainable")
    def guess(self, state: str) -> Choice:
        """You are in {state}. Guess."""
        ...


# ── Models ──


def scripted(text: str = "done") -> ScriptedModel:
    """One text turn, for ``@ai_function`` traces."""
    return ScriptedModel([Turn(text=text)])


def choosing(turns: int) -> Recording:
    """A recording model scripted with ``turns`` ``Choice`` tool calls; ``choosing(0)`` proves a guard fired first."""
    return Recording(
        [Turn(tool_calls=(("Choice", {"transition": "T0", "reason": "r"}),)) for _ in range(turns)],
    )


def texting(turns: int, text: str = "done") -> Recording:
    """A recording model scripted with ``turns`` text turns, for ``@ai_function`` traces."""
    return Recording([Turn(text=text) for _ in range(turns)])


class Recording(Model):
    """A ``ScriptedModel`` that also records the messages each call received.

    ``ScriptedModel`` is ``@final`` and ignores ``messages``, so composition: record what the model was
    handed, then delegate the response.
    """

    def __init__(self, turns: list[Turn]) -> None:
        super().__init__()
        self._inner = ScriptedModel(turns)
        self.contexts: list[list[Any]] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, object]:
        return {"calls": len(self.contexts)}

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("scripted turns only")

    def stream(self, messages: Any, tool_specs: Any = None, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        self.contexts.append(list(messages))
        return self._inner.stream(messages, tool_specs, *args, **kwargs)

    def prompts(self, call: int) -> list[str]:
        """Every text block the model saw on its ``call``-th invocation."""
        return [b["text"] for m in self.contexts[call] for b in m.get("content", []) if "text" in b]


# ── The backward pass, offline ──


class EchoBackward:
    """Stands in for the backward AI function: routes each incoming feedback item verbatim to every listed
    target, and records how many times it ran. ``replace`` returns ``self`` so the optimizer's per-node
    ``.replace(post_conditions=[...])`` is a no-op offline. The same seam ``tests/test_trace_dataflow.py`` uses."""

    def __init__(self, score: float = 0.5) -> None:
        self.calls = 0
        self.score = score
        self.inputs: list[str] = []

    def replace(self, **kwargs: object) -> EchoBackward:
        del kwargs
        return self

    def run_sync(self, *, inputs: str, feedback: list[str], **kwargs: object) -> Feedbacks:
        del kwargs
        self.calls += 1
        self.inputs.append(inputs)
        target_ids = list(yaml.safe_load(inputs).keys())
        return Feedbacks(
            feedbacks=[Feedback(node_id=tid, feedback=f, score=self.score) for tid in target_ids for f in feedback]
        )
