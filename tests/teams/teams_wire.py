"""Wire-recording fixtures shared by the teams core and members tests.

Every claim about delivery is checked ON THE WIRE — from the contexts and ``tool_specs`` a
recording model actually received — never from the returned ``TeamRun`` alone, which a
mis-wired run produces just as happily. Budget claims count model calls with ``Counting([])``,
so a guard that raises before the model call shows as ``contexts == []``.

``Counting`` composes rather than subclasses ``ScriptedModel`` (which is ``@final``), the
pattern every team test here shares. The cast is compiled functions (``decide``, ``read``), the
shape a ``FunctionMember`` wraps.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Sequence
from typing import Any

from pydantic import BaseModel, Field
from strands.models import Model

from ai_functions import AIFunction, ai_function
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types import ThreadId

# ── Output types ──


class Reading(BaseModel):
    source: str = Field(description="Which evidence this reading came from")
    detail: str = Field(description="What it shows")


class Ruling(BaseModel):
    admitted: bool = Field(description="Whether this ruling is ready")
    cites: list[str] = Field(default_factory=list, description="Which members were relied on")


# ── The cast: compiled functions, the shape a FunctionMember wraps ──


@ai_function[Ruling]
def decide(question: str, rigor: str = "normal") -> str | None:
    """Rule on {question}, with {rigor} rigor. Consult the members you hold as tools."""


@ai_function[Reading]
def read(focus: str, depth: int = 2) -> str | None:
    """Read the source with {focus} in mind, to depth {depth}."""


# ── Recording model: contexts, tool offers and tool results, all wire facts ──


class Counting(Model):
    """Composes a ``ScriptedModel`` and records what each call carried on the wire."""

    def __init__(self, turns: list[Turn], *, tag: str = "", journal: list[str] | None = None) -> None:
        super().__init__()
        self._inner = ScriptedModel(turns)
        self._tag = tag
        self._journal = journal
        self.contexts: list[list[Any]] = []
        self.tool_specs: list[list[str]] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, object]:
        return {"calls": len(self.contexts)}

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("scripted turns only")

    def stream(self, messages: Any, tool_specs: Any = None, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        self.contexts.append(list(messages))
        self.tool_specs.append([spec["name"] for spec in (tool_specs or [])])
        if self._journal is not None:
            self._journal.append(f"model:{self._tag}")
        return self._inner.stream(messages, tool_specs, *args, **kwargs)

    def prompts(self, call: int) -> list[str]:
        return [
            block["text"] for message in self.contexts[call] for block in message.get("content", []) if "text" in block
        ]

    def tool_results(self, call: int) -> list[str]:
        return [
            inner["text"]
            for message in self.contexts[call]
            for block in message.get("content", [])
            if "toolResult" in block
            for inner in block["toolResult"].get("content", [])
            if "text" in inner
        ]


# ── A member that never touches a model ──


class _FakeHandle:
    def __init__(self, ident: str) -> None:
        self.id = ThreadId(ident)


class Spy:
    """A ``Member`` that records its own lifecycle and never touches a model."""

    def __init__(self, name: str, *, answer: str = "ok") -> None:
        self.name = name
        self.answer = answer
        self.events: list[str] = []
        self.retirements = 0
        self.parent_id: ThreadId | None = None

    async def spawn(self, coordinator: Any, *, parent_id: ThreadId | None = None) -> _FakeHandle:
        self.events.append("spawn")
        self.parent_id = parent_id
        return _FakeHandle(f"tid-{self.name}")

    async def ask(self, request: str) -> Any:
        self.events.append(f"ask:{request}")
        return self.answer

    async def retire(self) -> None:
        self.retirements += 1
        self.events.append("retire")


def reading(source: str = "left", detail: str = "seen") -> Turn:
    return Turn(tool_calls=(("Reading", {"source": source, "detail": detail}),))


def ruling(*, admitted: bool = True, cites: Sequence[str] = ()) -> Turn:
    return Turn(tool_calls=(("Ruling", {"admitted": admitted, "cites": list(cites)}),))


def call_member(name: str, request: str = "report your evidence") -> Turn:
    return Turn(tool_calls=((name, {"request": request}),))


def scripted_lead(turns: list[Turn], **overrides: Any) -> tuple[AIFunction[..., Any], Counting]:
    """``decide`` bound to a recording model, plus that model, for scripting the lead's turns."""
    model = Counting(turns)
    return decide.replace(model=model, **overrides), model
