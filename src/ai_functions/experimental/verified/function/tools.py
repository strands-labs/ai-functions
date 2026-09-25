"""The model's tools: observation tools supplying values of opaque Lean symbols, and the Lean tools."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from strands import tool as _strands_tool
from strands.types.tools import AgentTool, ToolContext

from ..lean import LeanError, LeanProject, LeanSymbol
from ..lean.execution import run_in_thread
from ._session import _IDENTIFIER, Outcome, ProofSession, _check_opaque
from .types import Certified


def _report(name: str, outcome: Outcome) -> str:
    """The tool result the model reads, also shown in the console when Strands' tool console is enabled."""
    message = outcome.message + ("\n" + outcome.detail if outcome.detail else "")
    if os.environ.get("STRANDS_TOOL_CONSOLE_MODE") == "enabled":
        parts: list[Any] = [Text(message)]
        if outcome.code:
            parts.append(Syntax(outcome.code, "lean", word_wrap=True, background_color="default"))
        Console().print(Panel(Group(*parts), title=name, border_style="green" if outcome.ok else "red"))
    code = outcome.excerpt or outcome.code
    return message + (f"\n```lean\n{code}\n```" if code else "")


@dataclass(frozen=True)
class ObservationTool:
    """A Python function bound to an opaque symbol, exposed anew to each certified call."""

    fn: Callable[..., object]
    symbol: LeanSymbol
    stem: str
    name: str

    @property
    def project(self) -> LeanProject:
        """The symbol's project."""
        return self.symbol.project

    def bind(self, session: ProofSession) -> AgentTool:
        """Expose the tool to the model for one session."""
        _check_opaque(self.symbol)
        parameters = self.symbol.info.parameters
        signature = inspect.signature(self.fn)
        positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        names = [p.name for p in signature.parameters.values() if p.kind in positional]
        if not _IDENTIFIER.fullmatch(self.stem):
            raise LeanError(f"The stem of {self.name} is not a plain Lean identifier: {self.stem!r}")
        if len(names) != len(signature.parameters) or len(names) != len(parameters):
            raise LeanError(
                f"{self.name} must take {len(parameters)} positional parameters, those of {self.symbol.name}"
            )

        async def invoke(**kwargs: str) -> str:
            slots = [(name, kwargs[name], p.type) for name, p in zip(names, parameters, strict=True)]
            try:
                values = await run_in_thread(session.evaluate, slots)
            except ValueError as exc:
                return _report(self.name, Outcome(False, str(exc)))
            try:
                if inspect.iscoroutinefunction(self.fn):
                    returned = await self.fn(*values)
                else:
                    returned = await run_in_thread(self.fn, *values)
            except Exception as exc:
                message = f"TOOL ERROR: {type(exc).__name__}: {exc}\nNothing was recorded."
                return _report(self.name, Outcome(False, message))
            result = returned if isinstance(returned, Certified) else Certified(returned)
            origin = f"{self.name}({', '.join(f'{n}={v!r}' for n, v in zip(names, values, strict=True))})"
            outcome = await run_in_thread(
                session.observe,
                self.symbol,
                values,
                result.value,
                origin=origin,
                stem=self.stem,
                guarantees=result.guarantees,
                written=[kwargs[name] for name in names],
            )
            return _report(self.name, outcome)

        invoke.__name__ = self.name
        invoke.__doc__ = (
            (inspect.getdoc(self.fn) or f"Observe {self.symbol.name}.")
            + "\nEvery argument is Lean source evaluated at the declared type: ledger names such as H.message, "
            + 'expressions, or literals; string literals need quotes, e.g. `"widget"`.'
            + f"\nLean interface: {self.symbol.name} : {self.symbol.info.type_str}"
        )
        invoke.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str) for n in names],
            return_annotation=str,
        )
        invoke.__annotations__ = {**dict.fromkeys(names, str), "return": str}
        return _strands_tool(invoke)


def tool(
    symbol: LeanSymbol,
    *,
    stem: str | None = None,
    name: str | None = None,
) -> Callable[[Callable[..., object]], ObservationTool]:
    """Declare a tool observing ``symbol``."""
    if not isinstance(symbol, LeanSymbol):
        raise TypeError("tool() takes a LeanProject symbol")

    def decorate(fn: Callable[..., object]) -> ObservationTool:
        return ObservationTool(fn, symbol, stem or fn.__name__, name or fn.__name__)

    return decorate


class _Answer(BaseModel):
    answer: Any


def _lean_tools(session: ProofSession, *, judge: bool) -> list[AgentTool]:
    """The Lean tools for one session: ``lean``, ``lean_eval``, ``lean_judge`` if ``judge``, ``lean_submit``."""

    async def lean(code: str) -> str:
        """Append Lean definitions and lemmas to the ledger.

        Supply declarations with bare names (`def total`, not `def A.total`); the harness
        places them in namespace A. They can use the imported project symbols and the H
        input and observation names. A committed block reports the names it declared,
        such as A.total; refer to them as A.total outside `lean` blocks. A failed block
        leaves the ledger unchanged. Use lean_eval for scratch checks.
        """
        return _report("lean", await run_in_thread(session.append, code))

    async def lean_eval(expr: str) -> str:
        """Evaluate one Lean term, or run #check / #print commands, without changing the ledger.

        The term is reduced by Lean's elaborator and shown as a literal; do not write
        #eval. Values of Nat, Int, String, Bool, and lists and tuples of them can be
        shown. Write a do block as `Id.run do ...` and a proposition as `decide (...)`.
        Opaque functions do not reduce: compute with the transparent H observation
        values and use their _spec equations in proofs.
        """
        return _report("lean_eval", await run_in_thread(session.inspect, expr))

    async def lean_judge(symbol: str, args: list[str], value: str, justification: str) -> str:
        """Record your reading of an allowlisted opaque symbol at given arguments, as an axiom J.<name>.

        `args` and `value` are Lean expressions at the symbol's argument and result
        types, such as `H.message` or `4`; they must reduce to closed values. When the
        result is a proposition or a theory type, `value` is a term over the project's
        vocabulary, such as `Shop.inStock "widget" ∧ 2 ≤ Shop.stock "widget"`. The
        justification is one sentence of reasoning from the input to the value. A
        recorded reading cannot be revised. When an argument is an expression such as
        `H.message`, it also records J.<name>_as_written, the reading stated at your
        expressions, which rewrites the goal directly.
        """
        return _report("lean_judge", await run_in_thread(session.judge, symbol, args, value, justification))

    async def lean_submit(answer: str, proof: str, tool_context: ToolContext) -> str:
        """Submit a Lean answer and a proof of its fixed contract.

        `answer` is Lean source at the contract's result type, for example `4797`,
        `A.total`, or a function term when the goal requires a function. `proof` is a
        Lean proof term, such as `by ...` or `A.correct`; a tactic proof starts with
        `by`. A failure shows the exact theorem. Success requires an independent cold
        check and axiom audit, and ends the function immediately.
        """
        outcome = await run_in_thread(session.submit, answer, proof)
        if outcome.ok:
            state = tool_context.invocation_state["request_state"]
            state["tool_result"] = _Answer(answer=session.result)
            state["stop_event_loop"] = True
        return _report("lean_submit", outcome)

    return [
        _strands_tool(lean),
        _strands_tool(lean_eval),
        *([_strands_tool(lean_judge)] if judge else []),
        _strands_tool(context=True)(lean_submit),
    ]
