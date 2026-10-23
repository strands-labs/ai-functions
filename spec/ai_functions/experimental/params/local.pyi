"""``Trainable`` + ``Bind`` + ``Local``: a trainable value that lives in the process, declared on the signature.

The ``Recalled``/``Recall`` pair (``methods/recall.py``) makes a memory field a gradient target
by injecting a ``ParameterView`` at the trace boundary. This module does the same for a value
that is not a memory entry at all: a string literal on a pipeline, a weight vector on a router,
anything a signature can carry a default for. The marker declares, the binder performs, and the
host holds the value, its version and its combining rule.

    @ai_function
    def research(query: str, instructions: Trainable[str] = "Find relevant facts.") -> str:
        '''Instructions: {instructions}. Research: {query}.'''

    bound = Bind(research)
    result = await bound.trace(query="cats")
    await TextGradOptimizer(model=m).step(result, "Cite sources.", backends=[bound.local])
    bound.local.load("instructions")   # the initial text with "Cite sources." appended, at version 1

One annotation, one default, and the parameter is trainable under sensible defaults: the
default is the initial value, the reducer follows the value's shape (text appends in order,
numbers add), and a stale concurrent write is refused. Every knob is optional and named for
what it does: ``Annotated[str, Trainable(reducer=ORDERED, conflict="merge")]``.

**Why the same trace-boundary injection.** ``collect_nodes((args, kwargs))`` inside
``AIFunction.trace`` is the one place a value can enter and still be a gradient target
(``methods/recall.py`` explains the four silent ways to lose the edge). ``Local.view`` hands out a
fresh, un-emitted ``ParameterView`` per call whose ``backend`` is the ``Local``; ``trace`` emits the
recall event on the traced thread, ``build_graph`` rebuilds a ``ParameterNode`` whose ``backend`` is
the ``Local``, and ``TextGradOptimizer.consolidate`` calls ``Local.consolidate`` through the
structural ``ParameterHost`` seam. Nothing in the core learns a new class.

**Why versions.** Two learners that both read version 0 and both consolidate would silently
lose one write. Every view carries the version it read in ``meta["version"]``; the optimizer
hands it back as ``expected_version``; the parameter's declared ``conflict`` policy decides:
``refuse`` (the default) raises :class:`VersionConflictError` naming the parameter and both
versions, ``merge`` folds the stale feedback onto the current value, ``last_writer_wins``
rebuilds from the version the writer read and drops the other write.

Invariants:
    LP1 — the marker declares, the binder performs, the host holds. ``Trainable`` inside
    ``Annotated`` (``Trainable[T]`` is ``Annotated[T, Trainable()]``) says the parameter is
    local and how it learns; ``Bind`` injects a fresh view per traced call; ``Local`` owns the
    value. The parameter stays in the tool schema, so a peer can still supply it by hand,
    and a supplied value triggers no injection.

    LP2 — the Python default is the initial value. A ``Trainable`` parameter without a default,
    a positional-only one, one named after a ``Bind.trace`` keyword, or one also marked
    ``Recalled`` is refused at bind time, before any model call.

    LP3 — one view, one node, one event: ``Local.view`` returns a new un-emitted
    ``ParameterView`` per call, so a second trace yields a fresh node and a reused view is
    unrepresentable. The binder holds the target and the host and no view.

    LP4 — every local parameter carries a monotonically increasing ``version``, bumped by
    exactly one on every ``consolidate`` and every ``apply`` that changes the value; a write that
    leaves the value equal is not a new version. ``load`` returns ``(value, version)`` and
    ``history`` keeps one value per version. ``snapshot`` and ``restore`` are inverses: a host
    declared the same way and restored from a snapshot holds the same values at the same versions,
    and its history starts at the restored version.

    LP5 — a consolidate with ``expected_version`` set applies under the parameter's declared
    ``conflict`` policy; ``refuse`` raises :class:`VersionConflictError` naming the parameter and
    both versions and changes nothing. A version the host never had, or one before a restored
    version whose value is gone, is refused under every policy. A consolidate without
    ``expected_version`` applies to the current value.

    LP6 — the combining rule is the parameter's declared :class:`FeedbackReducer`, inferred
    from the initial value's shape when not given, refused when no built-in fits. Consolidation
    makes no model call.

    LP7 — ``components()`` and ``apply()`` are the bridge to a search optimizer: the declared
    parameters as one ``dict[str, str]`` mapping candidate, and a mapping candidate written back
    with a version bump per parameter whose value changed; an unknown name or a text that does
    not parse as the parameter's type is refused before anything changes.

    LP8 — a parameter declared twice with the same initial, reducer and policy is one parameter
    (two stages sharing it); declared twice with a different initial it is refused naming both.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from ...ai_thread import AIFunction
from ...protocols import Coordinator
from ...types.graph import GradFeedback, ParameterView, Result
from ...types.ids import ThreadId
from ..methods.core import MethodAgent
from .reducers import FeedbackReducer, ParamsError

Conflict = Literal["last_writer_wins", "merge", "refuse"]
"""What a stale concurrent consolidate does (LP5)."""

class ParamWiringError(ParamsError):
    """A marker, declaration or bind guard refused before any model call (LP2, LP6, LP7, LP8)."""

class VersionConflictError(ParamsError):
    """A consolidate against a version the parameter has moved past, under ``conflict="refuse"`` (LP5).

    Args:
        name: The parameter.
        expected: The version the writer read.
        current: The version the host holds now.
        backend_id: The host, for the message.
    """

    name: str
    expected: int
    current: int
    def __init__(self, name: str, expected: int, current: int, backend_id: str = "") -> None: ...

@dataclass(frozen=True)
class Trainable:
    """Annotated metadata: this parameter is a local trainable value (LP1).

    ``Trainable[str]`` is the zero-knob spelling and means ``Annotated[str, Trainable()]``. The
    Python default on the signature is the initial value (LP2). Type checkers read
    ``Trainable[str]`` as this marker class and reject its default, so type-checked code writes
    ``Annotated[str, Trainable()]``, the same annotation at run time. Every knob is optional:

    Args:
        reducer: How several pieces of feedback combine into the next value; inferred from the
            initial value's shape when ``None`` (text appends in order, numbers add). A bare
            callable ``(current, feedback) -> new`` is accepted.
        conflict: What a stale concurrent consolidate does; ``"refuse"`` raises
            :class:`VersionConflictError`, ``"merge"`` folds onto the current value,
            ``"last_writer_wins"`` rebuilds from the version the writer read.
        description: What the parameter is for, shown to the backward model; the reducer's own
            guidance is appended.
    """

    reducer: FeedbackReducer | Callable[[Any, Sequence[GradFeedback]], Any] | None = None
    conflict: Conflict = "refuse"
    description: str = ""
    def __class_getitem__(cls, item: Any) -> Any:
        """``Trainable[T]`` is ``Annotated[T, Trainable()]``."""
        ...

def trainable_params(fn: Callable[..., Any]) -> dict[str, Trainable]:
    """The ``Trainable`` parameters of a function or bound method, in signature order.

    Args:
        fn: A prompt function, or a bound ``@ai_method`` (whose hints already exclude ``self``).

    Returns:
        Parameter name to its marker, for the marked ones only.

    Raises:
        ParamWiringError: An annotation carries two distinct markers, or a ``Recalled`` too.
        NameError: An annotation names a type that cannot be resolved.
    """
    ...

class Param(BaseModel):
    """One local parameter as an orchestrator may store it; frozen, JSON round-trippable (LP4).

    A snapshot: the live value, reducer object and history stay on the :class:`Local`;
    :meth:`Local.restore` puts the value and version back into a host declared the same way.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    value: JsonValue
    """The current value in its JSON form."""
    version: int
    """Bumped by one on every consolidate and apply that changes the value (LP4)."""
    reducer: str
    """The declared reducer's ``name``."""
    conflict: Conflict
    description: str = ...

class Local:
    """The host of local parameters: values, versions, reducers, and the ``ParameterHost`` seam (LP3 to LP8).

    ``Local(instructions="Find facts.")`` declares one parameter under the defaults; :meth:`declare`
    takes the knobs. A :class:`Bind` declares from a signature's markers and defaults, so most
    users never construct one directly; ``bound.local`` is what ``TextGradOptimizer`` takes in
    ``backends=[...]``.

    Args:
        initial: Parameter name to initial value, each declared under the inferred reducer and
            ``conflict="refuse"``.
    """

    def __init__(self, **initial: Any) -> None: ...
    @property
    def backend_id(self) -> str:
        """``"Local:<id>"``, matched against recall events to recover this host."""
        ...

    def declare(
        self,
        name: str,
        initial: Any,
        *,
        reducer: FeedbackReducer | Callable[[Any, Sequence[GradFeedback]], Any] | None = None,
        conflict: Conflict = "refuse",
        description: str = "",
        annotation: object = None,
    ) -> Param:
        """Declare a parameter at version 0 (LP6, LP8).

        Args:
            name: The parameter name.
            initial: The initial value; its shape picks the reducer when none is given.
            reducer: The combining rule, or a bare callable; inferred when ``None``.
            conflict: The stale-write policy.
            description: What the parameter is for, shown to the backward model.
            annotation: The declared type, for rehydrating values from events; ``type(initial)`` when ``None``.

        Returns:
            The parameter's record; the existing one when an identical declaration already exists.

        Raises:
            ParamWiringError: ``conflict`` is not a policy, no reducer fits ``initial`` and none was
                given, or ``name`` is already declared with a different initial, reducer or policy.
        """
        ...

    @property
    def names(self) -> tuple[str, ...]:
        """Declared parameter names, in declaration order."""
        ...

    def param(self, name: str) -> Param:
        """The parameter's frozen record (LP4).

        Args:
            name: The parameter name.

        Raises:
            ParamWiringError: ``name`` is not declared.
        """
        ...

    def snapshot(self) -> dict[str, Param]:
        """Every parameter's record, by name; :meth:`restore` is its inverse (LP4)."""
        ...

    def restore(self, params: Mapping[str, Param]) -> None:
        """Put stored records back into a host declared the same way (LP4).

        A restarted process declares its parameters as before (``Bind`` or :meth:`declare`), then
        restores the snapshot it stored, so every graph traced before the restart consolidates
        against the version it read. A record's reducer name and conflict policy must agree with the
        declaration, because the record carries only the reducer's name. History before the restored
        version is not available: ``last_writer_wins`` has no base to rebuild from there and refuses
        (LP5). Nothing is written until every record has passed its checks.

        Args:
            params: Parameter name to its record, as :meth:`snapshot` returned it. Declared
                parameters absent from the mapping stay as declared.

        Raises:
            ParamWiringError: A name is not declared, a record's name, reducer or policy disagrees
                with the declaration, its value does not validate as the declared type, or the
                parameter has already been written in this host.
        """
        ...

    def load(self, name: str) -> tuple[Any, int]:
        """The current value and its version (LP4).

        Args:
            name: The parameter name.
        """
        ...

    def history(self, name: str) -> tuple[Any, ...]:
        """One value per version, oldest first (LP4); after :meth:`restore` it starts at the restored version.

        Args:
            name: The parameter name.
        """
        ...

    def view(self, name: str) -> ParameterView[Any]:
        """A fresh, un-emitted ``ParameterView`` of the parameter, for one traced call (LP3).

        The view's ``meta`` carries the version it read and the reducer's name; its
        ``description`` carries the reducer's guidance so the backward model knows what shape of
        feedback the target accepts.

        Args:
            name: The parameter name.
        """
        ...

    def deserialize_value(self, name: str, raw: Any) -> Any:
        """Rehydrate a value from its event representation; the raw value when validation fails."""
        ...

    def _is_procedural(self, name: str) -> bool:
        """Local parameters are never code parameters."""
        ...

    async def emit_recall(self, view: ParameterView[Any], coordinator: Coordinator, thread_id: ThreadId) -> None:
        """Emit the recall event for a view consumed by a traced thread (LP3).

        Called by ``AIFunction.trace``. No-op when the view was already emitted, so one logical
        read never lands in two logs. Confirms the append is readable before returning, bounded.
        """
        ...

    def consolidate(
        self,
        name: str,
        feedback: list[GradFeedback],
        retrieved: dict[str, str] | None = None,
        *,
        expected_version: int | None = None,
    ) -> None:
        """Fold routed feedback into the parameter under its reducer and conflict policy (LP5, LP6).

        Args:
            name: The parameter name.
            feedback: The gradients the backward pass routed here.
            retrieved: Unused; a local parameter is read whole. Part of the ``ParameterHost`` signature.
            expected_version: The version the graph read, when the caller knows it; ``None`` applies
                to the current value.

        Raises:
            VersionConflictError: ``expected_version`` is one the host never had or no longer holds
                the value of, or is stale under ``conflict="refuse"``.
            ReducerError: The reducer could not combine the feedback with the value.
        """
        ...

    def components(self) -> dict[str, str]:
        """The declared parameters as one mapping candidate (LP7): text as is, anything else as canonical JSON."""
        ...

    def apply(self, candidate: Mapping[str, str]) -> None:
        """Write a mapping candidate back, bumping the version of each parameter whose value changed (LP7).

        Args:
            candidate: Parameter name to its new text; a non-text parameter's text is parsed as JSON. A
                name whose text equals the current value is left alone, version included.

        Raises:
            ParamWiringError: A name is not declared, or a text does not parse as its parameter's
                type; nothing is written.
        """
        ...

    def __repr__(self) -> str:
        """The host id and its declared names."""
        ...

class Bind:
    """Binds an ``@ai_function`` or a ``MethodAgent`` to a ``Local`` and fills its ``Trainable`` parameters (LP1-LP3).

    Construction reads every marker and declares each parameter in the host from its default (LP2);
    ``trace`` injects a fresh view for each marked parameter the caller did not supply and traces the
    target, where ``collect_nodes`` finds the views. For an agent, the first positional argument of
    ``trace`` is the method name, as with ``Recall.trace``.

    Args:
        target: The ``AIFunction`` or ``MethodAgent`` whose signature carries ``Trainable`` markers.
        local: The host to declare into; a fresh one when ``None``. Several binders share one host
            when a pipeline's stages share parameters.

    Raises:
        ParamWiringError: ``target`` is neither, or a guard in LP2 refused a parameter.
    """

    RESERVED: tuple[str]
    """Keyword names ``trace`` owns. A bound parameter sharing one is refused at wiring time."""

    def __init__(self, target: AIFunction[..., Any] | MethodAgent, local: Local | None = None) -> None: ...
    @property
    def target(self) -> AIFunction[..., Any] | MethodAgent:
        """The bound function or agent. Read-only."""
        ...

    @property
    def local(self) -> Local:
        """The host: what ``TextGradOptimizer`` takes in ``backends=[...]``."""
        ...

    def bound(self, method: str | None = None) -> dict[str, Trainable]:
        """The marked parameters of the function, or of one method on the agent, in signature order.

        Args:
            method: The ``@ai_method`` name; ``None`` for a bound function.

        Raises:
            ParamWiringError: ``method`` is not an ``@ai_method`` of the agent, or was given for a function.
        """
        ...

    def components(self) -> dict[str, str]:
        """The host's declared parameters as one mapping candidate (LP7)."""
        ...

    async def trace(self, *args: Any, overrides: Mapping[str, Any] | None = None, **kwargs: Any) -> Result[Any]:
        """Inject a fresh view for every marked parameter the caller did not supply, then trace (LP1, LP3).

        Args:
            args: For an agent, the method name first, then positional arguments; for a function,
                positional arguments. They may not reach a marked parameter.
            overrides: ``ThreadConfig`` overrides (``{"model": ScriptedModel(...)}``): ``replace`` for a
                function, ``compiled`` for an agent.
            kwargs: Keyword arguments, forwarded verbatim. A marked name here is an explicit supply.

        Returns:
            The ``Result`` from ``AIFunction.trace``; its ``inputs`` are the views this call carried.

        Raises:
            ParamWiringError: A guard refused the call before any model call.
        """
        ...

    def __repr__(self) -> str:
        """The bound target's name and the host id."""
        ...
