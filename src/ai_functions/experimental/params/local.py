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

from __future__ import annotations

import asyncio
import inspect
import json
import typing
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from ai_functions.ai_thread import AIFunction
from ai_functions.experimental.methods.core import MethodAgent, _owner_name, _published_name, is_ai_method
from ai_functions.experimental.methods.recall import Recalled
from ai_functions.types.events import EventKind, ParameterRecalledEvent
from ai_functions.types.graph import GradFeedback, ParameterView, Result

from .reducers import FeedbackReducer, ParamsError, as_reducer, default_reducer

if TYPE_CHECKING:
    from ai_functions.protocols import Coordinator
    from ai_functions.types.ids import ThreadId

Conflict = Literal["last_writer_wins", "merge", "refuse"]
"""What a stale concurrent consolidate does (LP5)."""

_CONFLICTS: tuple[Conflict, ...] = ("last_writer_wins", "merge", "refuse")

# Bounded read-after-write confirmation for the recall event, the same shape ``MemoryBackend`` uses:
# an in-process coordinator confirms on the first read; a network one may need the loop to run.
_CONFIRM_MAX_READS = 50
_CONFIRM_BACKOFF_STEP = 0.01
_CONFIRM_BACKOFF_MAX = 0.2

# ── Errors ──


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

    def __init__(self, name: str, expected: int, current: int, backend_id: str = "") -> None:
        where = f" on {backend_id}" if backend_id else ""
        super().__init__(
            f"local parameter {name!r}{where}: this consolidate read version {expected} but the value is at "
            f"version {current}, so another learner wrote in between. Declare the parameter with "
            f"conflict='merge' to fold both writes in, or conflict='last_writer_wins' to keep this one."
        )
        self.name = name
        self.expected = expected
        self.current = current


# ── The marker ──


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

    reducer: FeedbackReducer | Callable[[Any, Sequence[GradFeedback]], Any] | None = None  # pyright: ignore[reportExplicitAny]
    conflict: Conflict = "refuse"
    description: str = ""

    def __class_getitem__(cls, item: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """``Trainable[T]`` is ``Annotated[T, Trainable()]``."""
        return Annotated[item, cls()]


def _marker_of(hint: object) -> Trainable | None:
    """The ``Trainable`` in an annotation, looking through unions, or ``None``.

    Two distinct markers are refused rather than resolved by order (which reducer a parameter
    learns under must not depend on metadata order), and a ``Recalled`` beside a ``Trainable`` is
    refused because a parameter comes from memory or from the process, not both.
    """
    found: list[Trainable] = []
    recalled: list[Recalled] = []

    def _walk(node: object) -> None:
        for meta in getattr(node, "__metadata__", ()):
            if isinstance(meta, Trainable) and meta not in found:
                found.append(meta)
            elif isinstance(meta, Recalled):
                recalled.append(meta)
        for arg in typing.get_args(node):
            _walk(arg)

    _walk(hint)
    if len(found) > 1:
        raise ParamWiringError(
            f"annotation carries {len(found)} distinct `Trainable` markers ({found!r}); which reducer and policy "
            f"apply would depend on metadata order, so declare exactly one"
        )
    if found and recalled:
        raise ParamWiringError(
            f"annotation carries both `Trainable` and `Recalled` ({recalled[0]!r}); a parameter is filled from "
            f"memory (Recalled, bound by Recall) or from the process (Trainable, bound by Bind), not both. Keep one."
        )
    return found[0] if found else None


def trainable_params(fn: Callable[..., Any]) -> dict[str, Trainable]:  # pyright: ignore[reportExplicitAny]
    """The ``Trainable`` parameters of a function or bound method, in signature order.

    ``get_type_hints`` is allowed to raise: an unresolvable annotation is a wiring bug and reads
    as one, because the cost of swallowing it here would be no injection at all.

    Args:
        fn: A prompt function, or a bound ``@ai_method`` (whose hints already exclude ``self``).

    Returns:
        Parameter name to its marker, for the marked ones only.

    Raises:
        ParamWiringError: An annotation carries two distinct markers, or a ``Recalled`` too.
        NameError: An annotation names a type that cannot be resolved.
    """
    hints = typing.get_type_hints(fn, include_extras=True)
    hints.pop("return", None)
    found: dict[str, Trainable] = {}
    for name, hint in hints.items():
        marker = _marker_of(hint)
        if marker is not None:
            found[name] = marker
    return found


def _unannotated(hint: object) -> object:
    """The hint with its ``Annotated`` wrapper removed, for a ``TypeAdapter``."""
    if typing.get_origin(hint) is Annotated:
        return get_args(hint)[0]
    return hint


# ── The record and the host ──


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
    description: str = ""


@dataclass
class _Slot:
    """The live state of one parameter."""

    value: Any  # pyright: ignore[reportExplicitAny]
    initial: Any  # pyright: ignore[reportExplicitAny]
    reducer: FeedbackReducer
    conflict: Conflict
    description: str
    adapter: TypeAdapter[Any]  # pyright: ignore[reportExplicitAny]
    version: int = 0
    history: list[Any] = field(default_factory=list)  # pyright: ignore[reportExplicitAny]
    """One value per version from ``history_base`` up to ``version``."""
    history_base: int = 0
    """The version ``history[0]`` holds: 0 when declared, the restored version after ``restore``."""

    def base_for(self, version: int) -> Any:  # pyright: ignore[reportExplicitAny]
        """The value at ``version``, or ``None`` when it is before the history or after the current one."""
        index = version - self.history_base
        if index < 0 or index >= len(self.history):
            return None
        return self.history[index]


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

    def __init__(self, **initial: Any) -> None:  # pyright: ignore[reportExplicitAny]
        self._id = uuid.uuid4().hex[:12]
        self._slots: dict[str, _Slot] = {}
        for name, value in initial.items():
            self.declare(name, value)

    # ── Identification (ParameterHost) ──

    @property
    def backend_id(self) -> str:
        """``"Local:<id>"``, matched against recall events to recover this host."""
        return f"Local:{self._id}"

    # ── Declaration ──

    def declare(
        self,
        name: str,
        initial: Any,  # pyright: ignore[reportExplicitAny]
        *,
        reducer: FeedbackReducer | Callable[[Any, Sequence[GradFeedback]], Any] | None = None,  # pyright: ignore[reportExplicitAny]
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
        if conflict not in _CONFLICTS:
            raise ParamWiringError(
                f"local parameter {name!r}: conflict={conflict!r} is not a policy; pick one of "
                f"{', '.join(repr(c) for c in _CONFLICTS)}"
            )
        chosen = as_reducer(reducer) if reducer is not None else default_reducer(initial)
        if chosen is None:
            raise ParamWiringError(
                f"local parameter {name!r}: no built-in reducer fits a {type(initial).__name__} initial value "
                f"({initial!r}); declare it with reducer=<FeedbackReducer or callable>"
            )
        existing = self._slots.get(name)
        if existing is not None:
            same = (
                existing.initial == initial and existing.reducer.name == chosen.name and existing.conflict == conflict
            )
            if not same:
                raise ParamWiringError(
                    f"local parameter {name!r} is already declared with initial {existing.initial!r} "
                    f"(reducer {existing.reducer.name!r}, conflict {existing.conflict!r}); the new declaration "
                    f"has initial {initial!r} (reducer {chosen.name!r}, conflict {conflict!r}). Two stages sharing "
                    f"a parameter must agree on it; otherwise name them differently."
                )
            return self.param(name)
        slot = _Slot(
            value=initial,
            initial=initial,
            reducer=chosen,
            conflict=conflict,
            description=description,
            adapter=TypeAdapter(_unannotated(annotation) if annotation is not None else type(initial)),
            history=[initial],
        )
        self._slots[name] = slot
        return self.param(name)

    @property
    def names(self) -> tuple[str, ...]:
        """Declared parameter names, in declaration order."""
        return tuple(self._slots)

    def _slot(self, name: str) -> _Slot:
        slot = self._slots.get(name)
        if slot is None:
            raise ParamWiringError(f"no local parameter named {name!r}; declared: {sorted(self._slots)}")
        return slot

    def param(self, name: str) -> Param:
        """The parameter's frozen record (LP4).

        Args:
            name: The parameter name.

        Raises:
            ParamWiringError: ``name`` is not declared.
        """
        slot = self._slot(name)
        return Param(
            name=name,
            value=self._dump(slot, slot.value),
            version=slot.version,
            reducer=slot.reducer.name,
            conflict=slot.conflict,
            description=slot.description,
        )

    def snapshot(self) -> dict[str, Param]:
        """Every parameter's record, by name; :meth:`restore` is its inverse (LP4)."""
        return {name: self.param(name) for name in self._slots}

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
        staged: list[tuple[_Slot, Any, Param]] = []  # pyright: ignore[reportExplicitAny]
        for name, record in params.items():
            slot = self._slot(name)
            if record.name != name:
                raise ParamWiringError(
                    f"restore: the record under {name!r} is for parameter {record.name!r}; key each record by its name"
                )
            if record.reducer != slot.reducer.name or record.conflict != slot.conflict:
                raise ParamWiringError(
                    f"local parameter {name!r} is declared with reducer {slot.reducer.name!r} and conflict "
                    f"{slot.conflict!r}, but the record has reducer {record.reducer!r} and conflict "
                    f"{record.conflict!r}; declare the parameter the way it was when the snapshot was taken"
                )
            if record.version < 0:
                raise ParamWiringError(
                    f"local parameter {name!r}: a record's version is never negative, got {record.version}"
                )
            if len(slot.history) != 1:
                raise ParamWiringError(
                    f"local parameter {name!r} has already been written in this host (version {slot.version}); "
                    f"restore into a freshly declared host"
                )
            try:
                value = slot.adapter.validate_python(record.value)
            except Exception as exc:  # noqa: BLE001 (pydantic raises ValidationError; the message names the parameter)
                raise ParamWiringError(
                    f"local parameter {name!r}: the record's value {record.value!r} does not validate as the declared "
                    f"type: {exc}"
                ) from exc
            staged.append((slot, value, record))
        for slot, value, record in staged:
            slot.value = value
            slot.version = record.version
            slot.history = [value]
            slot.history_base = record.version

    # ── Reads ──

    def load(self, name: str) -> tuple[Any, int]:  # pyright: ignore[reportExplicitAny]
        """The current value and its version (LP4).

        Args:
            name: The parameter name.
        """
        slot = self._slot(name)
        return slot.value, slot.version

    def history(self, name: str) -> tuple[Any, ...]:  # pyright: ignore[reportExplicitAny]
        """One value per version, oldest first (LP4); after :meth:`restore` it starts at the restored version.

        Args:
            name: The parameter name.
        """
        return tuple(self._slot(name).history)

    def view(self, name: str) -> ParameterView[Any]:  # pyright: ignore[reportExplicitAny]
        """A fresh, un-emitted ``ParameterView`` of the parameter, for one traced call (LP3).

        The view's ``meta`` carries the version it read and the reducer's name; its
        ``description`` carries the reducer's guidance so the backward model knows what shape of
        feedback the target accepts.

        Args:
            name: The parameter name.
        """
        slot = self._slot(name)
        description = f"{slot.description} {slot.reducer.guidance}".strip()
        return ParameterView(
            value=slot.value,
            name=name,
            backend=cast("Any", self),
            derivation="full",
            requires_grad=True,
            description=description,
            meta={"version": slot.version, "reducer": slot.reducer.name},
            emitted=False,
        )

    # ── ParameterHost ──

    def deserialize_value(self, name: str, raw: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Rehydrate a value from its event representation; the raw value when validation fails."""
        try:
            return self._slot(name).adapter.validate_python(raw)
        except Exception:  # noqa: BLE001 (mirrors MemoryBackend.deserialize_value)
            return raw

    def _is_procedural(self, name: str) -> bool:  # noqa: PLW3201
        """Local parameters are never code parameters."""
        self._slot(name)
        return False

    async def emit_recall(
        self,
        view: ParameterView[Any],  # pyright: ignore[reportExplicitAny]
        coordinator: Coordinator,
        thread_id: ThreadId,
    ) -> None:
        """Emit the recall event for a view consumed by a traced thread (LP3).

        Called by ``AIFunction.trace``. No-op when the view was already emitted, so one logical
        read never lands in two logs. Confirms the append is readable before returning, bounded.
        """
        if view.emitted:
            return
        slot = self._slot(view.name)
        event = ParameterRecalledEvent(
            thread_id=thread_id,
            name=view.name,
            value=self._dump(slot, view.value),
            derivation="full",
            requires_grad=view.requires_grad,
            backend_id=self.backend_id,
            description=view.description,
            meta=dict(view.meta),
        )
        coordinator.append_event(event)
        delay = 0.0
        for _ in range(_CONFIRM_MAX_READS):
            stored = await coordinator.get_events(thread_id, kinds=[EventKind.PARAMETER_RECALLED])
            if any(e.id == event.id for e in stored):
                break
            await asyncio.sleep(delay)
            delay = min(delay + _CONFIRM_BACKOFF_STEP, _CONFIRM_BACKOFF_MAX)
        view.emitted = True

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
        del retrieved
        slot = self._slot(name)
        base = slot.value
        if expected_version is not None and expected_version != slot.version:
            if expected_version < 0 or expected_version > slot.version:
                raise VersionConflictError(name, expected_version, slot.version, self.backend_id)
            if slot.conflict == "refuse":
                raise VersionConflictError(name, expected_version, slot.version, self.backend_id)
            if slot.conflict == "last_writer_wins":
                base = slot.base_for(expected_version)
                if base is None:  # before the restored version: nothing to rebuild from
                    raise VersionConflictError(name, expected_version, slot.version, self.backend_id)
        self._write(slot, slot.reducer.reduce(base, feedback))

    # ── The search-optimizer bridge ──

    def components(self) -> dict[str, str]:
        """The declared parameters as one mapping candidate (LP7): text as is, anything else as canonical JSON."""
        return {name: self._text(slot) for name, slot in self._slots.items()}

    def apply(self, candidate: Mapping[str, str]) -> None:
        """Write a mapping candidate back, bumping the version of each parameter whose value changed (LP7).

        Args:
            candidate: Parameter name to its new text; a non-text parameter's text is parsed as JSON. A
                name whose text equals the current value is left alone, version included.

        Raises:
            ParamWiringError: A name is not declared, or a text does not parse as its parameter's
                type; nothing is written.
        """
        unknown = [name for name in candidate if name not in self._slots]
        if unknown:
            raise ParamWiringError(
                f"candidate names undeclared local parameter(s) {unknown!r}; declared: {sorted(self._slots)}"
            )
        parsed: list[tuple[_Slot, Any]] = []  # pyright: ignore[reportExplicitAny]
        for name, text in candidate.items():
            slot = self._slots[name]
            if isinstance(slot.value, str):
                parsed.append((slot, text))
                continue
            try:
                parsed.append((slot, slot.adapter.validate_python(json.loads(text))))
            except Exception as exc:  # noqa: BLE001 (JSONDecodeError or pydantic ValidationError; both name the text)
                raise ParamWiringError(
                    f"candidate text for local parameter {name!r} does not parse as its {type(slot.value).__name__} "
                    f"value: {text!r} ({exc}). Nothing was written."
                ) from exc
        for slot, value in parsed:
            self._write(slot, value)

    # ── Internals ──

    @staticmethod
    def _write(slot: _Slot, value: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Take ``value`` as the next version; an equal value is not a new version (LP4)."""
        if value == slot.value:
            return
        slot.value = value
        slot.version += 1
        slot.history.append(value)

    @staticmethod
    def _dump(slot: _Slot, value: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        try:
            return slot.adapter.dump_python(value, mode="json")
        except Exception:  # noqa: BLE001 (mirrors MemoryBackend._serialize_value)
            return value

    def _text(self, slot: _Slot) -> str:
        if isinstance(slot.value, str):
            return slot.value
        return json.dumps(self._dump(slot, slot.value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def __repr__(self) -> str:
        """The host id and its declared names."""
        return f"<Local {self.backend_id} {list(self._slots)}>"


# ── The binder ──


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

    RESERVED = ("overrides",)
    """Keyword names ``trace`` owns. A bound parameter sharing one is refused at wiring time."""

    __slots__ = ("_local", "_markers", "_target")

    def __init__(self, target: AIFunction[..., Any] | MethodAgent, local: Local | None = None) -> None:  # pyright: ignore[reportExplicitAny]
        if not isinstance(target, (AIFunction, MethodAgent)):
            raise ParamWiringError(
                f"Bind takes an AIFunction or a MethodAgent, not {type(target).__name__}; wrap the function with "
                f"@ai_function or subclass MethodAgent"
            )
        self._target = target
        self._local = local if local is not None else Local()
        self._markers: dict[str | None, dict[str, Trainable]] = {}
        if isinstance(target, AIFunction):
            self._markers[None] = self._declare(target.prompt_fn, self._label(None))
        else:
            for name in dir(type(target)):
                if is_ai_method(getattr(type(target), name, None)):
                    self._markers[name] = self._declare(getattr(target, name), self._label(name))

    @property
    def target(self) -> AIFunction[..., Any] | MethodAgent:  # pyright: ignore[reportExplicitAny]
        """The bound function or agent. Read-only."""
        return self._target

    @property
    def local(self) -> Local:
        """The host: what ``TextGradOptimizer`` takes in ``backends=[...]``."""
        return self._local

    def bound(self, method: str | None = None) -> dict[str, Trainable]:
        """The marked parameters of the function, or of one method on the agent, in signature order.

        Args:
            method: The ``@ai_method`` name; ``None`` for a bound function.

        Raises:
            ParamWiringError: ``method`` is not an ``@ai_method`` of the agent, or was given for a function.
        """
        if method not in self._markers:
            if isinstance(self._target, AIFunction):
                raise ParamWiringError(f"{self._label(None)} is a function; bound() takes no method name")
            known = sorted(k for k in self._markers if k is not None)
            raise ParamWiringError(f"{_owner_name(self._target)} has no @ai_method named {method!r}; it has {known}")
        return dict(self._markers[method])

    def components(self) -> dict[str, str]:
        """The host's declared parameters as one mapping candidate (LP7)."""
        return self._local.components()

    async def trace(
        self,
        *args: Any,  # pyright: ignore[reportExplicitAny]
        overrides: Mapping[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> Result[Any]:  # pyright: ignore[reportExplicitAny]
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
        config = dict(overrides or {})
        if isinstance(self._target, AIFunction):
            method: str | None = None
            positional = args
            fn = self._target.replace(**config) if config else self._target
            signature = inspect.signature(self._target.prompt_fn)
        else:
            if not args or not isinstance(args[0], str):
                raise ParamWiringError(
                    f"{_owner_name(self._target)}: Bind.trace on an agent takes the method name as its first "
                    f"positional argument, e.g. trace('choose', state=...)"
                )
            method, positional = args[0], args[1:]
            self.bound(method)
            fn = self._target.compiled(method, **config)
            signature = inspect.signature(getattr(self._target, method))
        marked = self._markers[method]
        self._check_no_positional_shadow(method, marked, signature, len(positional))
        filled = dict(kwargs)
        for name in marked:
            if name not in kwargs:
                filled[name] = self._local.view(name)
        return await fn.trace(*positional, **filled)

    # ── Wiring ──

    def _declare(self, fn: Callable[..., Any], label: str) -> dict[str, Trainable]:  # pyright: ignore[reportExplicitAny]
        """Read the markers off ``fn`` and declare each in the host from its default (LP2)."""
        parameters = inspect.signature(fn).parameters
        shadowed = [name for name in self.RESERVED if name in parameters]
        if shadowed:
            raise ParamWiringError(
                f"{label}: parameter(s) {shadowed!r} collide with {self.RESERVED!r}, which are `Bind.trace`'s own "
                f"keywords; a call passing them would fill trace's parameter and leave the function's at its "
                f"default, silently. Rename the parameter."
            )
        marked = trainable_params(fn)
        hints = typing.get_type_hints(fn, include_extras=True)
        for name, marker in marked.items():
            parameter = parameters[name]
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise ParamWiringError(
                    f"{label}: trainable parameter {name!r} is positional-only, and the binder injects by keyword; "
                    f"drop the `/` or move the marker to a keyword-bindable parameter"
                )
            if parameter.default is inspect.Parameter.empty:
                raise ParamWiringError(
                    f"{label}: trainable parameter {name!r} has no default, and the default is the initial value; "
                    f"write {name}: Trainable[...] = <initial>"
                )
            self._local.declare(
                name,
                parameter.default,
                reducer=marker.reducer,
                conflict=marker.conflict,
                description=marker.description,
                annotation=hints.get(name),
            )
        return marked

    def _check_no_positional_shadow(
        self, method: str | None, marked: Mapping[str, Trainable], signature: inspect.Signature, positional: int
    ) -> None:
        """Refuse positional arguments that would land on a marked parameter (positionals bind left to right)."""
        if not positional or not marked:
            return
        kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        names = [name for name, parameter in signature.parameters.items() if parameter.kind in kinds]
        reached = [name for name in names[:positional] if name in marked]
        if reached:
            raise ParamWiringError(
                f"{self._label(method)}: {positional} positional argument(s) would land on trainable parameter(s) "
                f"{reached!r}, because positional arguments bind left to right and cannot skip one. Pass the plain "
                f"arguments by keyword (or supply {reached[0]!r} by keyword to override the injection deliberately)."
            )

    def _label(self, method: str | None) -> str:
        if isinstance(self._target, AIFunction):
            return self._target.name
        return _published_name(self._target, method or "")

    def __repr__(self) -> str:
        """The bound target's name and the host id."""
        who = self._label(None) if isinstance(self._target, AIFunction) else _owner_name(self._target)
        return f"<Bind {who} over {self._local.backend_id!r}>"
