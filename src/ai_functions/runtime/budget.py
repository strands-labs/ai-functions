"""Hierarchical cost reservations: ``Cost``, ``Budget`` and ``Reservation``.

Agents run under a live cost bound. The bound is a tree of budgets: one for the
whole deployment, one per team under it, one per member under that. Work reserves
against a budget before it starts and settles what it spent when it ends, and both
moves run up the tree, so the root's figure is never more than the sum of its parts.

Nothing here is a scheduler and nothing here prices tokens. A :class:`Cost` carries
two axes the runtime can measure or be told, ``tokens`` and ``dollars``, and a
budget constrains whichever of them its limit sets. Token counts arrive from
``TOKEN_USAGE`` events (:meth:`Cost.from_usage`); dollars arrive from a price
table (:mod:`~ai_functions.runtime.prices`), which prices each event by the model
it names, or from a caller that priced a spend itself. A team run is the first
caller: ``Team.run(budget=)`` charges its budget the run's measured spend and
reserves each member ask against it.

Unknown cost is a policy, never a skip. A budget built with ``on_unknown="refuse"``
(the default) refuses a reservation that brings no estimate; one built with
``on_unknown="estimate"`` holds its own ``default_estimate`` instead, and must be
given one at construction.

Invariants:
    BU1 — a ``Cost`` is a frozen value that round-trips through JSON; every quantity
    is finite and non-negative; a limit constrains at least one axis, and a limit
    with neither axis set is refused at construction.

    BU2 — ``reserve`` is all-or-nothing up the chain. A reservation holds the same
    estimate on the budget and on every ancestor; a refusal anywhere raises
    :class:`BudgetExhausted` naming the path of the budget that refused and leaves
    nothing held on any budget. A budget is exhausted when its committed total has
    reached its limit on any constrained axis, and an exhausted budget refuses
    every reservation, a zero one included.

    BU3 — ``settle`` replaces the hold and never raises. A reservation holds its
    estimate until settled, then the actual, on every budget in the chain; a
    later ``settle`` replaces the figure again, so a caller may settle a running
    total. Exhaustion after a settle is a state read through ``exhausted``,
    because the spend already happened.

    BU4 — unknown cost is a policy. ``reserve(None)`` under ``on_unknown="refuse"``
    raises :class:`UnknownCost` naming the path; under ``on_unknown="estimate"`` the
    hold is the budget's ``default_estimate``; ``"estimate"`` without a
    ``default_estimate`` is refused at construction. A child inherits both
    settings from its parent unless told otherwise.

    BU5 — ``charge`` records a spend that already happened and is never refused.
    It holds a reservation settled at the spend on the budget and on every
    ancestor, whether or not the budget is exhausted or the spend takes it past
    its limit; a later ``settle`` replaces the figure, so a caller that measures
    as it goes keeps one charge and settles its running total into it.
"""

from __future__ import annotations

import itertools
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt

from ..types import TokenUsage

UnknownCostPolicy = Literal["refuse", "estimate"]
"""What a budget does with a reservation that brings no estimate (BU4)."""

_AXES: tuple[str, ...] = ("tokens", "dollars")
_counter = itertools.count(1)


class Cost(BaseModel):
    """A spend or a limit on the two axes a live bound can measure.

    In a spend, an axis left ``None`` was not measured and counts as zero when
    summed with a measured one. In a limit, an axis left ``None`` is
    unconstrained. Frozen, because a figure someone reserved against must not
    move under them (BU1).
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    tokens: NonNegativeInt | None = None
    """Model tokens, all four ``TokenUsage`` counters summed; ``None`` when unmeasured or unconstrained."""
    dollars: NonNegativeFloat | None = None
    """Priced cost, from the coordinator's price table or a caller's own; ``None`` when unpriced or unconstrained."""

    @classmethod
    def from_usage(cls, usage: TokenUsage) -> Cost:
        """The token cost of one ``TokenUsage``: every counter summed, dollars unmeasured.

        Args:
            usage: The usage reported by a ``TOKEN_USAGE`` event.

        Returns:
            A ``Cost`` whose ``tokens`` is the sum of the four counters.
        """
        return cls(tokens=usage.input_tokens + usage.output_tokens + usage.cache_read_tokens + usage.cache_write_tokens)

    def __add__(self, other: Cost) -> Cost:
        """Componentwise sum; an axis is ``None`` only when neither side measured it.

        Args:
            other: The cost to add.

        Returns:
            The summed cost.
        """
        return Cost(
            tokens=_sum(self.tokens, other.tokens),
            dollars=_sum(self.dollars, other.dollars),
        )

    def constrains(self) -> tuple[str, ...]:
        """The axes this value sets, in declaration order (empty for ``Cost()``)."""
        return tuple(axis for axis in _AXES if getattr(self, axis) is not None)


def _sum[N: (int, float)](left: N | None, right: N | None) -> N | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right  # type: ignore[return-value]


def _axis(cost: Cost, axis: str) -> float:
    """A cost's figure on one axis, zero when unmeasured."""
    value = getattr(cost, axis)
    return 0.0 if value is None else float(value)


def _total(costs: list[Cost]) -> Cost:
    total = Cost()
    for cost in costs:
        total = total + cost
    return total


# ── Errors ──


class BudgetError(Exception):
    """Base of every budget fault; ``path`` names the budget concerned."""

    def __init__(self, path: str, message: str) -> None:
        self.path: str = path
        super().__init__(message)


class BudgetExhausted(BudgetError):
    """A reservation would take the budget past its limit, or the budget is already spent (BU2).

    Args:
        path: The path of the budget that refused; an ancestor's when the refusal came from up the chain.
        limit: That budget's limit.
        committed: What it held when it refused.
        requested: The estimate it refused.
    """

    def __init__(self, path: str, *, limit: Cost, committed: Cost, requested: Cost) -> None:
        self.limit: Cost = limit
        self.committed: Cost = committed
        self.requested: Cost = requested
        super().__init__(
            path,
            f"budget {path!r} is exhausted: limit {_show(limit)}, committed {_show(committed)}, "
            f"requested {_show(requested)}. Raise the limit, settle open reservations, or stop spawning under it.",
        )


class UnknownCost(BudgetError):
    """A reservation brought no estimate and the budget's policy is ``refuse`` (BU4).

    Args:
        path: The path of the refusing budget.
    """

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            f"budget {path!r} refuses a reservation with no estimate (on_unknown='refuse'). Pass an estimate, "
            f"or build the budget with on_unknown='estimate' and a default_estimate.",
        )


def _show(cost: Cost) -> str:
    parts = [f"{axis}={getattr(cost, axis)}" for axis in cost.constrains()]
    return "(" + ", ".join(parts) + ")" if parts else "(nothing)"


# ── The reservation ──


class Reservation:
    """One hold against a budget and its ancestors: the estimate until settled, the actual after (BU3).

    Built by :meth:`Budget.reserve`; never constructed directly.
    """

    __slots__ = ("_actual", "_budget", "_estimate", "_parent")

    def __init__(self, budget: Budget, estimate: Cost, parent: Reservation | None) -> None:
        self._budget = budget
        self._estimate = estimate
        self._parent = parent
        self._actual: Cost | None = None

    @property
    def budget(self) -> Budget:
        """The budget this reservation was taken on (the deepest one in the chain)."""
        return self._budget

    @property
    def estimate(self) -> Cost:
        """What was held when the reservation was taken."""
        return self._estimate

    @property
    def actual(self) -> Cost | None:
        """What the last ``settle`` recorded, or ``None`` while the estimate still stands."""
        return self._actual

    @property
    def settled(self) -> bool:
        """Whether ``settle`` has been called at least once."""
        return self._actual is not None

    @property
    def hold(self) -> Cost:
        """What this reservation currently counts against its budget: the actual once settled, else the estimate."""
        return self._estimate if self._actual is None else self._actual

    def settle(self, actual: Cost) -> None:
        """Replace the hold with what was really spent, here and on every ancestor (BU3).

        Never raises: the spend is a fact. Read ``budget.exhausted`` afterwards
        to learn whether it took the budget over its limit. A later call replaces
        the figure again, so a caller may settle a running total as usage arrives.

        Args:
            actual: The spend to record.
        """
        self._actual = actual
        if self._parent is not None:
            self._parent.settle(actual)

    def __repr__(self) -> str:
        """The budget path, the estimate and the actual so far."""
        return f"<Reservation on {self._budget.path!r} estimate={_show(self._estimate)} actual={self._actual!r}>"


# ── The budget ──


class Budget:
    """A cost bound with reservations, optionally under a parent (BU1..BU5).

    Args:
        limit: What this budget may commit; every axis it sets is enforced, the
            rest are unconstrained. Must set at least one axis.
        parent: The budget this one nests under. Every reservation here also
            holds on the parent, and on the parent's parent, up to the root.
            :meth:`child` is the usual way to make one.
        on_unknown: What ``reserve(None)`` does: ``"refuse"`` (the default) raises
            :class:`UnknownCost`; ``"estimate"`` holds ``default_estimate``. A child
            inherits the parent's setting when this is left ``None``.
        default_estimate: The hold for a reservation that brings no estimate
            under ``on_unknown="estimate"``; required with that policy, ignored
            under ``"refuse"``. Inherited from the parent when left ``None``.
        name: This budget's segment of the path, for error messages. Defaults to
            ``"budget"`` at the root and ``"child-<n>"`` below it.

    Raises:
        ValueError: ``limit`` constrains no axis; ``on_unknown`` is not one of the
            two policies; or ``on_unknown="estimate"`` without a ``default_estimate``
            anywhere in the chain.
    """

    def __init__(
        self,
        limit: Cost,
        *,
        parent: Budget | None = None,
        on_unknown: UnknownCostPolicy | None = None,
        default_estimate: Cost | None = None,
        name: str | None = None,
    ) -> None:
        if not limit.constrains():
            raise ValueError(
                "Budget(limit=Cost()) constrains nothing; a budget needs at least one of tokens= or dollars= "
                "set on its limit"
            )
        if on_unknown is None:
            on_unknown = "refuse" if parent is None else parent.on_unknown
        if on_unknown not in get_args(UnknownCostPolicy):
            raise ValueError(
                f"Budget(on_unknown={on_unknown!r}) is not a policy; pass 'refuse' (the default) or 'estimate'"
            )
        if default_estimate is None and parent is not None:
            default_estimate = parent.default_estimate
        if on_unknown == "estimate" and default_estimate is None:
            raise ValueError(
                "Budget(on_unknown='estimate') needs a default_estimate= to hold for a reservation that brings none; "
                "pass one, or use on_unknown='refuse'"
            )
        self._limit: Cost = limit
        self._parent: Budget | None = parent
        self._on_unknown: UnknownCostPolicy = on_unknown
        self._default_estimate: Cost | None = default_estimate
        self._name: str = name if name is not None else ("budget" if parent is None else f"child-{next(_counter)}")
        self._holds: list[Reservation] = []

    # ── Identity ──

    @property
    def limit(self) -> Cost:
        """What this budget may commit."""
        return self._limit

    @property
    def parent(self) -> Budget | None:
        """The budget this one nests under, or ``None`` at the root."""
        return self._parent

    @property
    def name(self) -> str:
        """This budget's own segment of :attr:`path`."""
        return self._name

    @property
    def path(self) -> str:
        """The names from the root down to this budget, joined by ``/``."""
        return self._name if self._parent is None else f"{self._parent.path}/{self._name}"

    @property
    def on_unknown(self) -> UnknownCostPolicy:
        """What ``reserve(None)`` does here (BU4)."""
        return self._on_unknown

    @property
    def default_estimate(self) -> Cost | None:
        """The hold for an estimate-less reservation under ``on_unknown="estimate"``."""
        return self._default_estimate

    # ── Figures ──

    @property
    def reserved(self) -> Cost:
        """What open (unsettled) reservations hold."""
        return _total([r.estimate for r in self._holds if not r.settled])

    @property
    def spent(self) -> Cost:
        """What settled reservations recorded."""
        return _total([r.hold for r in self._holds if r.settled])

    @property
    def committed(self) -> Cost:
        """Reserved plus spent: everything counted against the limit."""
        return _total([r.hold for r in self._holds])

    @property
    def remaining(self) -> Cost:
        """The limit less what is committed, floored at zero, on every constrained axis."""
        committed = self.committed
        figures: dict[str, float] = {}
        for axis in self._limit.constrains():
            figures[axis] = max(0.0, _axis(self._limit, axis) - _axis(committed, axis))
        return Cost(
            tokens=int(figures["tokens"]) if "tokens" in figures else None,
            dollars=figures.get("dollars"),
        )

    @property
    def exhausted(self) -> bool:
        """Whether the committed total has reached the limit on any constrained axis (BU2)."""
        return self._would_exhaust(Cost(), strict=False)

    def _would_exhaust(self, extra: Cost, *, strict: bool) -> bool:
        committed = self.committed
        for axis in self._limit.constrains():
            total = _axis(committed, axis) + _axis(extra, axis)
            limit = _axis(self._limit, axis)
            if (total > limit) if strict else (total >= limit):
                return True
        return False

    # ── Verbs ──

    def reserve(self, estimate: Cost | None) -> Reservation:
        """Hold ``estimate`` here and on every ancestor, or refuse and hold nothing (BU2, BU4).

        Args:
            estimate: What the work is expected to cost; ``None`` when unknown,
                which the ``on_unknown`` policy decides.

        Returns:
            The reservation, to be settled when the work ends.

        Raises:
            UnknownCost: ``estimate`` is ``None`` and the policy is ``"refuse"``.
            BudgetExhausted: This budget or an ancestor cannot hold the estimate;
                the error's ``path`` names which.
        """
        if estimate is None:
            if self._on_unknown == "refuse" or self._default_estimate is None:
                raise UnknownCost(self.path)
            estimate = self._default_estimate
        if self.exhausted or self._would_exhaust(estimate, strict=True):
            raise BudgetExhausted(self.path, limit=self._limit, committed=self.committed, requested=estimate)
        parent_hold = self._parent.reserve(estimate) if self._parent is not None else None
        reservation = Reservation(self, estimate, parent_hold)
        self._holds.append(reservation)
        return reservation

    def charge(self, spent: Cost) -> Reservation:
        """Record a spend that already happened, here and on every ancestor; never refused (BU5).

        The spend is a fact, as it is for :meth:`Reservation.settle`, so an
        exhausted budget takes it, and so does one it takes past its limit.

        Args:
            spent: What was spent; ``Cost()`` opens a charge to settle into later.

        Returns:
            A reservation already settled at ``spent``; settle it again to
            replace the figure with a running total.
        """
        parent_hold = self._parent.charge(spent) if self._parent is not None else None
        reservation = Reservation(self, spent, parent_hold)
        reservation.settle(spent)
        self._holds.append(reservation)
        return reservation

    def child(
        self,
        limit: Cost,
        *,
        name: str | None = None,
        on_unknown: UnknownCostPolicy | None = None,
        default_estimate: Cost | None = None,
    ) -> Budget:
        """A budget nested under this one; its reservations hold here too.

        Args:
            limit: The child's own limit.
            name: The child's path segment; ``child-<n>`` when omitted.
            on_unknown: The child's policy; inherited when omitted.
            default_estimate: The child's default hold; inherited when omitted.

        Returns:
            The child budget, with ``parent`` set to this one.
        """
        return Budget(limit, parent=self, on_unknown=on_unknown, default_estimate=default_estimate, name=name)

    def __repr__(self) -> str:
        """The path, the limit and what is committed."""
        return f"<Budget {self.path!r} limit={_show(self._limit)} committed={_show(self.committed)}>"


__all__ = [
    "Budget",
    "BudgetError",
    "BudgetExhausted",
    "Cost",
    "Reservation",
    "UnknownCost",
    "UnknownCostPolicy",
]
