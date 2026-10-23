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

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..types import TokenUsage

UnknownCostPolicy = Literal["refuse", "estimate"]

class Cost(BaseModel):
    """A spend or a limit on the two axes a live bound can measure.

    In a spend, an axis left ``None`` was not measured and counts as zero when
    summed with a measured one. In a limit, an axis left ``None`` is
    unconstrained. Frozen, because a figure someone reserved against must not
    move under them (BU1).
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    tokens: int | None = None
    dollars: float | None = None

    @classmethod
    def from_usage(cls, usage: TokenUsage) -> Cost:
        """The token cost of one ``TokenUsage``: every counter summed, dollars unmeasured.

        Args:
            usage: The usage reported by a ``TOKEN_USAGE`` event.

        Returns:
            A ``Cost`` whose ``tokens`` is the sum of the four counters.
        """
        ...

    def __add__(self, other: Cost) -> Cost:
        """Componentwise sum; an axis is ``None`` only when neither side measured it.

        Args:
            other: The cost to add.

        Returns:
            The summed cost.
        """
        ...

    def constrains(self) -> tuple[str, ...]:
        """The axes this value sets, in declaration order (empty for ``Cost()``)."""
        ...

class BudgetError(Exception):
    """Base of every budget fault; ``path`` names the budget concerned."""

    path: str

    def __init__(self, path: str, message: str) -> None: ...

class BudgetExhausted(BudgetError):
    """A reservation would take the budget past its limit, or the budget is already spent (BU2).

    Args:
        path: The path of the budget that refused; an ancestor's when the refusal came from up the chain.
        limit: That budget's limit.
        committed: What it held when it refused.
        requested: The estimate it refused.
    """

    limit: Cost
    committed: Cost
    requested: Cost

    def __init__(self, path: str, *, limit: Cost, committed: Cost, requested: Cost) -> None: ...

class UnknownCost(BudgetError):
    """A reservation brought no estimate and the budget's policy is ``refuse`` (BU4).

    Args:
        path: The path of the refusing budget.
    """

    def __init__(self, path: str) -> None: ...

class Reservation:
    """One hold against a budget and its ancestors: the estimate until settled, the actual after (BU3).

    Built by :meth:`Budget.reserve`; never constructed directly.
    """

    def __init__(self, budget: Budget, estimate: Cost, parent: Reservation | None) -> None: ...
    @property
    def budget(self) -> Budget:
        """The budget this reservation was taken on (the deepest one in the chain)."""
        ...

    @property
    def estimate(self) -> Cost:
        """What was held when the reservation was taken."""
        ...

    @property
    def actual(self) -> Cost | None:
        """What the last ``settle`` recorded, or ``None`` while the estimate still stands."""
        ...

    @property
    def settled(self) -> bool:
        """Whether ``settle`` has been called at least once."""
        ...

    @property
    def hold(self) -> Cost:
        """What this reservation currently counts against its budget: the actual once settled, else the estimate."""
        ...

    def settle(self, actual: Cost) -> None:
        """Replace the hold with what was really spent, here and on every ancestor (BU3).

        Never raises: the spend is a fact. Read ``budget.exhausted`` afterwards
        to learn whether it took the budget over its limit. A later call replaces
        the figure again, so a caller may settle a running total as usage arrives.

        Args:
            actual: The spend to record.
        """
        ...

    def __repr__(self) -> str: ...

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
    ) -> None: ...
    @property
    def limit(self) -> Cost:
        """What this budget may commit."""
        ...

    @property
    def parent(self) -> Budget | None:
        """The budget this one nests under, or ``None`` at the root."""
        ...

    @property
    def name(self) -> str:
        """This budget's own segment of :attr:`path`."""
        ...

    @property
    def path(self) -> str:
        """The names from the root down to this budget, joined by ``/``."""
        ...

    @property
    def on_unknown(self) -> UnknownCostPolicy:
        """What ``reserve(None)`` does here (BU4)."""
        ...

    @property
    def default_estimate(self) -> Cost | None:
        """The hold for an estimate-less reservation under ``on_unknown="estimate"``."""
        ...

    @property
    def reserved(self) -> Cost:
        """What open (unsettled) reservations hold."""
        ...

    @property
    def spent(self) -> Cost:
        """What settled reservations recorded."""
        ...

    @property
    def committed(self) -> Cost:
        """Reserved plus spent: everything counted against the limit."""
        ...

    @property
    def remaining(self) -> Cost:
        """The limit less what is committed, floored at zero, on every constrained axis."""
        ...

    @property
    def exhausted(self) -> bool:
        """Whether the committed total has reached the limit on any constrained axis (BU2)."""
        ...

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
        ...

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
        ...

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
        ...

    def __repr__(self) -> str: ...

__all__ = [
    "Budget",
    "BudgetError",
    "BudgetExhausted",
    "Cost",
    "Reservation",
    "UnknownCost",
    "UnknownCostPolicy",
]
