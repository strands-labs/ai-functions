"""Static per-model prices: what a ``TokenUsage`` cost in dollars on the model that spent it.

A team settles a budget's ``dollars`` axis from this table: every
``TOKEN_USAGE`` event of its run that names its model is priced here, and the
dollars join the tokens in the run's spend. The table is data in this repository,
read from Anthropic's published list prices and the Amazon Bedrock price list;
nothing here calls a pricing service. A caller whose prices differ (a negotiated
rate, another provider's models, a price change this table has not caught up
with) passes its own :class:`PriceTable` to ``InMemoryCoordinator(prices=...)``,
usually :data:`PRICES` extended with :meth:`PriceTable.with_prices`.

Invariants:
    PR1 — a :class:`ModelPrice` is a frozen value in dollars per million tokens,
    one finite, non-negative rate for each of the four ``TokenUsage`` counters.
    ``cache_write`` is the five-minute cache-write rate: a ``TokenUsage`` does
    not say which cache lifetime was written, and the five-minute write is the
    default on every Claude API.

    PR2 — :meth:`PriceTable.lookup` matches a model id exactly first, then its
    normalized form: the path of an ARN, a bracketed context-window suffix such
    as ``[1m]``, a Bedrock geographic or ``global.`` prefix, the ``anthropic.``
    provider prefix, a ``-v<N>[:<M>]`` version suffix and a ``-<YYYYMMDD>`` date
    suffix are dropped, so ``us.anthropic.claude-haiku-4-5-20251001-v1:0``
    prices as ``claude-haiku-4-5``. The 1M-token window costs the standard rates
    on every model this table lists, so the context suffix changes nothing.

    PR3 — Bedrock endpoint pricing. A geographic inference profile (``us.``,
    ``eu.``, ``jp.``, ``au.``, ``apac.``, ``ca.``, ``us-gov.``) or a bare
    in-region ``anthropic.`` id costs the table's ``regional_premium`` times the
    list price (1.1 by default, the 10% Bedrock charges over its global
    endpoint); a ``global.`` profile and a first-party id cost the list price.
    Of the two readings of a bare ``anthropic.`` id this is the higher one, so a
    budget reads it as no cheaper than it may be.

    PR4 — a model the table does not know has no price. :meth:`PriceTable.dollars`
    returns ``None`` for it, and for usage that names no model, never ``0.0``:
    an unpriced spend is not a free one, and what it does to a budget is the
    caller's policy (a team refuses further asks under a budget that bounds
    dollars).

    PR5 — a model priced by prompt length carries a second set of rates. A
    :class:`ModelPrice` with ``long_prompt`` set charges those rates for usage
    whose prompt (input, cache-read and cache-write tokens together) is over
    ``long_prompt_tokens``, and its own rates otherwise. A ``TokenUsage`` from one
    model call is priced exactly; one that sums several calls pays the long rates
    when the sum is over the threshold, which can overstate the cost and never
    understates it.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from ..types import TokenUsage

class ModelPrice(BaseModel):
    """One model's rates, in dollars per million tokens (PR1).

    ``cache_read`` is the cache-hit rate and ``cache_write`` the five-minute
    cache-write rate; both are billed separately from ``input``.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    input: float
    output: float
    cache_read: float
    cache_write: float
    long_prompt_tokens: int | None = None
    long_prompt: ModelPrice | None = None

    def dollars(self, usage: TokenUsage, *, premium: float = 1.0) -> float:
        """The dollars ``usage`` cost at these rates, or at ``long_prompt`` for a prompt over the threshold (PR5).

        Args:
            usage: The token counts to price.
            premium: A multiplier on every rate, such as a regional endpoint's.

        Returns:
            The cost in dollars.
        """
        ...

class PriceTable(BaseModel):
    """Model id to :class:`ModelPrice`, with the id normalization of PR2 and the Bedrock premium of PR3.

    Frozen, and ``prices`` is a read-only view of a copy the table owns, so a
    coordinator's prices never move under a budget being settled: neither
    the caller's mapping nor the shared :data:`PRICES` can be edited in place,
    and a different price is a different table (:meth:`with_prices`).
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    prices: Mapping[str, ModelPrice] = ...
    regional_premium: float = 1.1

    def with_prices(self, prices: Mapping[str, ModelPrice]) -> PriceTable:
        """A new table with this one's entries and ``prices``, replacing any entry of the same id.

        Args:
            prices: The entries to add, keyed by model id.

        Returns:
            The extended table; this one is unchanged.
        """
        ...

    def lookup(self, model: str) -> tuple[ModelPrice, float] | None:
        """The price ``model`` is billed at and the premium on it, or ``None`` when the table has no entry (PR2, PR3).

        Args:
            model: A model id as a thread reported it: first-party, a Bedrock id
                or inference profile, or an ARN naming one.

        Returns:
            ``(price, premium)``, or ``None`` for a model the table does not know.
        """
        ...

    def dollars(self, usage: TokenUsage, model: str | None) -> float | None:
        """The dollars ``usage`` cost on ``model``, or ``None`` when it cannot be priced (PR4).

        Args:
            usage: The token counts to price.
            model: The model that spent them; ``None`` when the thread did not say.

        Returns:
            The cost in dollars, or ``None`` for no model or one the table does not know.
        """
        ...

PRICES: PriceTable
"""List prices for the current Claude models, per million tokens, as published on 2026-10-21.

The first-party Claude API and Bedrock's global endpoint charge these rates;
Bedrock's regional endpoints and geographic profiles charge ``regional_premium``
times them (PR3). Pass a replacement or an extension to
``InMemoryCoordinator(prices=...)`` for any other model or rate.
"""
