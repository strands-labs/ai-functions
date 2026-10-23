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

import re
from collections.abc import Mapping
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)

from ..types import TokenUsage

_PER_TOKEN: float = 1e-6
_GEO_PREFIX = re.compile(r"^(?:us-gov|us|eu|jp|au|apac|ca)\.")
_VERSION_SUFFIX = re.compile(r"-v\d+(?::\d+)?$")
_DATE_SUFFIX = re.compile(r"-\d{8}$")
_CONTEXT_SUFFIX = re.compile(r"\[[^\]]*\]$")


class ModelPrice(BaseModel):
    """One model's rates, in dollars per million tokens (PR1).

    ``cache_read`` is the cache-hit rate and ``cache_write`` the five-minute
    cache-write rate; both are billed separately from ``input``.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    input: NonNegativeFloat
    """Uncached input tokens."""
    output: NonNegativeFloat
    """Output tokens, thinking included."""
    cache_read: NonNegativeFloat
    """Input tokens read from the prompt cache."""
    cache_write: NonNegativeFloat
    """Input tokens written to the five-minute prompt cache."""
    long_prompt_tokens: PositiveInt | None = None
    """The prompt length, in tokens, above which ``long_prompt`` applies (PR5); ``None`` for one set of rates."""
    long_prompt: ModelPrice | None = None
    """The rates for a prompt over ``long_prompt_tokens`` (PR5); ``None`` for one set of rates."""

    @model_validator(mode="after")
    def _long_prompt_pair(self) -> ModelPrice:
        """Refuse a threshold without long-prompt rates, or rates without a threshold."""
        if (self.long_prompt_tokens is None) != (self.long_prompt is None):
            raise ValueError("long_prompt_tokens and long_prompt are set together or not at all")
        if self.long_prompt is not None and self.long_prompt.long_prompt is not None:
            raise ValueError("long_prompt rates cannot carry a further tier")
        return self

    def dollars(self, usage: TokenUsage, *, premium: float = 1.0) -> float:
        """The dollars ``usage`` cost at these rates, or at ``long_prompt`` for a prompt over the threshold (PR5).

        Args:
            usage: The token counts to price.
            premium: A multiplier on every rate, such as a regional endpoint's.

        Returns:
            The cost in dollars.
        """
        rates = self
        if self.long_prompt is not None and self.long_prompt_tokens is not None:
            prompt = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
            if prompt > self.long_prompt_tokens:
                rates = self.long_prompt
        return (
            usage.input_tokens * rates.input
            + usage.output_tokens * rates.output
            + usage.cache_read_tokens * rates.cache_read
            + usage.cache_write_tokens * rates.cache_write
        ) * (_PER_TOKEN * premium)


class PriceTable(BaseModel):
    """Model id to :class:`ModelPrice`, with the id normalization of PR2 and the Bedrock premium of PR3.

    Frozen, and ``prices`` is a read-only view of a copy the table owns, so a
    coordinator's prices never move under a budget being settled: neither
    the caller's mapping nor the shared :data:`PRICES` can be edited in place,
    and a different price is a different table (:meth:`with_prices`).
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    prices: Mapping[str, ModelPrice] = Field(default_factory=lambda: MappingProxyType({}))
    """List prices keyed by first-party model id, or by any exact id a caller prices on its own; read-only."""
    regional_premium: float = Field(default=1.1, ge=1.0)
    """Multiplier for a Bedrock regional endpoint or geographic inference profile over the list price (PR3)."""

    @field_validator("prices", mode="after")
    @classmethod
    def _read_only(cls, prices: Mapping[str, ModelPrice]) -> Mapping[str, ModelPrice]:
        """Own a copy of ``prices`` and expose it read-only."""
        return MappingProxyType(dict(prices))

    @field_serializer("prices")
    def _plain(self, prices: Mapping[str, ModelPrice]) -> dict[str, ModelPrice]:
        """Serialize the read-only view as the plain mapping it wraps."""
        return dict(prices)

    def with_prices(self, prices: Mapping[str, ModelPrice]) -> PriceTable:
        """A new table with this one's entries and ``prices``, replacing any entry of the same id.

        Args:
            prices: The entries to add, keyed by model id.

        Returns:
            The extended table; this one is unchanged.
        """
        return PriceTable(prices={**self.prices, **prices}, regional_premium=self.regional_premium)

    def lookup(self, model: str) -> tuple[ModelPrice, float] | None:
        """The price ``model`` is billed at and the premium on it, or ``None`` when the table has no entry (PR2, PR3).

        Args:
            model: A model id as a thread reported it: first-party, a Bedrock id
                or inference profile, or an ARN naming one.

        Returns:
            ``(price, premium)``, or ``None`` for a model the table does not know.
        """
        exact = self.prices.get(model)
        if exact is not None:
            return exact, 1.0
        name = _CONTEXT_SUFFIX.sub("", model.rsplit("/", 1)[-1])
        premium = 1.0
        if name.startswith("global."):
            name = name.removeprefix("global.")
        elif (geo := _GEO_PREFIX.match(name)) is not None:
            name = name[geo.end() :]
            premium = self.regional_premium
        elif name.startswith("anthropic."):
            premium = self.regional_premium
        name = _DATE_SUFFIX.sub("", _VERSION_SUFFIX.sub("", name.removeprefix("anthropic.")))
        price = self.prices.get(name)
        return None if price is None else (price, premium)

    def dollars(self, usage: TokenUsage, model: str | None) -> float | None:
        """The dollars ``usage`` cost on ``model``, or ``None`` when it cannot be priced (PR4).

        Args:
            usage: The token counts to price.
            model: The model that spent them; ``None`` when the thread did not say.

        Returns:
            The cost in dollars, or ``None`` for no model or one the table does not know.
        """
        if model is None:
            return None
        found = self.lookup(model)
        if found is None:
            return None
        price, premium = found
        return price.dollars(usage, premium=premium)


def _price(
    input_rate: float,
    output_rate: float,
    cache_read_rate: float,
    *,
    long_prompt: tuple[int, ModelPrice] | None = None,
) -> ModelPrice:
    """A Claude list price, whose five-minute cache write is 1.25 times its input rate.

    ``long_prompt`` is the threshold and the rates for a model priced by prompt length (PR5).
    """
    return ModelPrice(
        input=input_rate,
        output=output_rate,
        cache_read=cache_read_rate,
        cache_write=input_rate * 1.25,
        long_prompt_tokens=None if long_prompt is None else long_prompt[0],
        long_prompt=None if long_prompt is None else long_prompt[1],
    )


PRICES: PriceTable = PriceTable(
    prices={
        "claude-fable-5-1": _price(10.0, 50.0, 0.25),
        "claude-mythos-5-1": _price(10.0, 50.0, 0.25),
        "claude-fable-5": _price(10.0, 50.0, 1.0),
        "claude-mythos-5": _price(10.0, 50.0, 1.0),
        "claude-opus-5-5": _price(4.0, 20.0, 0.2),
        "claude-opus-5": _price(5.0, 25.0, 0.5),
        "claude-opus-4-8": _price(5.0, 25.0, 0.5),
        "claude-opus-4-7": _price(5.0, 25.0, 0.5),
        "claude-opus-4-6": _price(5.0, 25.0, 0.5),
        "claude-opus-4-5": _price(5.0, 25.0, 0.5),
        "claude-sonnet-5-5": _price(2.0, 10.0, 0.1),
        "claude-sonnet-5": _price(2.0, 10.0, 0.2),
        "claude-sonnet-4-6": _price(3.0, 15.0, 0.3),
        "claude-sonnet-4-5": _price(3.0, 15.0, 0.3),
        "claude-haiku-5-5": _price(0.1, 0.5, 0.01, long_prompt=(100_000, _price(0.5, 2.5, 0.05))),
        "claude-haiku-4-5": _price(1.0, 5.0, 0.1),
    }
)
"""List prices for the current Claude models, per million tokens, as published on 2026-10-21.

The first-party Claude API and Bedrock's global endpoint charge these rates;
Bedrock's regional endpoints and geographic profiles charge ``regional_premium``
times them (PR3). Pass a replacement or an extension to
``InMemoryCoordinator(prices=...)`` for any other model or rate.
"""
