"""The static price table, and the table a coordinator carries.

- a ``ModelPrice`` prices all four ``TokenUsage`` counters per million tokens and is frozen (PR1);
- ``lookup`` reads first-party ids, Bedrock ids, inference profiles, ARNs, version, date and context suffixes (PR2);
- a Bedrock regional endpoint or geographic profile costs the regional premium, ``global.`` the list price (PR3);
- a model the table does not know has no price, never a zero one (PR4);
- ``InMemoryCoordinator(prices=)`` is the table a caller folding a run's usage reads, ``PRICES`` by default.
"""

from __future__ import annotations

import pytest

from ai_functions.runtime import PRICES, InMemoryCoordinator, ModelPrice, PriceTable
from ai_functions.types import TokenUsage

MILLION = 1_000_000


class TestModelPrice:
    def test_every_counter_is_priced_per_million_tokens(self) -> None:
        price = ModelPrice(input=4.0, output=20.0, cache_read=0.2, cache_write=5.0)
        usage = TokenUsage(
            input_tokens=MILLION, output_tokens=MILLION, cache_read_tokens=MILLION, cache_write_tokens=MILLION
        )
        assert price.dollars(usage) == pytest.approx(29.2)
        assert price.dollars(usage, premium=1.1) == pytest.approx(32.12)

    def test_a_price_is_frozen_and_refuses_a_negative_rate(self) -> None:
        price = ModelPrice(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)
        with pytest.raises(Exception):  # noqa: B017, PT011 — pydantic's frozen error type is its own business
            price.input = 2.0  # type: ignore[misc]
        with pytest.raises(ValueError):
            ModelPrice(input=-1.0, output=5.0, cache_read=0.1, cache_write=1.25)


class TestPriceTable:
    def test_the_current_claude_models_carry_their_list_prices(self) -> None:
        one_each = TokenUsage(input_tokens=MILLION, output_tokens=MILLION)
        expected = {
            "claude-fable-5-1": 60.0,
            "claude-opus-5-5": 24.0,
            "claude-opus-5": 30.0,
            "claude-sonnet-5": 12.0,
            "claude-sonnet-4-6": 18.0,
            "claude-haiku-4-5": 6.0,
        }
        for model, dollars in expected.items():
            assert PRICES.dollars(one_each, model) == pytest.approx(dollars), model

    def test_cache_rates_follow_each_models_own_multiplier(self) -> None:
        reads = TokenUsage(cache_read_tokens=MILLION)
        writes = TokenUsage(cache_write_tokens=MILLION)
        assert PRICES.dollars(reads, "claude-fable-5-1") == pytest.approx(0.25)
        assert PRICES.dollars(reads, "claude-opus-5-5") == pytest.approx(0.2)
        assert PRICES.dollars(reads, "claude-sonnet-5") == pytest.approx(0.2)
        assert PRICES.dollars(writes, "claude-opus-5-5") == pytest.approx(5.0), "a five-minute write is 1.25x input"

    @pytest.mark.parametrize(
        ("model", "premium"),
        [
            ("claude-opus-5-5", 1.0),
            ("claude-opus-5-5[1m]", 1.0),
            ("global.anthropic.claude-opus-5-5", 1.0),
            ("global.anthropic.claude-opus-5-5-v1:0", 1.0),
            ("us.anthropic.claude-opus-5-5", 1.1),
            ("eu.anthropic.claude-opus-5-5-v1:0", 1.1),
            ("anthropic.claude-opus-5-5", 1.1),
            ("arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-opus-5-5", 1.0),
        ],
    )
    def test_bedrock_ids_normalize_to_the_first_party_price_with_the_regional_premium(
        self, model: str, premium: float
    ) -> None:
        found = PRICES.lookup(model)
        assert found is not None, model
        price, got = found
        assert price == PRICES.prices["claude-opus-5-5"] and got == pytest.approx(premium)

    def test_a_dated_id_prices_as_its_alias(self) -> None:
        found = PRICES.lookup("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        assert found is not None and found[0] == PRICES.prices["claude-haiku-4-5"]

    def test_an_unknown_model_or_none_has_no_price_rather_than_a_free_one(self) -> None:
        usage = TokenUsage(input_tokens=MILLION)
        assert PRICES.lookup("gpt-5-codex") is None
        assert PRICES.dollars(usage, "gpt-5-codex") is None
        assert PRICES.dollars(usage, None) is None

    def test_a_caller_extends_or_overrides_the_table_without_touching_the_default(self) -> None:
        mine = ModelPrice(input=1.25, output=10.0, cache_read=0.125, cache_write=0.0)
        cheaper = ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
        table = PRICES.with_prices({"gpt-5-codex": mine, "claude-opus-5-5": cheaper})
        usage = TokenUsage(input_tokens=MILLION)
        assert table.dollars(usage, "gpt-5-codex") == pytest.approx(1.25)
        assert table.dollars(usage, "global.anthropic.claude-opus-5-5") == pytest.approx(3.0)
        assert PRICES.dollars(usage, "claude-opus-5-5") == pytest.approx(4.0), "the default table is unchanged"
        assert PriceTable(regional_premium=1.0).dollars(usage, "claude-opus-5-5") is None, (
            "an empty table prices nothing"
        )

    def test_an_exact_entry_wins_over_normalization(self) -> None:
        exact = ModelPrice(input=9.0, output=9.0, cache_read=9.0, cache_write=9.0)
        table = PRICES.with_prices({"us.anthropic.claude-opus-5-5": exact})
        assert table.lookup("us.anthropic.claude-opus-5-5") == (exact, 1.0)


class TestCoordinatorTable:
    def test_a_coordinator_carries_the_default_table_unless_given_one(self) -> None:
        table = PRICES.with_prices(
            {"gpt-5-codex": ModelPrice(input=1.25, output=10.0, cache_read=0.125, cache_write=0.0)}
        )
        assert InMemoryCoordinator().prices is PRICES
        assert InMemoryCoordinator(prices=table).prices is table


def test_a_tables_prices_cannot_be_edited_in_place_by_anyone() -> None:
    theirs = {"house-model": ModelPrice(input=1.0, output=1.0, cache_read=1.0, cache_write=1.0)}
    table = PriceTable(prices=theirs)
    theirs["house-model"] = ModelPrice(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0)
    with pytest.raises(TypeError):
        PRICES.prices["claude-opus-5"] = ModelPrice(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0)  # type: ignore[index]
    extended = PRICES.with_prices(theirs)

    assert table.lookup("house-model") == (ModelPrice(input=1.0, output=1.0, cache_read=1.0, cache_write=1.0), 1.0)
    assert "house-model" not in PRICES.prices and "house-model" in extended.prices
    assert PriceTable.model_validate_json(table.model_dump_json()) == table


# ── PR5: a model priced by prompt length ──


def test_the_table_prices_sonnet_5_5_at_its_list_rates() -> None:
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000)
    assert PRICES.dollars(usage, "claude-sonnet-5-5") == pytest.approx(2.0 + 10.0 + 0.1)
    assert PRICES.dollars(usage, "global.anthropic.claude-sonnet-5-5") == pytest.approx(12.1)


def test_haiku_5_5_pays_the_long_prompt_rates_only_over_100k_prompt_tokens() -> None:
    at_threshold = TokenUsage(input_tokens=60_000, cache_read_tokens=40_000, output_tokens=1_000)
    assert PRICES.dollars(at_threshold, "claude-haiku-5-5") == pytest.approx(
        (60_000 * 0.1 + 40_000 * 0.01 + 1_000 * 0.5) / 1_000_000
    )
    over = TokenUsage(input_tokens=60_000, cache_read_tokens=40_000, cache_write_tokens=1, output_tokens=1_000)
    assert PRICES.dollars(over, "claude-haiku-5-5") == pytest.approx(
        (60_000 * 0.5 + 40_000 * 0.05 + 1 * 0.625 + 1_000 * 2.5) / 1_000_000
    )


def test_the_long_prompt_tier_keeps_the_regional_premium() -> None:
    over = TokenUsage(input_tokens=200_000)
    assert PRICES.dollars(over, "us.anthropic.claude-haiku-5-5") == pytest.approx(200_000 * 0.5 * 1.1 / 1_000_000)


@pytest.mark.parametrize(
    "fields",
    [
        {"long_prompt_tokens": 100_000},
        {"long_prompt": ModelPrice(input=1.0, output=1.0, cache_read=0.1, cache_write=1.25)},
    ],
)
def test_a_long_prompt_threshold_and_its_rates_come_together(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="set together"):
        ModelPrice(input=1.0, output=1.0, cache_read=0.1, cache_write=1.25, **fields)  # pyright: ignore[reportArgumentType]


def test_long_prompt_rates_cannot_nest_a_further_tier() -> None:
    inner = ModelPrice(
        input=2.0,
        output=2.0,
        cache_read=0.2,
        cache_write=2.5,
        long_prompt_tokens=10,
        long_prompt=ModelPrice(input=3.0, output=3.0, cache_read=0.3, cache_write=3.75),
    )
    with pytest.raises(ValueError, match="further tier"):
        ModelPrice(input=1.0, output=1.0, cache_read=0.1, cache_write=1.25, long_prompt_tokens=5, long_prompt=inner)
