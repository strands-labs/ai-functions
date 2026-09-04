"""Tests for the economics module: search core, beliefs, and end-to-end routing."""

from __future__ import annotations

import math

import pytest

from ai_functions import ai_function
from ai_functions.ai_thread import AIFunctionError, PostConditionResult
from ai_functions.experimental.economics import (
    Abstained,
    AttemptRecord,
    Beliefs,
    BudgetExceeded,
    Candidate,
    CandidatesExhausted,
    EconomicFunction,
    EmpiricalBeliefs,
    PricedModel,
    Prices,
    RecordId,
    TaskView,
    attempts,
    decisions,
    routed,
    spend,
)
from ai_functions.experimental.economics.search import (
    Bernoulli,
    Categorical,
    Estimate,
    Exhaustive,
    Gaussian,
    Greedy,
    ReservationPricePolicy,
    RewardDistribution,
    Search,
)
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn

# ── Fixtures ──────────────────────────────────────────────────────

CHEAP_PRICES = Prices(input=1.0, output=1.0)
STRONG_PRICES = Prices(input=1.0, output=10.0)


def _task(prompt: str = "p", **args: object) -> TaskView:
    return TaskView(prompt=prompt, arguments=dict(args))


# ══════════════════════════════════════════════════════════════════
# Distributions and reservation prices
# ══════════════════════════════════════════════════════════════════


class TestReservationPrice:
    @pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
    @pytest.mark.parametrize("cost", [0.001, 0.01, 0.05])
    def test_bernoulli_closed_form_matches_bisection(self, p, cost):
        b = Bernoulli(p=p, value=1.0)
        closed = b.reservation_price(cost)
        bisected = RewardDistribution.reservation_price(b, cost)
        assert abs(closed - bisected) < 1e-6

    @pytest.mark.parametrize("mu", [0.05, 0.5])
    @pytest.mark.parametrize("sigma", [0.01, 0.1])
    @pytest.mark.parametrize("cost", [0.001, 0.01])
    def test_gaussian_self_consistency(self, mu, sigma, cost):
        """E[(R - g)_+] = cost must hold at the returned g."""
        g = Gaussian(mu=mu, sigma=sigma)
        gg = g.reservation_price(cost)
        assert abs(g.expected_improvement(gg) - cost) < 1e-5

    def test_zero_cost_is_infinite(self):
        assert Bernoulli(p=0.5, value=1.0).reservation_price(0.0) == math.inf
        assert Gaussian(mu=0.5, sigma=0.1).reservation_price(0.0) == math.inf

    def test_categorical_mean_and_improvement(self):
        cat = Categorical(values=(0.0, 1.0), probs=(0.25, 0.75))
        assert cat.mean() == pytest.approx(0.75)
        # E[(R - 0)_+] = 0.75 * 1.0
        assert cat.expected_improvement(0.0) == pytest.approx(0.75)

    def test_estimate_net_value(self):
        e = Estimate(dist=Bernoulli(p=0.4, value=0.10), cost=0.01)
        assert e.net_value() == pytest.approx(0.4 * 0.10 - 0.01)

    def test_bernoulli_rejects_bad_p(self):
        with pytest.raises(ValueError, match="p must be"):
            Bernoulli(p=1.5, value=1.0)

    def test_estimate_rejects_negative_cost(self):
        with pytest.raises(ValueError, match="cost must be"):
            Estimate(dist=Bernoulli(p=0.5, value=1.0), cost=-0.01)


# ══════════════════════════════════════════════════════════════════
# Search loop and policies
# ══════════════════════════════════════════════════════════════════


class TestSearch:
    def _estimates(self):
        return {
            "cheap": Estimate(Bernoulli(0.6, 0.10), 0.002),
            "strong": Estimate(Bernoulli(0.95, 0.10), 0.02),
        }

    def test_escalation_order_and_stop(self):
        s = Search(self._estimates(), budget=0.25, policy=ReservationPricePolicy())
        assert s.next() == "cheap"  # higher reservation price
        s.observe("cheap", reward=0.0, cost=0.002)
        assert s.next() == "strong"
        s.observe("strong", reward=0.10, cost=0.02)
        assert s.next() is None  # best (0.10) tops every reservation price
        assert s.best == pytest.approx(0.10)
        assert s.spent == pytest.approx(0.022)

    def test_max_tries_exhausts_labels(self):
        s = Search({"only": Estimate(Bernoulli(0.5, 1.0), 0.01)}, budget=1.0, max_tries=1)
        assert s.next() == "only"
        s.observe("only", reward=0.0, cost=0.01)
        assert s.next() is None  # single try used up

    def test_budget_blocks_unaffordable(self):
        s = Search(self._estimates(), budget=0.005)
        # only cheap (cost 0.002) fits; strong (0.02) is filtered out
        assert s.next() == "cheap"
        s.observe("cheap", reward=0.0, cost=0.002)
        assert s.next() is None  # strong unaffordable, cheap exhausted

    def test_default_policy_is_greedy(self):
        # cheap has the higher reservation price, strong the higher net value:
        # the default (Greedy) must pick strong and stop after one reward,
        # unlike ReservationPricePolicy, which would pick cheap.
        s = Search(self._estimates())
        assert s.next() == "strong"
        s.observe("strong", reward=0.10, cost=0.02)
        assert s.next() is None

    def test_greedy_one_shot(self):
        s = Search(self._estimates(), policy=Greedy())
        assert s.next() == "strong"  # highest net value
        s.observe("strong", reward=0.10, cost=0.02)
        assert s.next() is None

    def test_exhaustive_cheapest_first_no_stop(self):
        s = Search(self._estimates(), policy=Exhaustive(), budget=1.0, max_tries=1)
        assert s.next() == "cheap"
        s.observe("cheap", reward=0.10, cost=0.002)  # even a pass does not stop it
        assert s.next() == "strong"

    def test_update_estimates_rejects_unknown_label(self):
        s = Search(self._estimates())
        with pytest.raises(KeyError):
            s.update_estimates({"nope": Estimate(Bernoulli(0.5, 1.0), 0.01)})

    def test_explain_ranked(self):
        s = Search(self._estimates())
        ranking = s.explain()
        prices = [r.reservation_price for r in ranking]
        assert prices == sorted(prices, reverse=True)
        assert {r.label for r in ranking} == {"cheap", "strong"}

    def test_empty_estimates_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            Search({})


# ══════════════════════════════════════════════════════════════════
# Beliefs
# ══════════════════════════════════════════════════════════════════


def _dummy_fn():
    @ai_function[str](structured_output=False)
    def fn(x: str) -> str:
        """{x}"""

    return fn


class TestEmpiricalBeliefs:
    def _candidates(self):
        fn = _dummy_fn()
        return [
            Candidate("cheap", fn, CHEAP_PRICES),
            Candidate("strong", fn, STRONG_PRICES),
        ]

    @pytest.mark.asyncio
    async def test_prior_then_learns(self):
        b = EmpiricalBeliefs()
        cands = self._candidates()
        est0 = await b.estimate(_task(), cands, value=0.10, history=[])
        assert est0["cheap"].dist.mean() == pytest.approx(0.05)  # 1/(1+1) * 0.10

        b.update(
            AttemptRecord(id=RecordId("r1"), task=_task(), candidate="cheap", cost=0.002, reward=0.0, local_score=0.0)
        )
        b.update(
            AttemptRecord(id=RecordId("r2"), task=_task(), candidate="cheap", cost=0.002, reward=0.10, local_score=1.0)
        )
        est1 = await b.estimate(_task(), cands, value=0.10, history=[])
        # alpha=1+1, beta=1+1 -> p=0.5
        assert est1["cheap"].dist.mean() == pytest.approx(0.05)
        # cost now reflects observed mean, not the token prior
        assert est1["cheap"].cost == pytest.approx(0.002)

    @pytest.mark.asyncio
    async def test_rejects_missing_value_scale(self):
        # Under merge no constant value scale exists (estimate receives None);
        # a pass-rate provider must refuse loudly rather than price at a
        # fictitious scale and silently abstain.
        b = EmpiricalBeliefs()
        with pytest.raises(AIFunctionError, match="constant value scale"):
            await b.estimate(_task(), self._candidates(), value=None, history=[])

    @pytest.mark.asyncio
    async def test_settlement_revises_exactly(self):
        b = EmpiricalBeliefs()
        rec = AttemptRecord(
            id=RecordId("r1"), task=_task(), candidate="cheap", cost=0.002, reward=0.10, local_score=1.0
        )
        b.update(rec)
        assert "100% pass" in b.stats()["cheap"]  # displayed rate is evidence only
        b.settle(RecordId("r1"), 0.0)  # downstream says it was useless
        assert "0% pass" in b.stats()["cheap"]  # settlement replaces the verdict
        # The routing estimate smooths with the prior: alpha=1+0, beta=1+1 -> p=1/3
        est = await b.estimate(_task(), self._candidates(), value=1.0, history=[])
        assert est["cheap"].dist.mean() == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    async def test_fixed_beliefs_constant(self):
        b = Beliefs.fixed({"cheap": Estimate(Bernoulli(0.7, 0.10), 0.001)})
        cand = [Candidate("cheap", _dummy_fn(), CHEAP_PRICES)]
        est = await b.estimate(_task(), cand, value=0.10, history=[])
        assert est["cheap"].dist.mean() == pytest.approx(0.07)

    @pytest.mark.asyncio
    async def test_fixed_beliefs_missing_label_raises(self):
        b = Beliefs.fixed({"cheap": Estimate(Bernoulli(0.7, 0.10), 0.001)})
        with pytest.raises(KeyError):
            await b.estimate(_task(), self._candidates(), value=0.10, history=[])

    def test_decay_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="decay"):
            EmpiricalBeliefs(decay=1.5)

    def test_turn_stats_tracked_in_track_record(self):
        from ai_functions.types import TokenUsage

        b = EmpiricalBeliefs()
        b.update(
            AttemptRecord(
                id=RecordId("r1"),
                task=_task(),
                candidate="cheap",
                cost=0.002,
                usage=TokenUsage(output_tokens=300),
                turns=3,
                reward=0.10,
                local_score=1.0,
            )
        )
        b.update(
            AttemptRecord(
                id=RecordId("r2"),
                task=_task(),
                candidate="cheap",
                cost=0.002,
                usage=TokenUsage(output_tokens=100),
                turns=1,
                reward=0.10,
                local_score=1.0,
            )
        )
        # mean turns (3+1)/2 = 2; tokens/turn (100 + 100)/2 = 100
        assert "avg 2.0 turns at 100 output tokens/turn" in b.stats()["cheap"]

    def test_turn_stats_skip_unmeasured_records(self):
        b = EmpiricalBeliefs()
        # A Decision.report booking carries no turn data (turns=0) and must
        # not drag the average toward zero.
        b.update(
            AttemptRecord(id=RecordId("r1"), task=_task(), candidate="cheap", cost=0.002, reward=0.10, local_score=1.0)
        )
        assert "turns" not in b.stats()["cheap"]


class TestForecastCostModel:
    def test_single_turn_is_prompt_write_plus_output(self):
        from ai_functions.experimental.economics.beliefs import _approx_cost

        # 1 turn: prompt written once at output/4, no reads, one turn of output.
        cost = _approx_cost(output_price=10.0, prompt_tokens=1000, turns=1, output_tokens_per_turn=200)
        expected = (1000 * 2.5 + 200 * 10.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_multi_turn_reads_growing_context(self):
        from ai_functions.experimental.economics.beliefs import _approx_cost

        # 3 turns, prompt P=1000, o=100/turn, output price 10:
        #   written: P + 2o = 1200 @ 2.5
        #   read:    2P + o*(2*3/2) = 2300 @ 0.2
        #   output:  3o = 300 @ 10
        cost = _approx_cost(output_price=10.0, prompt_tokens=1000, turns=3, output_tokens_per_turn=100)
        expected = (1200 * 2.5 + 2300 * 0.2 + 300 * 10.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_more_turns_cost_more(self):
        from ai_functions.experimental.economics.beliefs import _approx_cost

        costs = [
            _approx_cost(output_price=10.0, prompt_tokens=1000, turns=t, output_tokens_per_turn=100)
            for t in (1, 2, 5, 10)
        ]
        assert costs == sorted(costs)


# ══════════════════════════════════════════════════════════════════
# EconomicFunction — construction guards
# ══════════════════════════════════════════════════════════════════


class TestConstruction:
    def _cand(self, label="c"):
        return {label: Candidate(label, _dummy_fn(), CHEAP_PRICES)}

    def test_empty_candidates_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            EconomicFunction({}, value=0.10, beliefs=EmpiricalBeliefs())

    def test_unbounded_without_budget_rejected(self):
        with pytest.raises(ValueError, match="budget"):
            EconomicFunction(self._cand(), value=0.10, beliefs=EmpiricalBeliefs(), max_tries=None)

    def test_callable_value_rejected(self):
        with pytest.raises(ValueError, match="constant dollar amount"):
            EconomicFunction(self._cand(), value=lambda r: 0.10, beliefs=EmpiricalBeliefs())

    def test_nonpositive_value_rejected(self):
        with pytest.raises(ValueError, match="positive dollars"):
            EconomicFunction(self._cand(), value=0.0, beliefs=EmpiricalBeliefs())

    def test_score_accepted(self):
        fn = EconomicFunction(self._cand(), value=0.10, beliefs=EmpiricalBeliefs(), scorer=lambda r: 0.5)
        assert fn._scorer is not None

    def test_mismatched_label_rejected(self):
        cand = {"wrong": Candidate("actual", _dummy_fn(), CHEAP_PRICES)}
        with pytest.raises(ValueError, match="mismatched"):
            EconomicFunction(cand, value=0.10, beliefs=EmpiricalBeliefs())


class TestPricedModel:
    def test_label_derived_from_string(self):
        m = PricedModel(model="claude-x", prices=CHEAP_PRICES)
        assert m.label == "claude-x"

    def test_explicit_label_wins(self):
        m = PricedModel(model="claude-x", prices=CHEAP_PRICES, label="mine")
        assert m.label == "mine"


# ══════════════════════════════════════════════════════════════════
# End-to-end over scripted models
# ══════════════════════════════════════════════════════════════════


def _require_yes(result, **kwargs):
    if "yes" not in result:
        return PostConditionResult(passed=False, message="must contain yes")
    return None


@ai_function[str](structured_output=False, post_conditions=[_require_yes])
def _solve(task: str) -> str:
    """Solve: {task}"""


def _fixed_beliefs():
    return Beliefs.fixed(
        {
            "cheap": Estimate(Bernoulli(0.6, 0.10), 0.00001),
            "strong": Estimate(Bernoulli(0.95, 0.10), 0.0002),
        }
    )


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_escalates_and_attributes_cost(self):
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([Turn(text="cheap says no", input_tokens=5, output_tokens=10)])
            strong = ScriptedModel([Turn(text="strong says yes", input_tokens=8, output_tokens=20)])
            cands = {
                "cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0)),
                "strong": Candidate("strong", _solve.replace(model=strong), Prices(input=1.0, output=10.0)),
            }
            fn = EconomicFunction(
                cands, value=0.10, beliefs=_fixed_beliefs(), budget=0.25, policy=ReservationPricePolicy()
            )
            handle = await h.spawn(fn)
            result = await handle.run(task="t")
            assert "yes" in result

            recs = await attempts(handle)
            assert [(r.candidate, r.local_score > 0) for r in recs] == [("cheap", False), ("strong", True)]
            # per-attempt usage measured independently
            assert recs[0].usage.output_tokens == 10
            assert recs[1].usage.output_tokens == 20
            # each attempt was a single model call
            assert recs[0].turns == 1
            assert recs[1].turns == 1
            # cost prices both input and output: cheap (5 in + 10 out) @ $1/M
            assert recs[0].cost == pytest.approx((5 * 1.0 + 10 * 1.0) / 1e6)
            # strong (8 in @ $1/M + 20 out @ $10/M)
            assert recs[1].cost == pytest.approx((8 * 1.0 + 20 * 10.0) / 1e6)

            # spend rolls the attempt costs; decisions records the ranking round
            assert await spend(handle) == pytest.approx(recs[0].cost + recs[1].cost)
            rounds = await decisions(handle)
            assert rounds and {r.label for r in rounds[0]} == {"cheap", "strong"}

    @pytest.mark.asyncio
    async def test_non_aifunction_error_is_a_failed_attempt(self):
        # An attempt that raises a non-AIFunctionError (here: a model/runtime
        # error surfaced as the script running dry) must count as a failed
        # attempt and escalate — a fallback layer swallows operational errors.
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([])  # raises ScriptExhausted on invocation
            strong = ScriptedModel([Turn(text="strong says yes", input_tokens=8, output_tokens=20)])
            cands = {
                "cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0)),
                "strong": Candidate("strong", _solve.replace(model=strong), Prices(input=1.0, output=10.0)),
            }
            fn = EconomicFunction(
                cands, value=0.10, beliefs=_fixed_beliefs(), budget=0.25, policy=ReservationPricePolicy()
            )
            handle = await h.spawn(fn)
            result = await handle.run(task="t")
            assert "yes" in result
            recs = await attempts(handle)
            assert [(r.candidate, r.local_score > 0) for r in recs] == [("cheap", False), ("strong", True)]

    @pytest.mark.asyncio
    async def test_first_pass_no_escalation(self):
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([Turn(text="cheap says yes", input_tokens=5, output_tokens=10)])
            strong = ScriptedModel([])
            cands = {
                "cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0)),
                "strong": Candidate("strong", _solve.replace(model=strong), Prices(input=1.0, output=10.0)),
            }
            fn = EconomicFunction(cands, value=0.10, beliefs=_fixed_beliefs(), budget=0.25)
            handle = await h.spawn(fn)
            result = await handle.run(task="t")
            assert "yes" in result
            assert strong.remaining_turns == 0  # never invoked

    @pytest.mark.asyncio
    async def test_all_fail_raises_exhausted_with_records(self):
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([Turn(text="no", output_tokens=1)])
            cands = {"cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0))}
            fn = EconomicFunction(cands, value=0.10, beliefs=_fixed_beliefs(), budget=0.25)
            handle = await h.spawn(fn)
            with pytest.raises(CandidatesExhausted) as exc:
                await handle.run(task="t")
            assert len(exc.value.records) == 1
            assert exc.value.records[0].candidate == "cheap"

    @pytest.mark.asyncio
    async def test_abstains_when_nothing_worth_it(self):
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([])
            cands = {"cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0))}
            # value 0.10 but cost 1.0 -> net value negative -> abstain, never runs
            beliefs = Beliefs.fixed({"cheap": Estimate(Bernoulli(0.5, 0.10), 1.0)})
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=2.0)
            handle = await h.spawn(fn)
            with pytest.raises(Abstained):
                await handle.run(task="t")
            assert cheap.remaining_turns == 0  # never invoked

    @pytest.mark.asyncio
    async def test_graded_best_of_n_keeps_highest_score(self):
        # Independent graded best-of-N: each attempt is scored in [0, 1] and
        # priced at value; the search keeps the best result by reward and stops
        # once a draw beats every reservation price. FIXED estimates stop on
        # their own as the best in hand rises.
        @ai_function[str](structured_output=False)
        def _draft(src: str) -> str:
            """{src}"""

        def score(s: str) -> float:
            # Fraction of the 5-item target found -> quality in [0, 1].
            return len([x for x in s.split(",") if x]) / 5.0

        async with RuntimeHarness() as h:
            # Draws of improving quality: 1 item (0.2), then 3 (0.6); a fourth
            # turn exists but must never run — the bar stops the search first.
            model = ScriptedModel(
                [
                    Turn(text="a", output_tokens=10),
                    Turn(text="a,b,c", output_tokens=10),
                    Turn(text="a,b,c,d,e", output_tokens=10),
                ]
            )
            cand = {"m": Candidate("m", _draft.replace(model=model), Prices(input=1.0, output=1.0))}
            # value=$0.10 for a perfect result, so reward = 0.10 * score:
            # draw 1 banks $0.02, draw 2 banks $0.06. A draw ~ N($0.05, $0.01)
            # at $0.005 cost has reservation price ~$0.048 — continue while the
            # best in hand is below it, stop the moment a draw banks more.
            fn = EconomicFunction(
                cand,
                value=0.10,
                scorer=score,
                beliefs=Beliefs.fixed({"m": Estimate(Gaussian(mu=0.05, sigma=0.01), 0.005)}),
                budget=1.0,
                policy=ReservationPricePolicy(),
                max_tries=None,
            )
            handle = await h.spawn(fn)
            result = await handle.run(src="doc")

            # Kept the best result (draw 2), stopped before the third turn.
            assert result.strip() == "a,b,c"
            recs = await attempts(handle)
            assert len(recs) == 2
            # reward = value * score: 0.10*0.2, then 0.10*0.6.
            assert recs[0].reward == pytest.approx(0.02)
            assert recs[1].reward == pytest.approx(0.06)
            # local_score carries the graded quality now, not just 0/1.
            assert recs[0].local_score == pytest.approx(0.2)
            assert recs[1].local_score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_out_of_range_score_raises_clearly(self):
        # A score outside [0, 1] must fail clearly at ingestion, not obscurely
        # later at Bernoulli construction.
        @ai_function[str](structured_output=False)
        def _draft(src: str) -> str:
            """{src}"""

        async with RuntimeHarness() as h:
            model = ScriptedModel([Turn(text="x", output_tokens=10)])
            cand = {"m": Candidate("m", _draft.replace(model=model), Prices(input=1.0, output=1.0))}
            fn = EconomicFunction(
                cand,
                value=0.10,
                scorer=lambda s: 5.0,  # bug: returns a dollar-like amount, not [0, 1]
                beliefs=EmpiricalBeliefs(),
                budget=1.0,
            )
            handle = await h.spawn(fn)
            with pytest.raises(ValueError, match=r"score must be in \[0, 1\]"):
                await handle.run(src="doc")

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises_with_records(self):
        async with RuntimeHarness() as h:
            # cheap fails; strong would pass but its cost exceeds the tiny budget
            cheap = ScriptedModel([Turn(text="no", input_tokens=1, output_tokens=1)])
            strong = ScriptedModel([Turn(text="yes", input_tokens=1, output_tokens=1)])
            cands = {
                "cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0)),
                "strong": Candidate("strong", _solve.replace(model=strong), Prices(input=1.0, output=1.0)),
            }
            beliefs = Beliefs.fixed(
                {
                    "cheap": Estimate(Bernoulli(0.5, 0.10), 0.000002),
                    "strong": Estimate(Bernoulli(0.99, 0.10), 0.01),  # exceeds budget below
                }
            )
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=0.000005)
            handle = await h.spawn(fn)
            with pytest.raises(BudgetExceeded) as exc:
                await handle.run(task="t")
            assert exc.value.records  # cheap attempt is booked before the block

    @pytest.mark.asyncio
    async def test_plan_reports_and_learns(self):
        # plan() estimates without executing, so no runtime is needed.
        cands = {"cheap": Candidate("cheap", _solve, Prices(input=1.0, output=1.0))}
        beliefs = EmpiricalBeliefs()
        fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0)
        decision = await fn.plan(task="t")
        assert decision.candidate is not None
        assert decision.candidate.label == "cheap"
        assert decision.ranking[0].label == "cheap"
        # report an externally executed success -> beliefs learn
        decision.report("yes", cost=0.001)
        assert "pass" in beliefs.stats()["cheap"]

    @pytest.mark.asyncio
    async def test_plan_reports_accumulate_across_calls(self):
        # Each plan()'s report must book a distinct record id: beliefs key
        # contributions by id, so a repeated id would overwrite the earlier
        # task's evidence instead of adding to it.
        cands = {"cheap": Candidate("cheap", _solve, Prices(input=1.0, output=1.0))}
        beliefs = EmpiricalBeliefs()
        fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0)
        for _ in range(3):
            decision = await fn.plan(task="t")
            decision.report("yes", cost=0.001)
        assert "over 3 attempts" in beliefs.stats()["cheap"]

    @pytest.mark.asyncio
    async def test_plan_declines_when_unprofitable(self):
        cands = {"cheap": Candidate("cheap", _solve, Prices(input=1.0, output=1.0))}
        beliefs = Beliefs.fixed({"cheap": Estimate(Bernoulli(0.5, 0.10), 1.0)})
        fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=2.0)
        decision = await fn.plan(task="t")
        assert decision.candidate is None

    @pytest.mark.asyncio
    async def test_plan_works_over_any_distribution(self):
        # The routing decision (candidate, ranking, report) needs only
        # reservation prices, which every RewardDistribution provides.
        cands = {"cheap": Candidate("cheap", _solve, Prices(input=1.0, output=1.0))}
        beliefs = Beliefs.fixed({"cheap": Estimate(Gaussian(mu=0.05, sigma=0.02), 0.002)})
        fn = EconomicFunction(cands, value=0.10, beliefs=beliefs)
        decision = await fn.plan(task="t")
        assert decision.candidate is not None
        assert decision.ranking[0].label == "cheap"


# ══════════════════════════════════════════════════════════════════
# Decorators
# ══════════════════════════════════════════════════════════════════


class TestDecorators:
    def test_routed_builds_candidates_from_models(self):
        models = [
            PricedModel(model="haiku", prices=CHEAP_PRICES),
            PricedModel(model="sonnet", prices=STRONG_PRICES),
        ]

        @routed(models=models, value=0.10)
        @ai_function[str](structured_output=False)
        def solve(task: str) -> str:
            """Solve: {task}"""

        assert set(solve.candidates) == {"haiku", "sonnet"}
        assert isinstance(solve.beliefs, EmpiricalBeliefs)

    def test_routed_rejects_both_sources(self):
        with pytest.raises(ValueError, match="exactly one"):

            @routed(models=[PricedModel(model="m", prices=CHEAP_PRICES)], candidates=[], value=0.10)
            @ai_function[str](structured_output=False)
            def solve(task: str) -> str:
                """{task}"""

    def test_routed_rejects_callable_value(self):
        with pytest.raises(ValueError, match="constant dollar amount"):

            @routed(models=[PricedModel(model="m", prices=CHEAP_PRICES)], value=lambda r: 0.10)  # type: ignore[arg-type]
            @ai_function[str](structured_output=False)
            def solve(task: str) -> str:
                """{task}"""

    def test_routed_rejects_nonpositive_value(self):
        with pytest.raises(ValueError, match="positive dollars"):

            @routed(models=[PricedModel(model="m", prices=CHEAP_PRICES)], value=0.0)
            @ai_function[str](structured_output=False)
            def solve(task: str) -> str:
                """{task}"""

    def test_routed_default_policy_is_reservation_price(self):
        @routed(models=[PricedModel(model="haiku", prices=CHEAP_PRICES)], value=0.10)
        @ai_function[str](structured_output=False)
        def solve(task: str) -> str:
            """{task}"""

        assert isinstance(solve._policy, ReservationPricePolicy)

    def test_routed_passes_score(self):
        def grade(r: str) -> float:
            return 0.5

        @routed(models=[PricedModel(model="haiku", prices=CHEAP_PRICES)], value=0.10, scorer=grade)
        @ai_function[str](structured_output=False)
        def solve(task: str) -> str:
            """{task}"""

        assert solve._scorer is grade

    def test_routed_defaults_beliefs_to_empirical(self):
        @routed(models=[PricedModel(model="haiku", prices=CHEAP_PRICES)], value=0.10)
        @ai_function[str](structured_output=False)
        def review(src: str) -> str:
            """{src}"""

        assert set(review.candidates) == {"haiku"}
        assert isinstance(review.beliefs, EmpiricalBeliefs)


# ══════════════════════════════════════════════════════════════════
# Backward-pass learning: settlement via the optimizer
# ══════════════════════════════════════════════════════════════════


class _RecordingBeliefs(Beliefs):
    """Beliefs that record what the backward pass drives into them."""

    def __init__(self) -> None:
        self.settled: list[tuple[str, float]] = []

    async def estimate(self, task, candidates, value, history):  # noqa: ANN001, ARG002
        return {c.label: Estimate(Bernoulli(0.5, value), 0.001) for c in candidates}

    def settle(self, record_id, score):  # noqa: ANN001
        self.settled.append((str(record_id), score))


def _rec(rid: str, candidate: str, *, passed: bool) -> AttemptRecord:
    """A minimal AttemptRecord for driving beliefs directly."""
    return AttemptRecord(
        id=RecordId(rid),
        task=_task(),
        candidate=candidate,
        cost=0.001,
        reward=0.10 if passed else 0.0,
        local_score=1.0 if passed else 0.0,
    )


def _host_fn(beliefs):  # noqa: ANN001, ANN202
    """An EconomicFunction over ``beliefs`` — it is its own ParameterHost."""
    return EconomicFunction({"cheap": Candidate("cheap", _dummy_fn(), CHEAP_PRICES)}, value=0.10, beliefs=beliefs)


class TestFunctionAsParameterHost:
    def test_consolidate_settles_every_record_to_averaged_score(self):
        from ai_functions.types.graph import GradFeedback

        beliefs = _RecordingBeliefs()
        fn = _host_fn(beliefs)
        assert fn.backend_id == "economics:fn"  # derived from the wrapped function name
        # ``retrieved`` maps record id -> candidate; two gradients averaged.
        fn.consolidate(
            "routing_decision",
            [GradFeedback(text="too shallow", score=0.1), GradFeedback(text="also thin", score=0.3)],
            retrieved={"r1": "cheap", "r2": "strong"},
        )
        assert beliefs.settled == [("r1", 0.2), ("r2", 0.2)]

    def test_consolidate_clamps_and_ignores_missing_pieces(self):
        from ai_functions.types.graph import GradFeedback

        beliefs = _RecordingBeliefs()
        fn = _host_fn(beliefs)
        fn.consolidate("d", [GradFeedback(text="great", score=1.5)], retrieved={"r1": "cheap"})  # clamp
        assert beliefs.settled == [("r1", 1.0)]
        # No score, or no records: nothing to settle.
        fn.consolidate("d", [GradFeedback(text="x", score=None)], retrieved={"r1": "cheap"})
        fn.consolidate("d", [GradFeedback(text="y", score=0.5)], retrieved=None)
        assert beliefs.settled == [("r1", 1.0)]  # unchanged


class TestBackwardWiring:
    """The graph-level contract: the decision parameter's host settles on consolidate."""

    def test_decision_parameter_settles_records_on_consolidate(self):
        from ai_functions.experimental.economics.function import DECISION_PARAMETER
        from ai_functions.optimizer import TextGradOptimizer
        from ai_functions.types.graph import ParameterNode, ThreadNode

        beliefs = _RecordingBeliefs()
        host = _host_fn(beliefs)  # the economic function is its own ParameterHost
        # The economic child owns a grad-enabled decision parameter hosted by
        # the function; its meta carries the run's record ids.
        decision = ParameterNode(
            node_id="research-1-decision",
            value="routing summary",
            requires_grad=True,
            name=DECISION_PARAMETER,
            backend=host,
            description="score settles the routing decision",
            meta={"results": {"r1": "cheap"}},
        )
        child = ThreadNode(node_id="research-1", thread_id="research-1", value="some sources", parameters=[decision])
        root = ThreadNode(node_id="write-1", thread_id="write-1", value="the report", child_threads=[child])

        # Stubbed backward: at the root, score + text the economic child; at the
        # child, the score routes onto its decision parameter.
        opt = TextGradOptimizer()
        opt._backward_fn = _RouteWithScore(  # noqa: SLF001
            {"research-1": ("too shallow", 0.2), "research-1-decision": ("shallow", 0.2)}
        )
        opt.backward(root, "the report was too shallow")
        opt.consolidate(root)

        assert beliefs.settled == [("r1", 0.2)]

    def test_node_kept_when_it_owns_grad_decision_parameter(self):
        """An economic node stays in the walk because it owns a grad parameter."""
        from ai_functions.experimental.economics.function import DECISION_PARAMETER
        from ai_functions.optimizer._graph import leads_to_grad_parameter
        from ai_functions.types.graph import ParameterNode, ThreadNode

        decision = ParameterNode(node_id="c-decision", requires_grad=True, name=DECISION_PARAMETER)
        child = ThreadNode(node_id="c", thread_id="c", parameters=[decision])
        root = ThreadNode(node_id="r", thread_id="r", child_threads=[child])
        assert leads_to_grad_parameter(root) is True


class _RouteWithScore:
    """Offline backward stand-in that emits (feedback, score) per matching node id."""

    def __init__(self, responses: dict[str, tuple[str, float]]) -> None:
        self.responses = responses

    def replace(self, **kwargs: object) -> _RouteWithScore:
        del kwargs
        return self

    def run_sync(self, *, inputs: str, **kwargs: object) -> object:  # noqa: ARG002
        import yaml

        from ai_functions.optimizer.textgrad import Feedback, Feedbacks

        rendered = yaml.safe_load(inputs) or {}
        return Feedbacks(
            feedbacks=[
                Feedback(node_id=nid, feedback=text, score=score)
                for nid, (text, score) in self.responses.items()
                if nid in rendered
            ]
        )


class TestGraphIntegration:
    """The reconstructed economic node: naming, adopted trace, grad-enabled notes."""

    @pytest.mark.asyncio
    async def test_node_named_after_wrapped_function(self):
        from ai_functions.optimizer import build_graph_from_result

        async with RuntimeHarness() as h:

            @ai_function[str](structured_output=False)
            def research(task: str) -> str:
                """Research: {task}"""

            model = ScriptedModel([Turn(text="sources: a, b", output_tokens=10)])
            cands = {"cheap": Candidate("cheap", research.replace(model=model), CHEAP_PRICES)}
            fn = EconomicFunction(cands, value=0.10, beliefs=_fixed_beliefs(), budget=1.0)
            assert fn.name == "research"  # not economic(research)

            run = await _trace_on(h, fn, task="X")
            graph = await build_graph_from_result(run, [])
            assert graph.func_name == "research"
            assert graph.node_id.startswith("research-")

    @pytest.mark.asyncio
    async def test_node_adopts_responsible_attempt_trace(self):
        from ai_functions.optimizer import build_graph_from_result

        async with RuntimeHarness() as h:

            @ai_function[str](structured_output=False, post_conditions=[lambda r: _must_say_yes(r)])
            def research(task: str) -> str:
                """Research: {task}"""

            cheap = ScriptedModel([Turn(text="no", output_tokens=1)])
            strong = ScriptedModel([Turn(text="yes indeed", output_tokens=5)])
            cands = {
                "cheap": Candidate("cheap", research.replace(model=cheap), CHEAP_PRICES),
                "strong": Candidate("strong", research.replace(model=strong), STRONG_PRICES),
            }
            beliefs = Beliefs.fixed(
                {
                    "cheap": Estimate(Bernoulli(0.9, 0.10), 0.000001),
                    "strong": Estimate(Bernoulli(0.99, 0.10), 0.00001),
                }
            )
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0, policy=ReservationPricePolicy())
            run = await _trace_on(h, fn, task="X")

            graph = await build_graph_from_result(run, [])
            text = "\n".join(str(m) for m in graph.messages)
            # The routing summary names both attempts and the responsible one...
            assert "Attempt 1: cheap — failed local checks" in text
            assert "Attempt 2: strong — passed local checks" in text
            assert "The returned output was produced by: strong" in text
            # ...and the adopted conversation is the responsible attempt's, not the loser's.
            assert "yes indeed" in text
            assert "role': 'user" in text or "'role': 'user'" in text

    @pytest.mark.asyncio
    async def test_forecaster_notes_surface_as_grad_parameter(self, tmp_path):
        from pydantic import BaseModel, Field

        from ai_functions.experimental.economics.beliefs import LLMForecaster, RoutingMemory, forecast
        from ai_functions.memory import JSONMemoryBackend
        from ai_functions.optimizer import build_graph_from_result

        class Mem(BaseModel):
            routing: RoutingMemory = Field(default_factory=RoutingMemory)

        async with RuntimeHarness() as h:

            @ai_function[str](structured_output=False)
            def research(task: str) -> str:
                """Research: {task}"""

            memory = JSONMemoryBackend(Mem, actor_id="t", path=tmp_path / "m.json")
            scripted_forecast = forecast.replace(
                model=ScriptedModel(
                    [
                        Turn(
                            tool_calls=(
                                (
                                    "ForecastResult",
                                    {
                                        "estimates": {
                                            "cheap": {
                                                "pass_percentage": 90,
                                                "turns": 1,
                                                "output_tokens_per_turn": 10,
                                            }
                                        }
                                    },
                                ),
                            )
                        )
                    ]
                )
            )
            beliefs = LLMForecaster(forecast_fn=scripted_forecast, memory=memory, memory_key="routing")
            model = ScriptedModel([Turn(text="sources", output_tokens=10)])
            cands = {"cheap": Candidate("cheap", research.replace(model=model), CHEAP_PRICES)}
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0)
            run = await _trace_on(h, fn, task="X")

            graph = await build_graph_from_result(run, [memory])
            params = {p.name: p for p in graph.parameters}
            # The notes recall landed in the economic run's log, grad-enabled —
            # the backward pass has a routable target at this node — while the
            # machine-managed stats never surface as a parameter.
            assert "routing/notes" in params
            assert params["routing/notes"].requires_grad is True
            assert params["routing/notes"].backend is memory
            assert "routing/stats" not in params
            # The attempt's statistics were persisted through the backend.
            stats = memory.fetch("routing/stats")
            assert len(stats) == 1 and stats[0].candidate == "cheap"

    @pytest.mark.asyncio
    async def test_empirical_stats_persist_across_instances(self, tmp_path):
        from pydantic import BaseModel, Field

        from ai_functions.experimental.economics.beliefs import RoutingMemory
        from ai_functions.memory import JSONMemoryBackend

        class Mem(BaseModel):
            routing: RoutingMemory = Field(default_factory=RoutingMemory)

        memory = JSONMemoryBackend(Mem, actor_id="t", path=tmp_path / "m.json")
        b1 = EmpiricalBeliefs(memory=memory, stats_key="routing/stats")
        b1.update(_rec("r1", "cheap", passed=True))
        b1.settle(RecordId("r1"), 0.25)

        # A fresh instance (as after a process restart) reloads the settled record.
        b2 = EmpiricalBeliefs(memory=memory, stats_key="routing/stats")
        assert "pass over 1 attempts" in b2.stats()["cheap"]
        # The settled 0.25 survived the round trip, not the local 1.0.
        assert "25% pass" in b2.stats()["cheap"]

    def test_empirical_memory_requires_key(self):
        from ai_functions.memory import JSONMemoryBackend  # noqa: F401 — import guard only

        with pytest.raises(ValueError, match="stats_key"):
            EmpiricalBeliefs(memory=object())  # type: ignore[arg-type]


def _must_say_yes(result: str) -> PostConditionResult | None:
    if "yes" not in result:
        return PostConditionResult(passed=False, message="must say yes")
    return None


class TestLearningEndToEnd:
    """trace() -> build_graph -> backward -> consolidate, over a live run."""

    @pytest.mark.asyncio
    async def test_settlement_moves_posterior_after_backward(self):
        from ai_functions.optimizer import TextGradOptimizer, build_graph_from_result

        async with RuntimeHarness() as h:
            # A routed function with no post-condition, so its attempt passes
            # locally — a success that downstream feedback will then overturn.
            @ai_function[str](structured_output=False)
            def research(task: str) -> str:
                """Research: {task}"""

            model = ScriptedModel([Turn(text="sources: a, b", output_tokens=10)])
            cands = {"cheap": Candidate("cheap", research.replace(model=model), Prices(input=1.0, output=1.0))}
            beliefs = EmpiricalBeliefs()
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0)

            run = await _trace_on(h, fn, task="research X")

            recs = await attempts(run)
            assert recs and recs[0].local_score == 1.0  # locally a success
            assert "100% pass" in beliefs.stats()["cheap"]  # displayed rate is evidence only

            # Reconstruct the graph with the function among the backends (it is
            # its own host), so the run's decision parameter matches and settles.
            from ai_functions.experimental.economics.function import DECISION_PARAMETER
            from ai_functions.types.graph import GradFeedback

            opt = TextGradOptimizer()
            graph = await build_graph_from_result(run, [fn])
            decision = next(p for p in graph.parameters if p.name == DECISION_PARAMETER)
            assert decision.requires_grad and decision.backend is fn

            # The downstream consumer judges the routing worthless (score 0.0).
            # Seed the decision parameter as the backward pass would, then
            # consolidate — the host settles the run's records.
            decision.gradients.append(GradFeedback(text="paraphrased news, needed primary sources", score=0.0))
            opt.consolidate(graph)

            # settle overrode the local success -> the displayed rate drops to 0
            assert "0% pass" in beliefs.stats()["cheap"]


async def _trace_on(h, fn, **kwargs):  # noqa: ANN001, ANN201
    """Trace ``fn`` on the harness coordinator so its event log is reachable."""
    handle = await h.spawn(fn)
    from ai_functions.types.graph import Result

    value = await handle.run(**kwargs)
    return Result(value=value, coordinator=h.coordinator, thread_id=handle.id, inputs=[])
