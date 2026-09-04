# Invariants

Cross-cutting facts whose violation would break reasoning in two or
more independent classes or modules. Method-local postconditions live
on the method as `:ensures:`, not here.

## I1 — Single host worker per thread

A thread has exactly one *host* worker. The thread is registered
with the coordinator (via `register_thread`) by that host iff it is
registered at all; deregistration is driven by the host. Violating
this splits message routing between two workers or leaves routes
pointing at a dead host.

Classes that must agree: every `WorkerAdapter` implementation (local
and remote), `Coordinator` implementations, and any caller of
`route_message`.

## I2 — Single append path for events

Every `Event` visible anywhere in the system was emitted through
`Coordinator.append_event`. Persistence (`get_events` replay) and
live broadcast (subscribers registered via `on`) are the two fan-outs
of that single call site; there is no side channel.

Classes that must agree: every emitter (worker dispatcher, executor,
session layer), `Coordinator` implementations, and every subscriber
that relies on event ordering.

## I3 — Rate-limit signals are driven by TOKEN_USAGE events

Per-thread pause signals exposed by `Coordinator.pause_signal` are
set and cleared by the coordinator subscribing to its own
`TOKEN_USAGE` events. There is no separate `report_tokens` path;
executors that await the signal see state coherent with the event
log.

Classes that must agree: `Coordinator` implementations, every
executor awaiting `ThreadContext.pause_signal`, and every producer of
`TOKEN_USAGE` events.

## I4 — Dispatcher starts only after all per-thread state is fully populated

The dispatcher task for a thread is created only after all per-thread
state (queues, signals, coordinator registration) and any seeded event
history (e.g. from `fork`) are fully in place. No work item may be
consumed by a dispatcher that observes partial state or a half-seeded
history.

Classes that must agree: every `Worker` implementation that creates
dispatcher tasks (local and any future distributed worker), and any
caller that seeds history before starting a thread (e.g. `fork`).

## I5 — Worker dispatcher is the sole emitter of lifecycle events

Lifecycle events (`STARTED`, `COMPLETED`, `CANCELLED`, `FAILED`,
`RESULT`) are emitted only by the worker's dispatcher task, never by
a thread's `execute` body. The runtime gates every `append_event`
call by emitter identity (`source="runtime"` vs `source="thread"`)
and rejects lifecycle kinds from threads. This is what makes the
"a thread cycle ran" claim trustworthy: the only way a `STARTED` /
`COMPLETED` pair can appear in the event log is via the dispatcher,
which always pairs them around a single `execute` call.

Classes that must agree: every `Worker` implementation, every
`Thread.execute` body (must not call `ctx.on_event` for lifecycle
kinds), and every test harness that synthesizes events.

## I6 — Fire-and-forget sends to a client must not raise unhandled exceptions

`OnEventCallback` is synchronous, so any network-hosted subscriber
(e.g. fan-out from `CoordinatorEndpoint` to a connected
`CoordinatorClient`) cannot `await` its sends inline — it must
dispatch them via `asyncio.create_task`. A task that is not awaited
has no caller to propagate its exception to; if the underlying send
raises (e.g. because the WebSocket closed between the event being
emitted and the task being scheduled), the exception becomes an
unhandled task exception and is silently swallowed or printed as a
warning, rather than surfacing to the subscriber or the coordinator.

The obligation is: every `create_task` that sends to a client
connection must catch all `Exception`s inside the task body. It must
not let a closed-connection error escape, because there is no handler
above it.

Note: this is a consequence of `OnEventCallback` being sync. The
alternative — making `OnEventCallback` async — would require
`append_event` to be async too, which conflicts with I2's requirement
that Strands hooks can call `append_event` from synchronous callbacks.
I6 is the invariant that names the obligation created by that design
choice.

Classes that must agree: `CoordinatorEndpoint` (and any future host
that bridges `on` subscriptions to a network transport).

## I7 — `MessageUserEvent` emission is paired with live-agent injection

**Scope**: this invariant constrains `AIThread` and any other `Thread`
implementation that drives a Strands agent with a cached prefix. It
does not apply to threads whose history is not rebuilt from the event
log.

All user-addressed text (whether from the initial prompt rendered
inside `AIThread.execute`, from `AIThread.notify`, or from the
post-condition retry loop) must reach the running Strands agent
through a single path: the thread's own inject buffer, drained by the
event-bridge hook at a safe work boundary.

Specifically, the hook's drain of the inject buffer must atomically:
1. emit a `MessageUserEvent` into the coordinator's event log, and
2. append a corresponding `{"role": "user", ...}` message onto
   `event.agent.messages` of the in-flight Strands agent.

These two steps must never drift. Any caller that emits a
`MessageUserEvent` without also injecting (or vice versa) breaks two
invariants at once:

- **Message order legality.** Draining only at `BeforeModelCallEvent`
  boundaries guarantees we never interleave a user message inside a
  tool_call → tool_result pair — the rehydrated history stays legal.
- **Cache prefix stability.** The agent observes messages in the same
  order they appear in the event log. The prefix seen by Strands during
  iteration N equals the prefix reconstructed for iteration N+1, so the
  prompt cache stays valid across cycles.

Corollary: the event-bridge hook is the **sole emitter** of
`MessageUserEvent` during a cycle. `AIThread.execute` appends its
generated prompt to the inject buffer (after any already-pending
injects); it does not call `ctx.on_event` directly for user turns.

Classes that must agree: `AIThread.execute`, `AIThread._run_cycle`,
`AIThread.notify`, `_EventBridgeHook._on_before_model_call`,
and any future `Thread` implementation that runs a Strands agent with
a cached prefix.

## I8 — Read-after-write coherence for coordinator state

Any call that reads state maintained by a `Coordinator` (`get_events`,
`thread_exists`, `is_paused`) must observe all events that were passed
to `append_event` by the same logical client before that read was
issued. In-process implementations satisfy this trivially (append and
read share the same data structure). Implementations that decouple the
write path from the network (e.g. an outbox queue) must flush any
pending writes to durable storage before executing the read.

Classes that must agree: every `Coordinator` implementation, and any
caller that reads coordinator state after emitting events (e.g.
`_run_cycle` calling `get_events` after `execute` emits `MESSAGE_USER`).

## I9 – Cache remains valid

Define a cache-invalidation point as any moment at which a history-rewriting effect is applied to agent.messages: a
ContextSummarizedEvent is emitted, a SessionResetEvent is emitted, or `conversation_manager.apply_management(agent)`
mutates `agent.messages` in a non-append-only way. These points partition the thread's lifetime into cache epochs.

For every `AIThread`, if at some moment `t` right before a model API call we observe `agent.messages = messages_0`,
then for every later model API call at `t'` such that no cache-invalidation point occurs in the open interval `(t, t']`,
`agent.messages.startswith(messages_0)` holds at `t'`.

### I9 rendering semantics after a ContextSummarizedEvent

When `reconstruct_messages` rebuilds a thread's history, the *last* `ContextSummarizedEvent` in the input (the one
with the greatest index in append order) defines a boundary. Events at that index or earlier contribute nothing to the
output directly; in their place, `reconstruct_messages` splices in
`[Message(role="user", content=[{"text": event.summary}]), *event.preserved_messages]`. Events strictly after the
boundary render normally and are appended to the spliced prefix. Earlier `ContextSummarizedEvent` instances in the
same log are superseded and contribute nothing.

This means a `ContextSummarizedEvent` is a cache-invalidation point *for the thread*, but not a cache-invalidation
point *for the summarization cycle* that produced it: when the thread runs summarization via a fork seeded from
its own resolved cycle config, the cache prefix the parent just populated on the provider side remains valid for
the fork's first model call.

### Custom events do not invalidate reconstruction boundaries

`CustomEvent` instances are inert to `reconstruct_messages` (skipped during the match). They therefore never shift
the summarization boundary, never introduce spurious messages, and never invalidate the cache on their own. Producers
may append custom events concurrently with any other thread activity, including during the window between a
summarization decision and the resulting `ContextSummarizedEvent` append, without breaking I9.

## I10 – No dangling tool calls

At every model-API-call boundary of every cycle of an AIThread, every toolUse block with id u in agent.messages is
followed (in the same or the next message) by a matching toolResult with `toolUseId == u`.

## I11 — Side-effect isolation in `append_event`

Once `Coordinator.append_event` has durably stored an event, any
failure in its side effects (status caching, subscriber fan-out,
rate-limit hook) MUST NOT propagate to the caller. The caller is
typically a worker dispatcher task; propagating would kill the thread
and cause downstream `handle.run(...)` futures to hang. Failures
must be logged and swallowed; subsequent side effects still run.

Classes that must agree: every `Coordinator` implementation, and
every subscriber registered through `Coordinator.on`.

## I12 — Liveness under failure

A `Coordinator` MUST guarantee that every pending operation (a
`handle.run` future, a pending `spawn`, etc.) eventually resolves —
either successfully or by raising — even if the backing transport,
worker, or internal task dies. No code path may silently orphan a
caller's await. Concretely, if a worker's dispatcher or a wire
connection fails, the coordinator is responsible for failing pending
futures with the root exception (or a clear surrogate like
`ConnectionClosedError`).

Classes that must agree: every `Coordinator` implementation,
`CoordinatorClient`, `CoordinatorEndpoint`, every `Worker`
implementation, and every `WorkerAdapter` shim.

# Economics invariants (E-series)

Invariants of the `ai_functions.experimental.economics` module. Same contract as
above, scoped to the economics layer; each is also stated inline in
the docstring of the module that enforces it.

## E1 — Dollars are the only unit

Rewards, costs, budgets, and reservation prices are all dollars. Any
layer that introduces a second unit (scores, token counts,
probabilities) must convert at its own boundary. `value` is the only
place task success is priced; post-conditions are never a reward
definition, only the local booking signal (E2 settles them later).
Violating this reintroduces an implicit exchange rate between layers,
silently miscalibrating the stopping rule.

Classes that must agree: `Prices`, every `Beliefs` implementation,
`Estimate` and every `RewardDistribution`, `Search` and every
`Policy`, and `EconomicFunction` (which owns `value`).

## E2 — Attempt records are revisable

An `AttemptRecord` is booked provisionally: `local_score` is booked
at run time from the candidate's own post-conditions (or a `score`
grader's `[0, 1]` value, when given). A `settled_score` is written
when downstream feedback arrives; consumers must treat `settled_score`
as authoritative when present. Learners store per-record contributions
(or sufficient statistics keyed by record id) such that `settle`
replaces a record's effect rather than double-counting it. Violating
this corrupts learned state on settlement, and repeated backward
passes compound the error.

Classes that must agree: every `Beliefs` implementation
(`update`/`settle`), every `AttemptRecord` consumer, and the
settlement path from the optimizer's backward pass.

## E3 — `Search` is deterministic and synchronous

Identical construction and an identical observe-sequence yield
identical decisions: no randomness, no hidden state, no async
anywhere in the search core. Violating this breaks
`blocked_by_budget()`'s counterfactual re-ask of the same policy and
makes the `DECISION_EVENT` trail unreproducible.

Classes that must agree: `Search`, every `Policy` implementation, and
every `RewardDistribution.reservation_price` implementation.

## E4 — Estimates are total

`Beliefs.estimate` must return an estimate for every candidate it is
given; a missing label is an error, never silently defaulted.
Violating this lets a fabricated estimate compete in the ranking, so
routing quietly degrades instead of erroring.

Classes that must agree: every `Beliefs` implementation and
`EconomicThread._estimate` (which enforces it after every estimation
round).

## E5 — Never knowingly spend more than the expected yield

An attempt is worth making only when it is expected to yield more than it costs.
When no candidate is worth trying at the start, the economic function abstains
rather than running the least-bad candidate (`Abstained`, or
`Decision.candidate is None` from `plan()`). At every later round the same
condition stops the search, keeping the best result so far.

Classes that must agree: every value-ranking `Policy`
(`ReservationPricePolicy`, `Greedy`), `EconomicThread.execute`
(which raises the typed failure), and `EconomicFunction.plan`.
