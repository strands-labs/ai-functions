"""Two economic functions in a workflow, settled by backprop.

A report pipeline: a *research* stage gathers sources, a *write* stage
drafts the report from them. Each stage is ``@routed`` between a cheap and
a strong model; an ``LLMForecaster`` reads each task and makes task-specific
estimates.

The point of this example is the learning loop. Each stage's post-conditions
are only *local* checks (format, citations present); whether the research
was actually useful is defined downstream, by the report. So run-time
bookings are provisional, and the optimizer's backward pass settles them:
one ``optimizer.step`` on the final report scores each stage's grad-enabled
routing-decision parameter, and from that one score —

- the numeric channel settles that run's attempt records, so if haiku's
  searches never feed a good report, haiku's posterior sinks even though its
  local checks passed;
- the text channel consolidates into the forecaster's notes and steers future
  task-dependent routing.

Both channels are ordinary parameter hosts: an economic function is its own
host, so ``step`` takes each stage alongside the notes' memory backend.
"""

import asyncio
import logging

from _economics_utils import HAIKU, SONNET
from _utils import display, rule
from pydantic import BaseModel, Field
from strands import tool

from ai_functions import ai_function
from ai_functions.experimental.economics import LLMForecaster, RoutingMemory, routed, spend
from ai_functions.experimental.economics.search import Greedy
from ai_functions.memory import JSONMemoryBackend
from ai_functions.optimizer import TextGradOptimizer


# A canned "web search" so this example runs with no API key. The findings are
# fixed and deliberately thin — one primary source, one secondary — which is
# what lets the downstream feedback ("needed primary sources") teach the
# forecaster something. Swap in a real search tool (see examples/_utils.py's
# get_websearch_tool) to run it against the live web.
@tool
def web_search(query: str) -> str:
    """Search the web for the query and return results with source URLs."""
    del query
    return (
        "1. The EU AI Act's obligations for general-purpose AI (GPAI) models took effect in "
        "August 2025: providers must publish training-data summaries and technical "
        "documentation. https://digital-strategy.ec.europa.eu/en/policies/ai-act\n"
        "2. GPAI models deemed to pose systemic risk face additional model-evaluation and "
        "incident-reporting duties. https://artificialintelligenceact.eu/"
    )


class Sources(BaseModel):
    findings: list[str] = Field(description="Key facts found, each with its source URL")


class Report(BaseModel):
    title: str
    body: str = Field(description="The report, citing the provided findings")


# One RoutingMemory field per routed stage: `notes` is the casebook the
# backward pass consolidates downstream feedback into (its format lives in the
# library-provided field description); `stats` is the observed attempt record
# the empirical beliefs persist and reload, so routing survives restarts.
class ForecastMemory(BaseModel):
    researcher_routing: RoutingMemory = Field(default_factory=RoutingMemory)
    writer_routing: RoutingMemory = Field(default_factory=RoutingMemory)


def cited(result: Sources):
    """Local check only: findings carry sources. Whether they are USEFUL is
    decided downstream and arrives later via settlement."""
    from ai_functions.ai_thread import PostConditionResult

    if not all("http" in f for f in result.findings):
        return PostConditionResult(passed=False, message="Every finding needs a source URL")
    return None


# Shared learned state: persisted across processes, readable after settlement.
memory = JSONMemoryBackend(
    schema=ForecastMemory, actor_id="economics-demo", path="./forecast_memory.json", model=SONNET.model
)


# Each stage: an LLM forecaster layered over learned statistics. The
# forecaster reads the task, so "quick factual query" can route cheap while
# "compare regulatory regimes" can route to a strong model. Value is declared
# once, on the decorator — the beliefs receive it per call.
@routed(
    models=[HAIKU, SONNET],
    value=0.05,  # good sources are worth 5 cents
    beliefs=LLMForecaster(memory=memory, memory_key="researcher_routing"),
    budget=0.10,
    policy=Greedy(),
)
@ai_function[Sources](tools=[web_search], post_conditions=[cited])
def research(query: str):
    """Research this topic on the web and return the key findings with sources:

    {query}"""


# The higher value favours the stronger model over the cheap one under the
# Greedy policy (highest net value wins).
@routed(
    models=[HAIKU, SONNET],
    value=0.25,  # a good report is worth 25 cents
    beliefs=LLMForecaster(memory=memory, memory_key="writer_routing"),
    budget=0.15,
    policy=Greedy(),
)
@ai_function[Report]
def write(query: str, findings: Sources):
    """Write a concise report answering: {query}

    Base it strictly on these findings, citing sources:
    {findings}"""


async def main():
    logging.basicConfig(level=logging.WARNING)

    rule("A report pipeline of two routed stages")

    query = "What changed in the EU AI Act's obligations for general-purpose models in 2025?"

    # Traced composition, exactly as with plain AIFunctions. Each stage
    # routes internally; the trace records its decision node.
    sources = await research.trace(query=query)
    report = await write.trace(query=query, findings=sources)

    display("Report", f"# {report.value.title}\n\n{report.value.body}")
    total = await spend(sources) + await spend(report)
    display("Spend", f"workflow total: ${total:.4f} (booked attempts, from the event log)", lang="text")

    rule("Feedback settles both stages")

    # The user judges the END result; backward distributes it. Each stage's
    # routing decision is a grad-enabled parameter: the score it receives
    # settles that stage's records (numeric channel), and the part attributable
    # to weak inputs flows on to the research stage, whose forecaster notes are
    # consolidated (text channel). Both channels are ordinary parameter hosts —
    # an economic function is its own host, so pass each stage alongside the
    # notes' memory backend.
    optimizer = TextGradOptimizer()
    await optimizer.step(
        report,
        "Too shallow: the report paraphrases news coverage. It needed primary sources "
        "(the Act's text, Commission guidance), and the writing itself was fine.",
        backends=[memory, research, write],
    )

    lines = ["after settlement:", ""]
    for fn, key in ((research, "researcher_routing"), (write, "writer_routing")):
        lines.append(f"{key} candidates (statistics now reflect downstream worth, not just local checks):")
        for label, summary in fn.beliefs.stats().items():
            lines.append(f"  {label:<18} {summary}")
        lines.append(f"{key} casebook (steers future task-dependent routing):")
        lines.append(f"  {(await memory.recall(f'{key}/notes')).value or '(none yet)'}")
        lines.append("")

    display("Learned state", "\n".join(lines), lang="text")
    memory.close()  # persist notes and stats for the next process


if __name__ == "__main__":
    asyncio.run(main())
