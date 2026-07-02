"""Proactive summarization — a long-running agent that compacts its own history.

An agent writes a multi-chapter article over a persistent thread, one chapter
per ``run()`` call. ``ThreadConfig.summarization_threshold`` sets a proactive
summarization threshold: when the accumulated history exceeds it at the start of
a cycle, the runtime compacts the conversation before the model call instead of
waiting for a context-window overflow. The emitted ``ContextSummarizedEvent`` s
(printed at the end) show when summarization kicked in.
"""

import asyncio

from strands import tool

from ai_functions import ai_function
from ai_functions.ai_thread.summarization import DefaultSummarizationStrategy
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types.events import ContextSummarizedEvent

MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CHAPTERS = [
    "Introduction: what quantum error correction is and why it matters",
    "Physical vs. logical qubits and the threshold theorem",
    "Surface codes and stabilizer measurement",
    "Recent hardware milestones (below-threshold demonstrations)",
    "Outlook: the road to fault-tolerant quantum computing",
]


class Article:
    """Accumulates the article as the agent writes each chapter."""

    def __init__(self) -> None:
        self.chapters: list[tuple[str, str]] = []

    @tool
    def write_chapter(self, title: str, content: str) -> str:
        """Add one chapter (title + a few paragraphs) to the article."""
        self.chapters.append((title, content))
        return f"Chapter '{title}' saved ({len(self.chapters)} total)."

    def to_markdown(self) -> str:
        return "\n\n".join(f"## {title}\n\n{content}" for title, content in self.chapters)


async def main() -> None:
    article = Article()

    # Keep the preserved tail well below the threshold so one compaction brings
    # the history under it and does not re-fire to the per-cycle cap (see the
    # summarization_threshold convergence note). preserve_min_messages=1 avoids a
    # floor that could exceed the threshold when individual turns are large.
    strategy = DefaultSummarizationStrategy(
        summarize_by_forking=False,
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=1200,
    )

    @ai_function(
        str,
        model=MODEL,
        tools=[article.write_chapter],
        summarization_strategy=strategy,
        summarization_threshold=4000,  # low enough to trigger proactive compaction mid-article
    )
    def chapter_writer(instruction: str):
        """You are writing one chapter of a research article on quantum error correction.

        {instruction}

        Write the chapter by calling `write_chapter` exactly once with a title and
        2-3 substantive paragraphs. Then reply with the chapter title.
        """

    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()
    handle = await worker.spawn_locally(chapter_writer, thread_name="chapter_writer")

    print(f"Writing a {len(CHAPTERS)}-chapter article (one cycle per chapter)...\n")
    for i, topic in enumerate(CHAPTERS, 1):
        instruction = f"Write chapter {i} of {len(CHAPTERS)}. Topic: {topic}."
        title = await handle.run(instruction=instruction)
        events = await coord.get_events(handle.id)
        n_summaries = sum(isinstance(e, ContextSummarizedEvent) for e in events)
        print(f"  chapter {i}: {title.strip()}   [summarizations so far: {n_summaries}]")

    events = await coord.get_events(handle.id)
    summaries = [e for e in events if isinstance(e, ContextSummarizedEvent)]
    print(f"\nProactive summarizations during the run: {len(summaries)}")
    if summaries:
        first = summaries[0].new_history[0]
        preview = getattr(first, "text", "")[:200]
        print(f"First compaction replaced the prefix with a summary turn:\n  {preview!r}")

    print("\n=== Article ===")
    print(article.to_markdown())

    await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
