"""Proactive summarization — a long-running agent that compacts its own history.

An agent writes a multi-chapter article over a persistent thread, one chapter
per ``run()`` call. ``ThreadConfig.summarization_threshold`` sets a proactive
summarization threshold: when the accumulated history exceeds it at the start of
a cycle, the runtime compacts the conversation before the model call instead of
waiting for a context-window overflow. The emitted ``ContextSummarizedEvent`` s
(printed at the end) show when summarization kicked in.
"""

import asyncio
from typing import Literal

from example_helpers import models
from example_helpers.utils import display, rule
from pydantic import BaseModel
from strands import tool

from ai_functions import ai_function
from ai_functions.ai_thread.summarization import DefaultSummarizationStrategy
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types.events import ContextSummarizedEvent

CHAPTERS = [
    "Introduction: what quantum error correction is and why it matters",
    "Physical vs. logical qubits and the threshold theorem",
    "Surface codes and stabilizer measurement",
    "Recent hardware milestones (below-threshold demonstrations)",
    "Outlook: the road to fault-tolerant quantum computing",
]


class Chapter(BaseModel):
    title: str
    content: str


class Article:
    """Accumulates the article as the agent writes each chapter."""

    def __init__(self) -> None:
        self.chapters: list[Chapter] = []

    @tool
    def write_chapter(self, title: str, content: str) -> str:
        """Add one chapter (title + a few paragraphs) to the article."""
        self.chapters.append(Chapter(title=title, content=content))
        return f"Chapter '{title}' saved ({len(self.chapters)} total)."

    def to_markdown(self) -> str:
        return "\n\n".join(f"## {chapter.title}\n\n{chapter.content}" for chapter in self.chapters)


async def main() -> None:
    article = Article()

    strategy = DefaultSummarizationStrategy(
        summarize_by_forking=False,
        preserve_min_messages=1,
        preserve_min_tokens=0,
        preserve_max_tokens=1200,
    )

    @ai_function(
        model=models.small,
        tools=[article.write_chapter],
        summarization_strategy=strategy,
        summarization_threshold=4000,
        # low enough to trigger proactive compaction mid-article
    )
    def chapter_writer(instruction: str) -> Literal["done"]:
        """You are writing one chapter of a research article on quantum error correction.

        {instruction}

        Write the chapter by calling `write_chapter` exactly once with a title and
        2-3 substantive paragraphs.
        """

    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()
    handle = await worker.spawn_locally(chapter_writer, thread_name="chapter_writer")

    rule(f"Writing a {len(CHAPTERS)}-chapter article (one cycle per chapter)")
    for i, topic in enumerate(CHAPTERS, 1):
        instruction = f"Write chapter {i} of {len(CHAPTERS)}. Topic: {topic}."
        _ = await handle.run(instruction=instruction)
        events = await coord.get_events(handle.id)
        n_summaries = sum(isinstance(e, ContextSummarizedEvent) for e in events)
        chapter = article.chapters[-1]
        display(f"Chapters {i}. Summarizations: {n_summaries}", chapter.content, lang="md")

    events = await coord.get_events(handle.id)
    summaries = [e for e in events if isinstance(e, ContextSummarizedEvent)]
    lines = [f"Proactive summarizations during the run: {len(summaries)}"]
    if summaries:
        # The first compaction replaced the history prefix with a summary turn.
        first = summaries[0].new_history[0]
        preview = getattr(first, "text", "")[:200]
        lines.append(f"First summary turn: {preview!r}")
    display("Summarization", "\n".join(lines), lang="text")

    display("Article", article.to_markdown())

    await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
