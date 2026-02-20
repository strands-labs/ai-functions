import logging
from typing import Literal

from pydantic import BaseModel
from strands import tool

from ai_functions import ai_function
from ai_functions.context_management.context_manager import ContextManager
from ai_functions.context_management.summarizing_window_manager import SummarizingWindowConversationManager

model = 'global.anthropic.claude-sonnet-4-5-20250929-v1:0'
fast_model = 'global.anthropic.claude-haiku-4-5-20251001-v1:0'


# Configure logging to see when managers are triggered
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)


class Chapter(BaseModel):
    chapter_number: int
    title: str
    content: str
    key_concepts: list[str]


class ArticleState:
    def __init__(self):
        self.chapters: list[Chapter] = []

    @tool
    def write_chapter(self, title: str, content: str, key_concepts: list[str]) -> str:
        chapter = Chapter(chapter_number=len(self.chapters) + 1,
                          title=title,
                          content=content,
                          key_concepts=key_concepts)
        self.chapters.append(chapter)
        return "Chapter added successfully."

    @tool
    def read_book(self) -> str:
        return self.to_markdown()

    def check_length(self, target_length: int):
        assert len(self.chapters) >= target_length

    def to_markdown(self) -> str:
        parts = []
        for chapter in self.chapters:
            parts.append(f"\n## {chapter.title}\n")
            parts.append("*Key concepts:*")
            for concept in chapter.key_concepts:
                parts.append(f"  - {concept}")
            parts.append("\n")
            parts.append(chapter.content)
        return "\n".join(parts)


article_state = ArticleState()

@ai_function(model=model, tools=[article_state.read_book])
def summarizer() -> str:
    """
    Summarize in one paragraph the content of the article so far, and explain what still needs to be written.
    """

@ai_function(
    model=model,
    tools=[article_state.write_chapter],
    hooks=[ContextManager(
        manage_conversation_every_cycle=True,  # Check and manage cache after each tool call
        max_non_cache_tokens=1024,  # Reset cache when uncached tokens exceed this (low to demo)
    )],
    conversation_manager=SummarizingWindowConversationManager(
        summarization_function=summarizer, # AI Function that will provide the summary
        max_tokens=1024,  # Trigger summarization when total tokens exceed this (low to demo)
        preserve_recent_messages=2,
    ),  # Use TaskConversationManager for summarization
    post_conditions=[lambda _: article_state.check_length(target_length=5)]
)
def write_article(
        article_title: str,
        field: str,
) -> Literal["done"]:
    """
    Write research article titled "{article_title}" about {field}.
    Add exactly 5 chapters to the article using the provided tool.
    Do not call multiple tools in parallel. Wait for confirmation before moving to the next chapter.
    """


if __name__ == '__main__':
    print("Generating an article\n(This will take a few minutes)...")

    # Write an entire article over multiple tool calls
    # This will create a long conversation that triggers summarization
    write_article(
        article_title="Advances in Quantum Computing Error Correction",
        field="quantum computing"
    )
    print("\n\n... generation successful")
    print("=" * 70)
    print(article_state.to_markdown())
