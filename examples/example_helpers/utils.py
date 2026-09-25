"""Shared display helpers for the examples."""

from __future__ import annotations

import os

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

# One console shared across every example so output stays visually consistent.
console = Console()


def display(title: str, content: str, lang: str = "markdown") -> None:
    """Render ``content`` in a titled panel with syntax highlighting.

    ``lang`` selects the highlighter (``markdown`` for prose, ``json``/``python``
    for structured output, ``html`` for reports, and so on).
    """
    body = Syntax(content, lang, theme="monokai", word_wrap=True)
    console.print(Panel(body, title=title, border_style="cyan", expand=True))


def rule(title: str) -> None:
    """Print a horizontal divider to mark a step in a multi-stage example."""
    console.print(Rule(title, style="cyan"))


def get_websearch_tool():
    """Return a Strands websearch tool for whichever API key is in the environment.

    Prefers Exa, falls back to Tavily, and raises if neither key is set.
    """
    if os.environ.get("EXA_API_KEY"):
        from strands_tools import exa as websearch_tool
    elif os.environ.get("TAVILY_API_KEY"):
        from strands_tools import tavily as websearch_tool
    else:
        raise ValueError("Set EXA_API_KEY or TAVILY_API_KEY to run this example.")
    return websearch_tool
