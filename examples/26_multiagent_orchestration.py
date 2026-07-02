"""Multi-agent orchestration — write a report using a team of agents.

Shows:
- an orchestrator agent that uses other agents (websearch, planner, critique)
  as tools
- post-conditions that correct sub-agent results before returning them to the
  orchestrator
- an ``@ai_function`` used as a class method

Requires a websearch API key (TAVILY_API_KEY or EXA_API_KEY) in the environment.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from strands import tool

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult
from ai_functions.ai_thread.config import ThreadConfig


def get_websearch_tool():
    """Return a Strands websearch tool for whichever API key is in the environment."""
    if os.environ.get("EXA_API_KEY"):
        from strands_tools import exa as websearch_tool
    elif os.environ.get("TAVILY_API_KEY"):
        from strands_tools import tavily as websearch_tool
    else:
        raise ValueError("Set EXA_API_KEY or TAVILY_API_KEY to run this example.")
    return websearch_tool


websearch_tool = get_websearch_tool()

FAST_MODEL = ThreadConfig(model="global.anthropic.claude-haiku-4-5-20251001-v1:0")


# === SEARCH AGENT ===


def check_length(summary: str, max_words: int):
    assert len(summary.split()) <= max_words


@ai_function(PostConditionResult)
def check_citations(summary: str):
    """
    Validate if all the claims made in the following summary are supported by an inline citation.
    <summary>
    {summary}
    </summary>
    """


@ai_function(
    str,
    config=FAST_MODEL,
    description="A web search agent that researches `query` (a description of the search task in natural language) "
    "and writes a summary of its finding. Optionally use `max_words` to specify the maximum summary length",
    tools=[websearch_tool],
    post_conditions=[check_length],
)
def websearch_agent(query: str, max_words: int = 150):
    """
    Perform a web search on the following topic and return a summary of your findings.
    <query>
    {query}
    </query>

    RULES:
    - The summary must be at most {max_words} words long.
    - Every claim in the summary should be supported by citations (in markdown format) to the sources you found.
    - Use a bullet point format for the summary.
    - Squeeze as much information as possible in the report with no concern for the writing style.
    """


# === PLANNER AGENT ===


class ReportPlan(BaseModel):
    sections: list[str] = Field(
        ...,
        description="List of descriptions of sections to include in the report. "
        "Each section entry should list the arguments to cover in the section.",
    )
    research_topics: list[str] = Field(..., description="List of topics to research before writing the report.")


@ai_function(
    ReportPlan,
    description="Tool to suggest the plan and organization of a report. "
    "It will also suggest some initial topics to research. "
    "Call this tool before starting to write the report.",
    tools=[websearch_tool],
)
def report_planner(topic: str):
    """
    Generate a plan to write a report on the following topic:
    <topic>
    {topic}
    </topic>

    If needed, perform an initial cursory websearch to understand the topic and figure out what topics
    should be covered.
    """


# === REPORT ===


class Report:
    def __init__(self, path: Path | str):
        self._sections: list[str] = []
        self._path = Path(path)

    @tool
    def add_section(self, title: str, section_content: str):
        """Add a new section to the report and save it."""
        self._sections.append(f"## {title}\n\n{section_content}")
        self._path.write_text(self.to_markdown())

    @ai_function(str, description="Give constructive criticism on the current state of the report.")
    def critique_report(self) -> str:
        return f"""
        Provide a constructive critique of the following report.
        {self.to_markdown()}
        """

    def to_markdown(self) -> str:
        return "\n\n".join(self._sections)


# === Orchestrator ===


def main():
    report = Report(Path(__file__).parent / "multiagent_report.md")

    # The orchestrator drives other agents/tools via normal tool-use and returns
    # a structured ``Literal["done"]`` — it doesn't run Python, so LOCAL code
    # execution is intentionally NOT enabled here (it would add the python_executor
    # tool + sandbox system prompt to every call, bloating the context and
    # triggering overflow). See examples 14/27 for code_execution_mode=LOCAL.
    @ai_function(
        Literal["done"],
        config=FAST_MODEL,
        tools=[report_planner, websearch_agent, report.add_section, report.critique_report],
    )
    def report_orchestrator(topic: str):
        """
        Write a SHORT report on the following topic:
        <topic>
        {topic}
        </topic>

        RULES:
        - Keep it small so it fits in context: at most 4 sections, each 1-2
          short paragraphs. Research each section with a single websearch_agent
          call (its default length is fine — do not ask for long summaries).
        - Call critique_report at most once, near the end.
        - Include an executive summary section. Write the executive summary last.
        - Provide citations to support the claims.
        """

    report_orchestrator.run_sync(topic="recent practical advances in quantum computing")
    print("=== Report ===")
    print(report.to_markdown())


if __name__ == "__main__":
    main()
