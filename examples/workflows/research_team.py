"""Multi-agent orchestration — write a report using a team of agents.

Shows:
- an orchestrator agent that uses other agents (websearch, planner, critique)
  as tools
- post-conditions that correct sub-agent results before returning them to the
  orchestrator
- an ``@ai_function`` used as a class method

Requires a websearch API key (TAVILY_API_KEY or EXA_API_KEY) in the environment.
"""

from pathlib import Path
from typing import Literal

from example_helpers import models
from example_helpers.utils import display, get_websearch_tool
from pydantic import BaseModel, Field
from strands import tool

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult

websearch_tool = get_websearch_tool()


def check_length(summary: str, max_words: int):
    assert len(summary.split()) <= max_words


@ai_function(model=models.medium)
def check_citations(summary: str) -> PostConditionResult:
    """
    Validate if all the claims made in the following summary are supported by an inline citation.
    <summary>
    {summary}
    </summary>
    """


@ai_function(
    model=models.small,
    description="A web search agent that researches `query` (a description of the search task in natural language) "
    "and writes a summary of its finding. Optionally use `max_words` to specify the maximum summary length",
    tools=[websearch_tool],
    post_conditions=[check_length],
)
def websearch_agent(query: str, max_words: int = 150) -> str:
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


class ReportPlan(BaseModel):
    sections: list[str] = Field(
        ...,
        description="List of descriptions of sections to include in the report. "
        "Each section entry should list the arguments to cover in the section.",
    )
    research_topics: list[str] = Field(..., description="List of topics to research before writing the report.")


@ai_function(
    model=models.medium,
    description="Tool to suggest the plan and organization of a report. "
    "It will also suggest some initial topics to research. "
    "Call this tool before starting to write the report.",
    tools=[websearch_tool],
)
def report_planner(topic: str) -> ReportPlan:
    """
    Generate a plan to write a report on the following topic:
    <topic>
    {topic}
    </topic>

    If needed, perform an initial cursory websearch to understand the topic and figure out what topics
    should be covered.
    """


class Report:
    def __init__(self, path: Path | str):
        self._sections: list[str] = []
        self._path = Path(path)

    @tool
    def add_section(self, title: str, section_content: str):
        """Add a new section to the report and save it."""
        self._sections.append(f"## {title}\n\n{section_content}")
        self._path.write_text(self.to_markdown())

    @ai_function(model=models.medium, description="Give constructive criticism on the current state of the report.")
    def critique_report(self) -> str:
        return f"""
        Provide a constructive critique of the following report.

        {self.to_markdown()}
        """

    def to_markdown(self) -> str:
        return "\n\n".join(self._sections)


def main():
    report = Report(Path(__file__).parent / "multiagent_report.md")

    # The orchestrator only drives tools; it doesn't run Python, so LOCAL code
    # execution is left off here. See stock_report.py / ../memory/procedural.py
    # for code_execution_mode=LOCAL.
    @ai_function(
        model=models.small,
        tools=[report_planner, websearch_agent, report.add_section, report.critique_report],
    )
    def report_orchestrator(topic: str) -> Literal["done"]:
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
    display("Report", report.to_markdown())


if __name__ == "__main__":
    main()
