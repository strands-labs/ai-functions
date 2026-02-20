"""
Example of workflow to write a report using multi-agent orchestration. This shows:
- let an orchestrator agent use other agents (websearch, planner, critique) as tools
- use post-conditions to correct results of sub-agents before returning them to the orchestrator
- use ai_functions as class methods
"""
from pathlib import Path
from typing import Any, Sequence, Literal

from strands import tool
from strands.tools import ToolProvider

from ai_functions import ai_function
import utils
from ai_functions.types import PostConditionResult, CodeExecutionMode, AIFunctionConfig
from pydantic import Field, BaseModel

# find a websearch tool for which we have the API key in the env
websearch_tool = utils.get_websearch_tool()

FAST_MODEL = AIFunctionConfig(model="global.anthropic.claude-haiku-4-5-20251001-v1:0")

# === SEARCH AGENT ===

# Define a reusable search agent with various correctness checks.
# In a larger project, this function would be part of a library and be imported and used as any other python function.

def check_length(summary: str, max_words: int):
    assert len(summary.split()) <= max_words

@ai_function
def check_citations(summary: str) -> PostConditionResult:
    """
    Validate if all the claims made in the following summary are supported by an inline citation.
    <summary>
    {summary}
    </summary>
    """

@ai_function(
    config=FAST_MODEL,
    description="A web search agent that researches `query` (a description of the search task in natural language)"
                "and writes a summary of its finding. Optionally use `max_words` to specify the maximum summary length",
    tools=[websearch_tool],
    post_conditions=[check_length],
)
def websearch_agent(query: str, max_words: int = 500) -> str:
    """
    Perform a web search on the following topic and return a summary of your findings.
    <query>
    {query}
    </query>

    RULES:
    - The summary must be at most of {max_words} long.
    - Every claim in the summary should be supported by citations (in markdown format) to the sources you found.
    - Use a bullet point format for the summary.
    - Squeeze as much information as possible in the report with no concern for the writing style.
    """

# === PLANNER AGENT ===

# define a planner agent to perform an initial cursory search and decide the structure of the report

class ReportPlan(BaseModel):
    sections: list[str] = Field(..., description="List of descriptions of sections to include in the report. "
                                                 "Each section entry should list the arguments to cover in the section.")
    research_topics: list[str] = Field(..., description="List of topics to research before writing the report.")

@ai_function(
    description="Tool to suggest the plan and organization of a report."
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

    If needed, perform an initial cursory websearch to understand the topic and figure out what topics should be covered.
    Then,
    """

# === REPORT ===

# Define a Report object to keep track of the current state of the report. Provides tools to add to the report,
# and an ai_function to critique the current report.

class Report:
    def __init__(self, path: Path | str):
        self._sections: list[str] = []
        self._path = Path(path)

    @tool
    def add_section(self, title: str, section_content: str):
        """
        Add a new section to the report and save it.
        """
        self._sections.append(f"## {title}\n\n{section_content}")
        self._path.write_text(self.to_markdown())

    @ai_function(description="Give constructive criticism on the current state of the report.")
    def critique_report(self) -> str:
        return t"""
        Provide a constructive critique of the following report.
        {self.to_markdown()}
        """

    def to_markdown(self) -> str:
        return "\n\n".join(self._sections)

# === Orchestrator ===

def main():
    report = Report('multiagent_report.md')

    @ai_function(
        tools=[report_planner, websearch_agent, report.add_section, report.critique_report],
        code_execution_mode=CodeExecutionMode.LOCAL
    )

    def report_orchestrator(topic: str) -> Literal["done"]:
        """
        Write a report on the following topic:
        <topic>
        {topic}
        </topic>

        RULES:
        - The report should have an executive summary. Write the executive summary last.
        - The report should provide citations to support the claims
        """

    report_orchestrator("recent practical advances in quantum computing")

if __name__ == '__main__':
    main()