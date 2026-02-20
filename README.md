# Strands AI Functions

Strands AI Functions is a Python library for building reliable AI-powered applications through a new abstraction: functions that behave like standard Python functions, but are evaluated by reasoning AI Agents.

AI Functions extend the expressivity of standard programming by offering developers a computational model that can solve tasks not easily expressible as traditional code. They can both leverage text generation capabilities (e.g., to write summaries or retrieve information) and dynamically generate and execute code to process inputs and return native Python objects. For example, an AI Function can load a user-uploaded file in an arbitrary format and convert it to a normalized `DataFrame` for use in the rest of the workflow.

Direct integration of AI agents in standard workflows is often avoided due to their non-deterministic nature and lack of assurance that instructions will be followed, which can cause cascading errors throughout the workflow. AI Functions address this through extensive use of *post-conditions*. Unlike traditional prompt-based approaches, which try to ensure correctness by relying on prompt engineering alone, AI Functions enforce correctness through runtime post-condition checking: users can specify explicit post-conditions that the output of any given step needs to satisfy. AI Functions will automatically initiate self-correcting loops to ensure these properties are respected, avoiding cascading errors in complex workflows.

Through AI Functions, developers can construct agentic workflows and agent graphs — including asynchronous ones — by writing and composing functions. They can build shareable libraries of robust, reusable agentic flows in exactly the same way they build software libraries today, and can use standard software development practices to collaborate on refining and ensuring the safety of each component.


## Getting Started

### Prerequisites

- Python 3.12 or higher (Python 3.14+ recommended for all features)
- Valid credentials for a supported model provider (AWS Bedrock, OpenAI, etc.)
- (Recommended) [uv](https://docs.astral.sh/uv/getting-started/installation/) to run the provided examples


### Installation

```bash
# Using pip
pip install strands-ai-functions
# Using uv
uv add strands-ai-functions
```

### Configure Model Provider

Strands AI Functions support various [model providers](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/). Change the `model` option in the examples below to use a different provider, model or authentication options (see also [Configuring Credentials](https://strandsagents.com/latest/documentation/docs/user-guide/quickstart/python/#configuring-credentials)). For example:

```python
from ai_functions import ai_function
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel

# Use Claude Sonnet on Amazon Bedrock (default if `model` is not specified)
model = BedrockModel(model_id="anthropic.claude-sonnet-4-20250514-v1:0")

# Or use a different provider and model
model = OpenAIModel(client_args={"api_key": "<KEY>"}, model_id="gpt-4o")

@ai_function(model=model)
def my_function() -> None:
    ...
```

### AI Function Basics

Below is a basic example of AI Functions in action to build a simple meeting summarization workflow with validation. The @ai_function decorator will automatically ensure the model output a result using the desired return type. The result is validated using the provided post-conditions: if any of them fail, the model is automatically prompted to correct the errors and try again. The function only returns when all properties pass.   


```python
import textwrap

from pydantic import BaseModel

from ai_functions import ai_function
from ai_functions.types import PostConditionResult


# We start by defining the structured output type for our meeting summarization agent
# AI Functions can return any data type: primitive (str, int, ...), json-serializable (pydantic models) and general python objects (numpy arrays, ...)
# The library takes care of the necessary conversions and validation under the hood
class MeetingSummary(BaseModel):
    attendees: list[str]
    summary: str
    action_items: list[str]

    
# Post conditions can be any python function validating the output...
def check_length(response: MeetingSummary):
    """Post-condition: summary must be less than 50 words."""
    length = len(response.summary.split())
    assert len(response.summary.split()) < 50, "Summary must be less than 50 words long."

# ... or they can be ai_functions, since ai_functions *are* just functions
@ai_function
def check_style(response: MeetingSummary) -> PostConditionResult:
    """
    Check if the summary below satisfies the following criteria:
    - It must use bullet points
    - It must provide the reader with the necessary context

    <summary>
    {response.summary}
    </summary>
    """

# Finally we define the main ai_function, specifying the desired behavior both through prompt 
# (which in this case is generated automatically from the docstring using the provided arguments)
# and the provided post-conditions. The library will ensure the result pass all the
# requirements before returning it.
@ai_function(post_conditions=[check_length, check_style], max_attempts=5)
def summarize_meeting(transcripts: str) -> MeetingSummary:
    """
    Write a summary of the following meeting in less than 50 words.
    <transcripts>
        {transcripts}
    </transcripts>
    """

# `summarize_meeting` can now be called just like any other function inside the code
if __name__ == '__main__':
    transcripts = "..."
    # `meeting_summary` will be an instance of `MeetingSummary`
    meeting_summary = summarize_meeting(transcripts)
    print(meeting_summary)
```

### Multi-Agent Workflows

Below is a more realistic example of a complex workflow, including the use of async functions to run multiple agents in parallel, and the use of Python integration to process native data types like `pandas.DataFrame`. See `examples/stock_report.py` for the complete example with additional functionalities.

```python
from ai_functions import ai_function
import asyncio, datetime, pandas as pd 
from typing import Literal
from dataclasses import dataclass
from strands_tools import exa
from pathlib import Path

@dataclass
class StockInfo:
    symbol: str
    news: str
    prices: pd.DataFrame
    
websearch_model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# ai_functions work with any model and tool compatible with Strands 
@ai_function(model=websearch_model, tools=[exa])
async def research_news(stock: str) -> str:
    """
    Research and summarize the current news regarding the following stock symbol: {stock}
    """

# Agents can optionally access python execution environment, allowing it to use libraries and receive and return native Python objects.
@ai_function(code_execution_mode="local", code_executor_additional_imports=["pandas", "yfinance"])
async def research_price(stock: str, period: datetime.timedelta) -> pd.DataFrame:
    """
    Use the `yfinance` Python package to retrieve the historical prices of {stock} in the last 30 days.
    Return a dataframe with columns: ["date", "price" (float, price at market close)]
    """

# Function inputs (e.g., `stock_info`) are available inside the Python environment for further processing
@ai_function(code_execution_mode="local", code_executor_additional_imports=["pandas", "plotly.*"])
def write_report(stock_info: list[StockInfo]) -> str:
    """
    Write a html report analyzing the trend of the following stock symbol: {stock_info}.
    Use the information in `stock_info` for your analysis. Use `plotly` to embed plots illustrating the trend. Return the full html content.
    """

# Asynchronous agent graphs can be constructed combining standard Python function calls
async def research_stock(stock: str) -> StockInfo:
    # Run the two functions in parallel
    news, prices = await asyncio.gather(research_news(stock), research_price(stock))
    return StockInfo(stock, news, prices)

async def write_stock_report(stocks: list[str], output_path: Path):
    # gather information about all stocks in parallel:
    stock_info = await asyncio.gather(*(research_stock(stock) for stock in stocks))
    # Use their results to write a report
    report = write_report(stock_info)
    output_path.write_text(report)
```

AI Functions can also be used as tools by other agents to build multi-agent systems with orchestration:
```python
@ai_function(
    description="Perform multiple web searches relevant to query and returns a summary of the results",
    tools=[...]
)
def websearch(query: str) -> str:
    """
    Perform a web search on the following topic and return a summary of your findings.
    ---
    {query}
    """
    
@ai_function(tools=[websearch])
def report_writer(topic: str) -> str:
    """
    Research the following topic and write a report.
    ---
    {topic}
    """
```


## Tutorial

See the [tutorial](docs/tutorial.md).

## Security

The `"local"` execution mode uses AST-based validation of the generated code with controlled imports and timeouts. The validation attempts to prevent malicious imports and block dangerous operations, but does not offer sandboxing and does not prevent resource exhaustion (e.g., infinite loops, excessive memory allocation). For production deployments, run AI Functions inside a container or other isolated environment to provide additional protection against resource exhaustion and process-level isolation. Use `"disabled"` mode for untrusted input or restricted environments. Limit imports to necessary packages and monitor execution in production.

## Examples

This repository includes several complete examples demonstrating different capabilities. To run the examples:
```bash
# Clone the repository
git clone https://github.com/strandslabs/strands-ai-functions
cd strands-ai-functions/examples

# optional: set env variable to enable rich tool visualization in the terminal
export STRANDS_TOOL_CONSOLE_MODE="enabled"

# run the examples using uv (recommended)
uv run meeting_summary.py
```

**Note**: Configure model provider credentials before running examples (see [Configure Model Provider](#configure-model-provider)). You may need to change the examples to use a different model provider.