# Strands AI Functions

Strands AI Functions is a Python library for building reliable AI-powered applications through a new abstraction: functions that behave like standard Python functions, but are evaluated by reasoning AI Agents.

AI Functions extend the expressivity of standard programming by offering developers a computational model that can solve tasks not easily expressible as traditional code. They can both leverage generative capabilities (e.g., writing summaries or retrieving information) and can dynamically generate and execute code to process inputs and return native Python objects. For example, an AI Function can load a user-uploaded file in an arbitrary format and convert it to a normalized `DataFrame` for use in the rest of the workflow.

Direct integration of AI agents in standard workflows is often avoided due to their non-deterministic nature and lack of assurance that instructions will be followed, which can cause cascading errors throughout the workflow. AI Functions address this through extensive use of *post-conditions*. Unlike traditional prompt-based approaches, which try to ensure correctness by relying on prompt engineering alone, AI Functions enforce correctness through runtime post-condition checking: users can specify explicit post-conditions that the output of any given step needs to satisfy. AI Functions will automatically initiate self-correcting loops to ensure these properties are respected, avoiding cascading errors in complex workflows.

Through AI Functions, developers can construct agentic workflows and agent graphs — including asynchronous ones — by writing and composing functions. They can build shareable libraries of robust, reusable agentic flows in exactly the same way they build software libraries today, and can use standard software development practices to collaborate on refining and ensuring the safety of each component.

## Getting Started

While the minimum supported version if Python >=3.12, we recommend using Python >=3.14 to support all features.
We also recommend using `uv` (see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)) to run the provided examples.

To install the Strands AI Functions extension, run:
```bash
# using pip:
pip install strands-ai-functions
# using, if using uv, add strands-ai-function as a dependency to your project:
uv add strands-ai-functions
```

This repo provides several examples. To run the examples, first configure the credentials for one of the supported model providers (see [Configuring Credentials](https://strandsagents.com/latest/documentation/docs/user-guide/quickstart/python/#configuring-credentials)).
Then, clone the repo and run the examples using `uv` from within their folder: 
```bash
# clone the repo
git clone https://github.com/strandslabs/strands-ai-functions
cd strands-ai-functions/examples
# optional: set env variable to enable rich tool visualization in the terminal
export STRANDS_TOOL_CONSOLE_MODE="enabled"
# run the examples using uv
# (change the model settings inside the example if not using Bedrock as the model provider)
uv run [name_of_the_example].py
```

AI Functions use the same default model provider as Strands (Amazon Bedrock). You can change the model provider used in the examples by changing the `model` argument (see [Model Providers](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/)):
```python
from ai_functions import ai_function
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel

# Use Bedrock
model = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0"
)
# Alternatively, use OpenAI by just switching model provider
model = OpenAIModel(
    client_args={"api_key": "<KEY>"},
    model_id="gpt-4o"
)

@ai_function(model=model)
def my_function() -> None:
    """[...]"""
```

## AI Functions Basics

AI Functions behave like a standard function, but their code is written in natural language rather than Python, and are executed by an LLM rather than a CPU. To define an AI Function, we use the `@ai_function` decorator and specify what the function should do inside its docstring (we will cover alternative methods later).
```python
from ai_functions import ai_function

@ai_function
def translate_text(text: str, lang: str) -> str:
    """
    Translate the text below to the following language: {lang}.
    ---
    {text}
    """

text = 'It was the best of times, it was the worst of times'
for lang in ['fr', 'ja', 'it', 'zh']:
    translation = translate_text(text, lang=lang)
    print(translation)
```

When an AI Function is called, the library will automatically create an agent, generate a prompt based on the docstring template and the provided arguments, parse and validate the result and return it. From the outside, it behaves like any other Python function.    

AI Functions can return arbitrary data types, including primitive types (str, int, float), Pydantic models and even native Python objects (see next section). The following example shows how to build a simple meeting summarization workflow using structured output.

```python
from ai_functions import ai_function
from pydantic import BaseModel

class MeetingSummary(BaseModel):
    attendees: list[str]
    summary: str
    action_items: list[str]

@ai_function
def summarize_meeting(transcripts: str) -> MeetingSummary:
    """
    Write a summary of the following meeting in less than 50 words.
    <transcripts>
    {transcripts}
    </transcripts>
    """

if __name__ == "__main__":
    transcripts = "[add your meeting transcripts here]"
    meeting_summary = summarize_meeting(transcripts)
    
    print("=== Meeting Summary ===")
    print("Attendees:" + ", ".join(meeting_summary.attendees))
    print("Summary:\n" + meeting_summary.summary)
    print("Action Items:")
    for action_item in meeting_summary.action_items:
        print(action_item)
```

### Python Integration

AI Agents are usually limited working with serializable input-output types (strings, JSON-objects, ...) rather than with native objects of the programming language. AI Functions, on the other hand, aim to provide a natural extension of the programming language itself enabling new kind of programming patterns and abstractions. In particular, we optionally provide agents with a Python environment allowing them to dynamically generate code to process arbitrary input data and return native Python objects.

Consider for example a webapp that allows the user to upload an invoice in an arbitrary format (pdf, csv, json). 
The following snippet implements a "universal data loader" that given the path to a file inspects its content and automatically  decide the appropriate processing pipeline to load the file and convert it to a DataFrame in the desired format. It then uses another AI Function to apply transformations which cannot be expressed in pure Python.  See `examples/universal_loader.py` for a complete implementation that also validates the returned DataFrame using post-conditions (which we will introduce later).

```python
from ai_functions import ai_function 
from pandas import DataFrame, api 

def check_invoice_dataframe(df: DataFrame): 
    """Post-condition: validate DataFrame structure.""" 
    assert {'product_name', 'quantity', 'price', 'purchase_date'}.issubset(df.columns) 
    assert api.types.is_integer_dtype(df['quantity']), "quantity must be an integer" 
    assert api.types.is_float_dtype(df['price']), "price must be a float" 
    assert api.types.is_datetime64_any_dtype(df['purchase_date']), "purchase_date must be a datetime64" 
    assert not df.duplicated(subset=['product_name', 'price', 'purchase_date']).any(), "The combination of product_name, price, and purchase_date must be unique" 
 
# code execution has to be explicitly enabled  
@ai_function( 
    code_execution_mode="local", 
    code_executor_additional_imports=["pandas.*", "sqlite3", "json"],
    post_conditions=[check_invoice_dataframe],
) 
def import_invoice(path: str) -> DataFrame: 
    """ 
    The file `{path}` contains purchase logs. Extract them in a DataFrame with columns: 
    - product_name (str) 
    - quantity (int) 
    - price (float) 
    - purchase_date (datetime) 
    """ 
 
@ai_function( 
    code_execution_mode="local", 
    code_executor_additional_imports=["pandas.*"], 
    post_conditions=[check_invoice_dataframe], 
) 
def fuzzy_merge_products(invoice: DataFrame) -> DataFrame: 
    """ 
    Find product names that denote different versions of the same product, normalize them 
    by removing version suffixes and unifying spelling variants, update the product names 
    with the normalized names, and return a DataFrame with the same structure  
    (same number of rows and columns). 
    """ 
 
# Load a JSON (the agent has to inspect the JSON to understand how to map it to a DataFrame) 
df = import_invoice('data/invoice.json') 
print("Invoice total:", df['price'].sum()) 
 
# Load a SQLite database. The agent will dynamically check the schema and generate 
# the necessary queries to read it and convert it to the desired format) 
df = import_invoice('data/invoice.sqlite3') 
print(df)

# Merge revisions of the same product 
df = fuzzy_merge_products(df)
```

Right now Strands AI Function support only "local" execution. This will create a local Python environment (similar to a Jupyter notebook) for the agent to use. Execution in a safe remote sandboxed interpreter is a planned extension.

> [!CAUTION]
> The local execution environment attempts to restrict execution to explicitly allowed libraries and methods. However, executing Python code in a non-sandboxed environment is inherently unsafe. Please make sure you understand the risk and consider running the code inside a `docker` or other sandbox. 

### AI Function Configuration

AI Functions use Strands Agent in the backend. Any valid option of `strands.Agent` (such as `model`, `tools`, `system_prompt`) can be passed in the decorator.

```python
from ai_functions import ai_function
from strands_tools import file_read, file_write
from typing import Literal

@ai_function(tools=[file_read, file_write])
def summarize_file(path: str, output_path: str) -> Literal["done"]:
    """
    Read the file {path} and write a summary in {output_path}.
    """

summarize_file("report.md", output_path="summary.md")
```

To simplify maintaining and sharing configuration between different AI Functions, we can use a `AIFunctionConfig` object:
```python
from ai_functions import ai_function, AIFunctionConfig
from pandas import DataFrame

class Configs:
    FAST_MODEL = AIFunctionConfig(model="global.anthropic.claude-haiku-4-5-20251001-v1:0")
    DATA_ANALYSIS = AIFunctionConfig(
        code_executor_additional_imports=["pandas.*", "numpy.*", "plotly.*"],
        code_execution_mode="local",
    )

# reuse a config
@ai_function(config=Configs.DATA_ANALYSIS)
def return_of_investment(data: DataFrame) -> DataFrame:
    """
    Analyze `data` and return a DataFrame with the return of investment for each year.
    """
    
# keyword arguments can be used to override config arguments for this specific function
@ai_function(config=Configs.FAST_MODEL, tools=[web_search])
def websearch(topic: str) -> str:
    """
    Research the following topic online and return a summary of your findings:
    {topic} 
    """
```


**Providing instructions.** The instructions/prompt of an AI Function can be provided in two ways. The simplest is to specify the prompt as a docstring as we have done until now:

```python
from ai_functions import ai_function

@ai_function
def translate(text: str, lang: str) -> str:
    """
    Translate the text below to the following language: `{lang}`.
    {text}
    """
```

The AI Function will interpret the docstring as template and attempt to replace the values using the provided arguments. This method however has limitations in some corner cases, for example if the docstring references a non-local variable. It also makes it difficult to construct prompts whose structure depends on the inputs.

Alternatively, we can construct the prompt inside the function and return it. In addition, the body of the function can also be used to perform input validation.

```python
from ai_functions import ai_function

# noinspection PyTypeChecker
@ai_function
def translate(text: str, lang: str) -> str:
    assert text, "`text` cannot be empty"
    assert lang, "`lang` cannot be empty"
    
    return t"""
    Translate the text below to the following language: `{lang}`.
    {text}
    """
```

The preferred way is to return a Template (t-string, available since Python >= 3.14) like in the example above. This allows the AI Function to apply custom formatting logic to preserve the correct indentation when replacing multi-line values in the template. On older Python versions, a standard string can be returned, but the user has to take care of ensuring the string will have correct indentation to avoid confusing the agent with improper formatting. 

Internally, the AI Function will always execute the function with the provided arguments. If the function returns a string or a `Template` (t-string), it will be used as the prompt to the agent. Else, it will try to fall back to interpreting the docstring as a template.

Since the type of the returned prompt is not generally compatible with the actual return type of the AI function, a type-checker directive must be added to avoid type-checking warnings.

## Post-conditions

A core notion of AI Functions is that programmers should not "prompt-and-pray" for the result returned by the agent to be correct. Rather, they should *verify* that the result satisfies the conditions required by their pipeline.

To this end, AI Functions expose *post-conditions* as a fundamental component in defining AI Functions. Post-conditions are functions (both standard Python functions or other AI Functions) that validate the input and provide feedback to the agent. This automatically instantiate a self-correcting feedback loop ensuring the correctness of the final return value of the function.

The following example extends the previous Meeting Summary adding user-defined post-conditions.

```python
from ai_functions import ai_function, PostConditionResult
from pydantic import BaseModel

class MeetingSummary(BaseModel):
    attendees: list[str]
    summary: str
    action_items: list[str]

# Post-conditions can be standard Python functions that raise an error if validation fails
def check_length(response: MeetingSummary):
    length = len(response.summary)
    assert length <= 50, f"Summary should be less than 50 words, but is {length} words long"

# Equivalently, the function can return a PostConditionResult object
def check_length(response: MeetingSummary) -> PostConditionResult:
    length = len(response.summary)
    if length > 50:
        return PostConditionResult(passed=False,message=f"Summary should be less than 50 words, but is {length} words long")
    return PostConditionResult(passed=True)

# A post-condition can also be an AI Function, since AI Functions *are* just functions
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

# Now we can add the functions above as post-conditions to validate the model output
@ai_function(post_conditions=[check_length, check_style])
def summarize_meeting(transcripts: str) -> MeetingSummary:
    """
    Write a summary of the following meeting in less than 50 words.
    <transcripts>
    {transcripts}
    </transcripts>
    """
``` 

All post-conditions are checked in parallel. The agent receives a message reporting all errors, and can address all of them at the same time thus cutting on the number of iterations necessary to converge to a correct output.

Post-conditions are not limited to checking the answer of the agent. They can more generally enforce invariants about the state of the system after the agent's execution. The example below shows how to implement a coding agent that verifies correctness of the implementation before moving on to new tasks.
```python
from ai_functions import ai_function
import pytest
from contextlib import redirect_stderr, redirect_stdout
import io
from typing import Literal, Any
from pydantic import BaseModel

class FeatureRequest(BaseModel):
    description: str
    test_files: list[str]

# A post-condition can request one of the original input arguments (e.g., `feature`)
# by adding it to the function signature. In this case, we ignore the actual response
# of the agent (`_answer`) and validate by running the feature's tests.
def run_tests(_answer: Any, feature: FeatureRequest):
    retcode = pytest.main()
    stdio_capture = io.StringIO()
    with redirect_stdout(stdio_capture), redirect_stderr(stdio_capture):
        retcode = pytest.main(feature.test_files)
    pytest_output = stdio_capture.getvalue()
    if retcode:
        raise RuntimeError(pytest_output)

@ai_function(post_conditions=[run_tests])
def implement_feature(feature: FeatureRequest) -> Literal["done"]:
    """
    Implement the following feature in the current code base:
    <feature>
    {feature.description}
    </feature>
    
    Once done the code base should pass the following tests: {feature.test_files}
    """

def implement_all_features(features: list[FeatureRequest]):
    for feature in features:
        implement_feature(feature)
```

Note that we are telling the agents what tests to pass both in the prompt and as a post-condition which may feel redundant. However, agents are generally much more effective in responding to validation messages than they are at following the prompts. Moreover, this provides a strong guarantee to the user that if the pipeline terminates all required tests are indeed passing without any need of manual inspection. 

## Async invocation and parallel workflows

AI Functions can be defined as either `sync` or `async`. The latter is particularly useful to define parallel workflows.

In the example below, we define a workflow to write a report on the current trends for a given stock. First, we conduct several searches in parallel. Then we use the result to write a report (see `examples/stock_report.py` for a more complex runnable example).

```python
from ai_functions import ai_function
from pandas import DataFrame
from datetime import timedelta
from typing import Literal
import asyncio

@ai_function(tools=[...])
async def research_news(stock: str) -> str:
    """
    Research and summarize the current news regarding the following stock: {stock} 
    """

@ai_function(tools=[...])
async def research_price(stock: str, past_days: int) -> DataFrame:
    """
    Use the `yfinance` Python package to retrieve the historical prices of {stock} in the last 30 days.
    Return a dataframe with columns [date, price (float, price at market close)]  
    """

@ai_function
def write_report(stock: str, news: str, prices: DataFrame) -> str:
    """
    Write and return a HTML report on the trend of the stock {stock} in the last 30 days.
    Use the provided `prices` DataFrame and the following summary of recent news:
    {news}
    """

async def stock_research_workflow(stock: str):
    # Run the two agents in parallel
    news, prices = await asyncio.gather(research_news(stock), research_price(stock))
    # Use their results to write a report
    write_report(stock, news, prices)
```
