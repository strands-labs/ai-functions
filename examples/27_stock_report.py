"""Stock report — parallel async agent graph returning rich data types.

Shows:
- two ``@ai_function`` s run in parallel per stock via ``asyncio.gather``, their
  results composed into a plain dataclass, then a third ``@ai_function`` writes
  an HTML report — standard Python control flow around the agent calls
- ``code_execution_mode="local"`` lets an agent use libraries (``yfinance``,
  ``plotly``) and return rich types like a ``pd.DataFrame``

Requires a websearch API key (TAVILY_API_KEY or EXA_API_KEY) in the environment.
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_functions import ai_function
from ai_functions.ai_thread.config import CodeExecutionMode, ThreadConfig


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


# Configs shared across the ai_functions below.
class Configs:
    FAST_MODEL = ThreadConfig(model="global.anthropic.claude-haiku-4-5-20251001-v1:0")
    DATA_ANALYSIS = ThreadConfig(
        code_execution_mode=CodeExecutionMode.LOCAL,
        code_executor_additional_imports=["pandas.*", "numpy.*", "yfinance.*", "plotly.*"],
    )


# Collected stock research results.
@dataclass
class StockInfo:
    symbol: str
    news: str
    prices: pd.DataFrame


@ai_function(str, config=Configs.FAST_MODEL, tools=[websearch_tool])
async def research_news(stock: str):
    """
    Research and summarize the current news regarding the following stock symbol: {stock}
    """


def check_nan(df: pd.DataFrame):
    assert not df.isnull().any().any(), "Returned DataFrame contains NaN values"


# The sandboxed Python environment lets the agent use libraries and return rich data types.
@ai_function(pd.DataFrame, config=Configs.DATA_ANALYSIS, post_conditions=[check_nan])
async def research_price(stock: str):
    """
    Use the `yfinance` Python package to retrieve the historical prices of {stock} in the last 30 days.
    Return a dataframe with columns: ["date", "price" (float, price at market close)]
    """


# Function inputs are available inside the Python environment for further processing.
@ai_function(str, config=Configs.DATA_ANALYSIS)
def write_report(stock_info: list[StockInfo]):
    """
    Write a html report comparing the trend of the following stocks: {",".join(s.symbol for s in stock_info)}.
    Use the information in `stock_info` for your analysis. Use `plotly` to embed plots illustrating the trend.
    Return the full html content.
    """


async def research_stock(stock: str) -> StockInfo:
    # Run the two research functions in parallel.
    news, prices = await asyncio.gather(research_news(stock), research_price(stock))
    return StockInfo(stock, news, prices)


async def write_stock_report(stocks: list[str]) -> str:
    # Gather information about all stocks in parallel, then write a report from the results.
    stock_info = await asyncio.gather(*(research_stock(stock) for stock in stocks))
    return await write_report(list(stock_info))


async def main():
    print("Generating report...")
    html_content = await write_stock_report(["AAPL", "JNJ", "JPM", "XOM"])
    output_path = Path(__file__).parent / "stock_report.html"
    output_path.write_text(html_content)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
