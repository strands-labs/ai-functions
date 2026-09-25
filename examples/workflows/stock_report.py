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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from example_helpers import models
from example_helpers.utils import display, get_websearch_tool, rule

from ai_functions import ai_function
from ai_functions.ai_thread.config import CodeExecutionMode, ThreadConfig

websearch_tool = get_websearch_tool()


# Configs shared across the ai_functions below.
class Configs:
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


@ai_function(model=models.small, tools=[websearch_tool])
async def research_news(stock: str) -> str:
    """
    Research and summarize the current news regarding the following stock symbol: {stock}
    """


def check_nan(df: pd.DataFrame):
    assert not df.isnull().any().any(), "Returned DataFrame contains NaN values"


# The sandboxed Python environment lets the agent use libraries and return rich data types.
@ai_function(model=models.medium, config=Configs.DATA_ANALYSIS, post_conditions=[check_nan])
async def research_price(stock: str) -> pd.DataFrame:
    """
    Use the `yfinance` Python package to retrieve the historical prices of {stock} in the last 30 days.
    Return a dataframe with columns: ["date", "price" (float, price at market close)]
    """


# Function inputs are available inside the Python environment for further processing.
@ai_function(model=models.medium, config=Configs.DATA_ANALYSIS)
def write_report(stock_info: list[StockInfo]) -> str:
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
    rule("Generating report")
    html_content = await write_stock_report(["AAPL", "JNJ", "JPM", "XOM"])
    output_path = Path(__file__).parent / "stock_report.html"
    output_path.write_text(html_content)
    display("Report saved", str(output_path), lang="text")


if __name__ == "__main__":
    asyncio.run(main())
