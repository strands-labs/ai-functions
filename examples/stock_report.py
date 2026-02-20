import os
from pathlib import Path

from pydantic import ConfigDict

from ai_functions import ai_function
import asyncio, datetime, pandas as pd
from typing import Literal
from dataclasses import dataclass
import pypandoc

from ai_functions.types import CodeExecutionMode, AIFunctionConfig
import utils

websearch_tool = utils.get_websearch_tool()

# define some configs that we can share across different ai_functions
class Configs:
    FAST_MODEL = AIFunctionConfig(model="global.anthropic.claude-haiku-4-5-20251001-v1:0")
    DATA_ANALYSIS = AIFunctionConfig(
        code_executor_additional_imports=["pandas.*", "numpy.*", "yfinance.*", "plotly.*"],
        code_execution_mode="local"
    )

# collect stock research results
@dataclass
class StockInfo:
    symbol: str
    news: str
    prices: pd.DataFrame


# ai_functions work with any model and tool compatible with Strands
@ai_function(config=Configs.FAST_MODEL, tools=[websearch_tool])
async def research_news(stock: str) -> str:
    """
    Research and summarize the current news regarding the following stock symbol: {stock}
    """

def check_nan(df: pd.DataFrame):
    assert not df.isnull().any().any(), "Returned DataFrame contains NaN values"

# By default, agents have access to a python execution environment, allowing it to use libraries and return rich data-types
@ai_function(config=Configs.DATA_ANALYSIS, post_conditions=[check_nan])
async def research_price(stock: str) -> pd.DataFrame:
    """
    Use the `yfinance` Python package to retrieve the historical prices of {stock} in the last 30 days.
    Return a dataframe with columns: ["date", "price" (float, price at market close)]
    """


# Function inputs are also available inside the Python environment for further processing
@ai_function(config=Configs.DATA_ANALYSIS)
def write_report(stock_info: list[StockInfo]) -> str:
    """
    Write a html report comparing the trend of the following stocks: {",".join(s.symbol for s in stock_info)}.
    Use the information in `stock_info` for your analysis. Use `plotly` to embed plots illustrating the trend.
    Return the full html content.
    """


# Asynchronous agent graphs can be constructed combining standard Python function calls
async def research_stock(stock: str) -> StockInfo:
    # Run the two functions in parallel
    news, prices = await asyncio.gather(research_news(stock), research_price(stock))
    return StockInfo(stock, news, prices)


async def write_stock_report(stocks: list[str]):
    # gather information about all stocks in parallel:
    stock_info = await asyncio.gather(*(research_stock(stock) for stock in stocks))
    # Use their results to write a report
    return write_report(stock_info)


async def main():
    print("Generating report...")
    html_content = await write_stock_report(['AAPL', 'JNJ', 'JPM', 'XOM'])
    output_path = Path('stock_report.html')
    output_path.write_text(html_content)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
