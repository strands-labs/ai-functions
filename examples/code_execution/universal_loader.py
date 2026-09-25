"""Load a file of unknown format into a DataFrame via a code-executing AI function.

Given a path, the AI function writes and runs the parsing code itself, so one
function handles CSV, JSON, and SQLite alike. A post-condition validates the
resulting DataFrame's structure before it flows into the rest of the workflow.
"""

from example_helpers import models
from example_helpers.utils import display, rule
from pandas import DataFrame, api

from ai_functions import ai_function


def check_invoice_dataframe(df: DataFrame):
    """Post-condition: validate DataFrame structure."""
    assert {"product_name", "quantity", "price", "purchase_date"}.issubset(df.columns)
    assert api.types.is_integer_dtype(df["quantity"]), "quantity must be an integer"
    assert api.types.is_float_dtype(df["price"]), "price must be a float"
    assert api.types.is_datetime64_any_dtype(df["purchase_date"]), "purchase_date must be a datetime64"
    assert not df.duplicated(subset=["product_name", "price", "purchase_date"]).any(), (
        "The combination of product_name, price, and purchase_date must be unique"
    )


@ai_function(
    model=models.medium,
    post_conditions=[check_invoice_dataframe],
    code_execution_mode="local",
    code_executor_additional_imports=["pandas", "sqlite3"],
    code_executor_kwargs={"timeout_seconds": 10},
)
def import_invoice(path: str) -> DataFrame:
    """
    The file `{path}` contains purchase logs. Extract them in a DataFrame with columns:
    - product_name (str)
    - quantity (int)
    - price (float)
    - purchase_date (datetime)
    """


if __name__ == "__main__":
    import tempfile

    from example_helpers.invoice_data import create_data

    # Save the same data as csv, json, and sqlite3, then load each back.
    with tempfile.TemporaryDirectory(delete=True) as temp_dir:
        filenames = create_data(temp_dir)
        results = []
        for filename in filenames:
            rule(f"Reading data from {filename.name}")
            df = import_invoice.run_sync(path=filename)
            results.append(df)
        for filename, df in zip(filenames, results, strict=False):
            display(f"Parsed data from {filename.name}", df.to_string(), lang="text")
