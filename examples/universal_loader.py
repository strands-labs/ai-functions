"""
Example showing how to use AI Functions with Python integration to solve tasks which would be infeasible
with standard programming paradigms. The AI Function takes as input a file in an unknown format and
dynamically writes and execute the code to convert it to DataFrame.

The structure of the DataFrame is validated using a post-condition to ensure it will be compatible with
the remaining parts of the workflow.
"""
from pandas import DataFrame, api

from ai_functions import ai_function
from ai_functions.types import PostConditionResult, CodeExecutionMode

def check_invoice_dataframe(df: DataFrame):
    """Post-condition: validate DataFrame structure."""
    assert {'product_name', 'quantity', 'price', 'purchase_date'}.issubset(df.columns)
    assert api.types.is_integer_dtype(df['quantity']), "quantity must be an integer"
    assert api.types.is_float_dtype(df['price']), "price must be a float"
    assert api.types.is_datetime64_any_dtype(df['purchase_date']), "purchase_date must be a datetime64"
    assert not df.duplicated(subset=['product_name', 'price', 'purchase_date']).any(), "The combination of product_name, price, and purchase_date must be unique"


@ai_function(
    post_conditions=[check_invoice_dataframe],
    code_execution_mode="local",
    code_executor_additional_imports=["pandas", "sqlite3"],
    code_executor_kwargs={"timeout_seconds": 10}
)
def import_invoice(path: str) -> DataFrame:
    """
    The file `{path}` contains purchase logs. Extract them in a DataFrame with columns:
    - product_name (str)
    - quantity (int)
    - price (float)
    - purchase_date (datetime)
    """


if __name__ == '__main__':
    import os
    import sys
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _create_data import create_data

    # Save data in a few different formats (csv, json, sqlite3) and try to load them back
    with tempfile.TemporaryDirectory(delete=True) as temp_dir:
        filenames = create_data(temp_dir)
        results = []
        for filename in filenames:
            print(f"===== Reading data from {filename.name} =====")
            df = import_invoice(filename)
            results.append(df)
        for filename, df in zip(filenames, results):
            print(f"\n===== Parsed data from {filename.name} =====")
            print(df)
