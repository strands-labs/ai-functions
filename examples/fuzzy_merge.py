# """
# Fuzzy merge example - normalize product names by removing version suffixes.

# Note: Use _create_data.py to create test data before running this example.
# """
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

# Merge revisions of the same product 
df = fuzzy_merge_products(df)