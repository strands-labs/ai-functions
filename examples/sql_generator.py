"""SQL Generator — Natural Language to SQL with sqlglot validation.

Converts natural language questions to SQL using an AI function with code
execution (LOCAL mode, the default). The LLM gets a python_executor tool
with sqlglot available, so it can build/validate queries programmatically.
Three post-conditions validate progressively deeper: syntax, table
references, then column references — all using sqlglot's AST parser.
"""

import sqlglot
from sqlglot import exp

from ai_functions import ai_function
from ai_functions.types import AIFunctionConfig

# ── Helpers ──────────────────────────────────────────────────────────────────
# Parses CREATE TABLE DDL into a lookup table used by the post-condition validators.

def _build_schema_catalog(schema_ddl: str) -> dict[str, set[str]]:
    """Parse CREATE TABLE DDL to build a table -> columns mapping.

    Returns a dict like: {"customers": {"id", "name", "email", ...}, ...}
    """
    catalog: dict[str, set[str]] = {}
    for statement in sqlglot.parse(schema_ddl):
        if not isinstance(statement, exp.Create):
            continue
        table = statement.find(exp.Table)
        if not table:
            continue
        table_name = table.name.lower()
        columns = {col.name.lower() for col in statement.find_all(exp.ColumnDef)}
        catalog[table_name] = columns
    return catalog


# ── Post-condition 1: Syntax ────────────────────────────────────────────────
# Three post-conditions validate progressively deeper: syntax → tables → columns.
# `dialect` and `schema` are injected from generate_sql's parameters of the same name.


def validate_syntax(result: str, dialect: str) -> None:
    """Parse the SQL with sqlglot to catch syntax errors."""
    if not result or not result.strip():
        raise ValueError("Empty SQL generated")

    parsed = sqlglot.parse(result, dialect=dialect)

    if not parsed or not parsed[0]:
        raise ValueError("sqlglot returned empty parse result")

    stmt_type = type(parsed[0]).__name__
    valid_types = {"Select", "Insert", "Update", "Delete", "Union"}
    if stmt_type not in valid_types:
        raise ValueError(f"Expected a query statement, got: {stmt_type}")


# ── Post-condition 2: Table references ──────────────────────────────────────

def validate_table_references(result: str, schema: str, dialect: str) -> None:
    """Validate that all table references exist in the schema (AST-based).

    Walks the AST to find FROM/JOIN table references, excludes CTE names and
    subquery aliases (which are query-defined, not schema tables).
    """
    catalog = _build_schema_catalog(schema)
    valid_tables = set(catalog.keys())

    parsed = sqlglot.parse(result, dialect=dialect)
    stmt = parsed[0]

    # Collect CTE names — these are defined in the query, not in the schema
    cte_names = {cte.alias.lower() for cte in stmt.find_all(exp.CTE)}

    # Collect subquery aliases — also not real tables
    subquery_aliases = set()
    for subquery in stmt.find_all(exp.Subquery):
        if subquery.alias:
            subquery_aliases.add(subquery.alias.lower())

    # All table references from FROM / JOIN
    table_refs = {t.name.lower() for t in stmt.find_all(exp.Table)}

    # Remove CTEs and subquery aliases — they're query-defined, not schema tables
    real_table_refs = table_refs - cte_names - subquery_aliases
    unknown = real_table_refs - valid_tables

    if unknown:
        raise ValueError(f"Unknown tables: {sorted(unknown)}. Valid tables: {sorted(valid_tables)}")


# ── Post-condition 3: Column references ─────────────────────────────────────

def validate_column_references(result: str, schema: str, dialect: str) -> None:
    """Validate column references against the schema, resolving table aliases.

    Resolves table aliases (e.g., `o` → `orders`) before checking that each
    column exists on its table. Skips SELECT aliases used in ORDER BY / HAVING.
    """
    catalog = _build_schema_catalog(schema)
    all_columns = {col for cols in catalog.values() for col in cols}

    parsed = sqlglot.parse(result, dialect=dialect)
    stmt = parsed[0]

    # Skip column validation if SELECT * is used (all columns implicitly valid)
    if stmt.find(exp.Star):
        return

    # Collect SELECT aliases (e.g., SUM(qty) AS total_sold) — valid in ORDER BY / HAVING
    select_aliases = {alias.alias.lower() for alias in stmt.find_all(exp.Alias)}

    # Build alias -> real table mapping from FROM / JOIN clauses
    alias_map: dict[str, str] = {}
    for table in stmt.find_all(exp.Table):
        name = table.name.lower()
        if table.alias:
            alias_map[table.alias.lower()] = name
        else:
            alias_map[name] = name

    invalid: list[str] = []
    for col in stmt.find_all(exp.Column):
        col_name = col.name.lower()
        table_ref = col.table.lower() if col.table else ""

        if table_ref:
            # Table-qualified: resolve alias, then check column exists on that table
            real_table = alias_map.get(table_ref, table_ref)
            if real_table in catalog and col_name not in catalog[real_table]:
                valid_cols = sorted(catalog[real_table])
                invalid.append(
                    f"Column '{col_name}' not in table '{real_table}'. "
                    f"Valid columns: {valid_cols}"
                )
        else:
            # Unqualified: skip if it's a SELECT alias (e.g., ORDER BY total_sold)
            if col_name in select_aliases:
                continue
            # Otherwise check column exists in at least one table
            if col_name not in all_columns:
                invalid.append(f"Column '{col_name}' not found in any table")

    if invalid:
        raise ValueError("\n".join(invalid))


# ── AI Function ─────────────────────────────────────────────────────────────
# LOCAL mode (default): LLM gets a python_executor tool with sqlglot available,
# so it can build/validate queries programmatically before returning the result.
# code_executor_kwargs sets a 5s timeout to prevent runaway code.

@ai_function(
    config=AIFunctionConfig(
        code_executor_additional_imports=["sqlglot"],
        code_executor_kwargs={"timeout_seconds": 5},
        post_conditions=[validate_syntax, validate_table_references, validate_column_references],
        max_attempts=3,
    )
)
def generate_sql(question: str, schema: str, dialect: str = "postgres") -> str:
    """
    Given the database schema below, write a SQL query that answers the question.

    Database dialect: {dialect}

    Database schema:
    {schema}

    Question: {question}

    Use the sqlglot library to build or validate the query before returning it.
    Return well-formatted SQL.
    """


# ── Sample Data ──────────────────────────────────────────────────────────────

ECOMMERCE_SCHEMA = """\
CREATE TABLE customers (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(200),
    city VARCHAR(100),
    created_at DATE
);

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(200),
    category VARCHAR(100),
    price DECIMAL(10,2),
    stock INT
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    order_date DATE,
    status VARCHAR(50)
);

CREATE TABLE order_items (
    id INT PRIMARY KEY,
    order_id INT REFERENCES orders(id),
    product_id INT REFERENCES products(id),
    quantity INT,
    unit_price DECIMAL(10,2)
);
"""

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "List all customers from New York who placed an order in the last 30 days",
        "What are the top 5 best-selling products by total quantity sold?",
        "Show the total revenue per product category for completed orders",
    ]
    for question in questions:
        sql = generate_sql(question, ECOMMERCE_SCHEMA)
        print(f"Q: {question}\n{sql}\n")
