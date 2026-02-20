"""Receipt Parser Tool — AI Function as Strands Tool.

An agent that processes receipts and invoices. It reads raw receipt text,
uses an AI function directly as a tool to extract structured expense data
with math validation, and saves each expense to an in-memory tracker.
The agent orchestrates three tools: read, parse, and save.
"""

import json
import textwrap

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models import BedrockModel

from ai_functions import ai_function

# ── Pydantic Models ─────────────────────────────────────────────────────────
# These models define the structured output schema.
# The LLM's response is constrained to produce JSON matching this exact schema.
# The Field descriptions are included in the schema sent to the LLM, so they
# serve as guidance for what each field should contain.


class LineItem(BaseModel):
    """A single line item from a receipt/invoice."""

    description: str = Field(description="Item or service description")
    quantity: int = Field(description="Number of units")
    unit_price: float = Field(description="Price per unit")
    amount: float = Field(description="Total for this line item (quantity * unit_price)")


class ReceiptData(BaseModel):
    """Structured data extracted from a receipt/invoice."""

    vendor: str = Field(description="Vendor or company name")
    invoice_number: str = Field(description="Invoice or receipt number")
    date: str = Field(description="Invoice date (YYYY-MM-DD format)")
    items: list[LineItem] = Field(description="List of line items")
    subtotal: float = Field(description="Sum of all line item amounts before tax")
    tax: float = Field(description="Tax amount")
    total: float = Field(description="Final total (subtotal + tax)")


# ── Post-conditions ──────────────────────────────────────────────────────────
# Post-conditions are regular Python functions that run AFTER the LLM produces
# its structured output. They validate the result and, if any fail, the error
# message is fed back to the LLM which retries (up to max_attempts times).

def validate_math(result: ReceiptData) -> None:
    """Validate that all math is internally consistent."""
    errors: list[str] = []

    # ±0.01 tolerance throughout for floating-point rounding differences
    for i, item in enumerate(result.items):
        expected = item.quantity * item.unit_price
        if abs(item.amount - expected) > 0.01:
            errors.append(
                f"Line item {i} ({item.description}): amount {item.amount} != "
                f"quantity {item.quantity} * unit_price {item.unit_price} = {expected}"
            )

    # Check subtotal ≈ sum of line item amounts
    items_sum = sum(item.amount for item in result.items)
    if abs(result.subtotal - items_sum) > 0.01:
        errors.append(
            f"Subtotal {result.subtotal} != sum of line items {items_sum}"
        )

    # Check total ≈ subtotal + tax
    expected_total = result.subtotal + result.tax
    if abs(result.total - expected_total) > 0.01:
        errors.append(
            f"Total {result.total} != subtotal {result.subtotal} + tax {result.tax} = {expected_total}"
        )

    if errors:
        raise ValueError("\n".join(errors))


def validate_completeness(result: ReceiptData) -> None:
    """Validate that all required fields are populated."""
    errors: list[str] = []

    if not result.vendor.strip():
        errors.append("Vendor name is empty")
    if not result.invoice_number.strip():
        errors.append("Invoice number is empty")
    if not result.date.strip():
        errors.append("Date is empty")
    if not result.items:
        errors.append("No line items found")

    if errors:
        raise ValueError("\n".join(errors))


# ── AI Function (used directly as a tool) ──────────────────────────────────
# The `description` argument is used as the tool description that the
# orchestrating agent sees when deciding which tool to call.

@ai_function(
    description="Parse a receipt or invoice text and extract structured expense data",
    post_conditions=[validate_math, validate_completeness],
    max_attempts=3,
)
def parse_receipt(receipt_text: str) -> ReceiptData:
    """
    Extract structured data from this receipt/invoice.

    Receipt text:
    {receipt_text}

    Instructions:
    - Extract all line items with their quantity, unit price, and total amount
    - Calculate subtotal as the sum of all line item amounts
    - Extract tax amount (if no tax is listed, use 0.0)
    - Calculate total as subtotal + tax
    - Use YYYY-MM-DD format for the date
    - Ensure all math is consistent: each line item amount = quantity * unit_price,
      subtotal = sum of line item amounts, total = subtotal + tax
    """


# ── Strands Agent's Tools ────────────────────────────────────────────────────

# Simulates a database or file store — the agent's save_expense tool writes here
_saved_expenses: list[dict] = []


@tool(description="Save parsed expense data for later reporting")
def save_expense(receipt_json: str) -> str:
    """Save a parsed receipt to the expense tracker.

    Args:
        receipt_json: JSON string of parsed receipt data

    Returns:
        Confirmation message with running total
    """
    data = json.loads(receipt_json)
    _saved_expenses.append(data)
    running_total = sum(e["total"] for e in _saved_expenses)
    return (
        f"Saved expense from {data['vendor']} for ${data['total']:.2f}. "
        f"Running total: ${running_total:.2f} across {len(_saved_expenses)} receipts."
    )


@tool(description="Read a receipt by its ID from the available receipts")
def read_receipt(receipt_id: str) -> str:
    """Read the text content of a receipt by its ID.

    Args:
        receipt_id: The receipt identifier (e.g., 'receipt_1', 'receipt_2', 'receipt_3')

    Returns:
        The raw text content of the receipt, or an error message if not found
    """
    if receipt_id in SAMPLE_RECEIPTS:
        return SAMPLE_RECEIPTS[receipt_id]
    available = ", ".join(SAMPLE_RECEIPTS.keys())
    return f"Receipt '{receipt_id}' not found. Available: {available}"


# ── Sample Data ──────────────────────────────────────────────────────────────
# Three receipts in intentionally different formats (tabular, inline, dotted)
# to test the LLM's ability to parse varied real-world receipt styles.

SAMPLE_RECEIPTS = {
    "receipt_1": textwrap.dedent("""\
        INVOICE #ACT-2025-0847
        AnyCompany Technologies
        Date: March 15, 2025

        Description                  Qty    Unit Price    Amount
        ─────────────────────────────────────────────────────────
        Cloud Hosting (Annual)         1      $8,400.00   $8,400.00
        SSL Certificate (Wildcard)     3        $120.00     $360.00
        Domain Registration            5         $14.99      $74.95

        Subtotal:                                        $8,834.95
        Tax (7.25%):                                       $640.53
        Total:                                           $9,475.48

        Payment Terms: Net 30
        """),

    "receipt_2": textwrap.dedent("""\
        Example Corp
        inv no: EXC-0392
        2025-06-22

        2x  Ergonomic Standing Desk   @ $549.00 ea   $1,098.00
        4x  Monitor Arm Mount         @ $89.95 ea      $359.80
        1x  Cable Management Kit      @ $34.50 ea       $34.50

        subtotal  $1,492.30
        sales tax   $119.38
        TOTAL DUE  $1,611.68
        """),

    "receipt_3": textwrap.dedent("""\
        *** AnyOrganization Cybersecurity ***
        Receipt #AOC-7721
        Date of purchase: Jan 8 2025

        Managed Firewall Service (12 mo) ........ 1 x $3,600.00 = $3,600.00
        Endpoint Protection Licenses ............ 50 x $12.00 = $600.00
        Security Audit (one-time) ............... 1 x $2,500.00 = $2,500.00
        Incident Response Retainer .............. 1 x $1,800.00 = $1,800.00

        Sub-total: $8,500.00
        Tax: $0.00
        Amount due: $8,500.00

        Thank you for your business!
        """),
}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _saved_expenses.clear()
    agent = Agent(
        model=BedrockModel(model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        tools=[parse_receipt, read_receipt, save_expense],
    )
    result = agent(
        "Read all available receipts (receipt_1, receipt_2, receipt_3), "
        "parse each one to extract the expense data, save each expense, "
        "and then give me a summary of all expenses with the grand total."
    )
    print(result)
