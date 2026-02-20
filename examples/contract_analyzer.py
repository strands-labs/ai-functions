"""Contract Analyzer — Legal Contract Clause Extraction.

Extracts specific clause types from a contract and flags risks using an AI
function. Uses nested Pydantic models for structured output and chains 3 post-conditions 
that validate clause coverage, verify excerpts exist verbatim in the source 
text (anti-hallucination), and check risk flag referential integrity. 
Post-condition parameter injection lets validators
access the original contract text and requested clause types.
"""

import re
import textwrap

from pydantic import BaseModel, Field

from ai_functions import ai_function

# ── Pydantic Models ─────────────────────────────────────────────────────────
# Nested Pydantic models: ExtractedClause and RiskFlag are composed inside ContractAnalysis.
# Field descriptions guide the LLM on what to produce for each field.


class ExtractedClause(BaseModel):
    """A single clause extracted from a contract."""

    clause_type: str = Field(description="Type of clause, e.g., 'termination', 'liability'")
    excerpt: str = Field(description="Verbatim text excerpt from the contract")
    section_ref: str = Field(description="Section reference, e.g., 'Section 4.2'")
    interpretation: str = Field(description="Plain-English interpretation of the clause")


class RiskFlag(BaseModel):
    """A risk identified in an extracted clause."""

    clause_index: int = Field(description="Index into the clauses list (0-based)")
    risk_type: str = Field(description="Type of risk, e.g., 'one_sided', 'missing_cap', 'ambiguous'")
    severity: str = Field(description="Risk severity: 'low', 'medium', or 'high'")
    explanation: str = Field(description="Why this clause poses a risk")


class ContractAnalysis(BaseModel):
    """Complete contract analysis with extracted clauses and risk assessment."""

    clauses: list[ExtractedClause]
    risk_flags: list[RiskFlag]
    overall_risk: str = Field(description="Overall risk level: 'low', 'medium', or 'high'")
    summary: str = Field(description="Brief summary of the contract analysis")


VALID_SEVERITIES = {"low", "medium", "high"}


# ── Post-condition 1: Clause coverage ───────────────────────────────────────
# Three chained post-conditions run in order after each LLM attempt.
# If any fail, the error message is fed back to the LLM for retry (up to max_attempts).
# `clause_types` is injected from analyze_contract's parameter of the same name.

def validate_clause_coverage(result: ContractAnalysis, clause_types: list[str]) -> None:
    """Validate that every requested clause type has at least one extracted clause."""
    extracted_types = {c.clause_type.lower() for c in result.clauses}
    requested_types = {t.lower() for t in clause_types}

    missing = requested_types - extracted_types
    if missing:
        raise ValueError(
            f"Missing clause types: {sorted(missing)}. "
            f"Extracted: {sorted(extracted_types)}"
        )


# ── Post-condition 2: Excerpt verification (anti-hallucination) ─────────────
# Prevents the LLM from fabricating quotes that don't exist in the contract.
# `contract_text` is injected from analyze_contract's parameter of the same name.
def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces for fuzzy matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_excerpts(result: ContractAnalysis, contract_text: str) -> None:
    """Validate that each excerpt exists verbatim in the original contract text."""
    normalized_source = _normalize_whitespace(contract_text)
    hallucinated: list[str] = []

    for i, clause in enumerate(result.clauses):
        normalized_excerpt = _normalize_whitespace(clause.excerpt)
        if not normalized_excerpt:
            hallucinated.append(f"Clause {i} ({clause.clause_type}): empty excerpt")
        elif normalized_excerpt not in normalized_source:
            # Show first 80 chars of the excerpt in the error
            preview = clause.excerpt[:80] + ("..." if len(clause.excerpt) > 80 else "")
            hallucinated.append(
                f"Clause {i} ({clause.clause_type}): excerpt not found in source: "
                f"'{preview}'"
            )

    if hallucinated:
        raise ValueError("Hallucinated excerpts detected:\n" + "\n".join(hallucinated))


# ── Post-condition 3: Risk flag consistency ─────────────────────────────────
# Only needs `result` (no parameter injection) — checks referential integrity
# between risk_flags and the clauses list the LLM produced.
def validate_risk_flags(result: ContractAnalysis) -> None:
    """Validate risk flag indices and severity values."""
    errors: list[str] = []

    # Check overall_risk is valid
    if result.overall_risk.lower() not in VALID_SEVERITIES:
        errors.append(
            f"Invalid overall_risk: '{result.overall_risk}'. "
            f"Must be one of: {sorted(VALID_SEVERITIES)}"
        )

    for i, flag in enumerate(result.risk_flags):
        # Check clause_index is valid
        if flag.clause_index < 0 or flag.clause_index >= len(result.clauses):
            errors.append(
                f"Risk flag {i}: clause_index {flag.clause_index} out of range "
                f"[0, {len(result.clauses) - 1}]"
            )

        # Check severity is valid
        if flag.severity.lower() not in VALID_SEVERITIES:
            errors.append(
                f"Risk flag {i}: invalid severity '{flag.severity}'. "
                f"Must be one of: {sorted(VALID_SEVERITIES)}"
            )

    if errors:
        raise ValueError("\n".join(errors))


# ── AI Function ─────────────────────────────────────────────────────────────
# LLM fills the ContractAnalysis schema directly, no code execution.
# The docstring is the prompt template — {contract_text} and {clause_types} are
# substituted with actual argument values. The function body is empty by design.

@ai_function(
    post_conditions=[validate_clause_coverage, validate_excerpts, validate_risk_flags],
    max_attempts=3,
)
def analyze_contract(contract_text: str, clause_types: list[str]) -> ContractAnalysis:
    """
    Analyze the following contract and extract the requested clause types.

    Clause types to extract: {clause_types}

    Contract text:
    {contract_text}

    For each clause:
    - Set clause_type to match one of the requested types exactly
    - Set excerpt to a VERBATIM copy-paste from the contract (do not paraphrase or modify)
    - Set section_ref to the section number where the clause appears
    - Provide a plain-English interpretation

    For risk flags:
    - Identify clauses that are one-sided, ambiguous, missing caps/limits, or unusual
    - Set clause_index to the 0-based index of the clause in your clauses list
    - Set severity to "low", "medium", or "high"

    Set overall_risk to "low", "medium", or "high" based on the aggregate risk.
    """


# ── Sample Data ──────────────────────────────────────────────────────────────

SAMPLE_CONTRACT = textwrap.dedent("""\
    SAAS SUBSCRIPTION AGREEMENT

    This SaaS Subscription Agreement ("Agreement") is entered into as of January 1, 2025,
    by and between AnyCompany ("Provider") and Example Corp ("Customer").

    SECTION 1: SERVICE DESCRIPTION
    Provider shall provide Customer with access to the AnyCompany Analytics Platform
    ("Service"), including data processing, dashboard visualization, and automated
    reporting capabilities, subject to the terms of this Agreement.

    SECTION 2: TERM AND TERMINATION
    2.1 This Agreement shall commence on the Effective Date and continue for an initial
    term of thirty-six (36) months ("Initial Term").
    2.2 Either party may terminate this Agreement for cause upon sixty (60) days written
    notice if the other party materially breaches this Agreement and fails to cure such
    breach within the notice period.
    2.3 Provider may terminate this Agreement immediately and without notice if Customer
    fails to pay any amounts due within fifteen (15) days of the payment due date.
    2.4 Upon termination, Customer shall have thirty (30) days to export its data, after
    which Provider may delete all Customer data without liability.

    SECTION 3: FEES AND PAYMENT
    3.1 Customer shall pay Provider an annual subscription fee of $120,000, payable in
    quarterly installments of $30,000, due on the first business day of each quarter.
    3.2 Provider reserves the right to increase fees by up to twenty percent (20%) upon
    each annual renewal, with thirty (30) days prior written notice.
    3.3 All fees are non-refundable, including in the event of early termination by Customer.

    SECTION 4: LIABILITY AND INDEMNIFICATION
    4.1 IN NO EVENT SHALL PROVIDER'S TOTAL AGGREGATE LIABILITY EXCEED THE FEES PAID BY
    CUSTOMER IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM. PROVIDER SHALL NOT BE LIABLE
    FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES.
    4.2 Customer shall indemnify, defend, and hold harmless Provider and its officers,
    directors, employees, and agents from and against any and all claims, damages, losses,
    liabilities, costs, and expenses (including reasonable attorneys' fees) arising out of
    or relating to Customer's use of the Service, Customer's breach of this Agreement, or
    Customer's violation of any applicable law or regulation.
    4.3 Provider shall have no obligation to indemnify Customer under any circumstances.

    SECTION 5: DATA PROTECTION
    5.1 Provider shall implement commercially reasonable security measures to protect
    Customer data, but makes no guarantees regarding the absolute security of data.
    5.2 In the event of a data breach, Provider shall notify Customer within seventy-two
    (72) hours of becoming aware of such breach.
    5.3 Customer acknowledges that Provider may use anonymized and aggregated Customer
    data for purposes of improving the Service, benchmarking, and marketing.
    5.4 Provider may transfer Customer data to third-party subprocessors without prior
    notice to or consent from Customer.

    SECTION 6: GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with the laws of the
    State of Delaware, without regard to its conflict of laws principles.
""")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clause_types = ["termination", "liability", "indemnification", "data protection"]
    analysis = analyze_contract(SAMPLE_CONTRACT, clause_types)
    print(analysis.model_dump_json(indent=2))
