"""Data Normalizer — Survey/Freetext Response Classification.

Classifies messy freetext survey responses into predefined categories
using an AI function. Uses a cheaper model (Nova Lite 2) since classification
doesn't need a large model, and validates that every response gets mapped
to a valid category.
"""

from pydantic import BaseModel
from strands.models import BedrockModel

from ai_functions import ai_function

# ── Model Configuration ─────────────────────────────────────────────────────
# Nova Lite 2 is cheap and fast — classification doesn't need a large model.
# temperature=0 for deterministic output (same input always maps to same category).

model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
    temperature=0,
)


# ── Pydantic Models ─────────────────────────────────────────────────────────
# With DISABLED mode, the LLM's output is constrained to match this schema exactly.


class NormalizedResponses(BaseModel):
    """Mapping of each survey response to its assigned category."""

    mappings: dict[str, str]


# ── Post-conditions ─────────────────────────────────────────────────────────
# Post-condition with parameter injection: `responses` and `categories` are
# auto-matched by name to the ai_function's parameters and injected at validation time.
# This lets the validator check the LLM's output against the original inputs.

def validate_mapping(
    result: NormalizedResponses,
    responses: list[str],
    categories: list[str],
) -> None:
    """Validate that every response is mapped to a valid category."""
    # Check all responses are present as keys
    missing = [r for r in responses if r not in result.mappings]
    if missing:
        raise ValueError(f"Missing mappings for {len(missing)} responses: {missing[:3]}...")

    # Check all values are valid categories
    invalid = {v for v in result.mappings.values() if v not in categories}
    if invalid:
        raise ValueError(f"Invalid categories used: {invalid}. Valid: {categories}")


# ── AI Function ─────────────────────────────────────────────────────────────
# LLM fills the NormalizedResponses schema directly, no code execution.
# The docstring is the prompt template — {responses} and {categories} are substituted
# with the actual argument values. The function body is empty (returns None) by design.

@ai_function(
    model=model,
    post_conditions=[validate_mapping],
    max_attempts=3,
)
def normalize_responses(responses: list[str], categories: list[str]) -> NormalizedResponses:
    """
    Classify each survey response into exactly one of the given categories.

    Responses to classify:
    {responses}

    Valid categories (use these exactly):
    {categories}

    Return a mappings dict where each key is the exact original response string and each value is its category.
    """


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    survey_responses = [
        "absolutely loved it, best experience ever!!",
        "it was ok i guess, nothing special",
        "TERRIBLE. wasted my money. never again.",
        "pretty good but the shipping took forever",
        "meh",
        "10/10 would recommend to friends & family",
        "the product broke after 2 days, very disappointed",
        "decent quality for the price, not bad not great",
        "worst customer service ive ever dealt with smh",
        "exceeded my expectations, pleasantly surprised!",
    ]
    categories = ["positive", "negative", "neutral", "mixed"]
    result = normalize_responses(survey_responses, categories)
    print(result.model_dump_json(indent=2))
