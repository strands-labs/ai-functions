"""Review Analyzer — Sentiment Classification and Summarization Pipeline.

Classifies customer reviews by sentiment using a Literal return type,
then summarizes each review with a post-condition that enforces ALL CAPS output.

Demonstrates:
- Literal type as a return type to constrain classification output
- PostConditionResult from a plain Python function for runtime validation
- A post-condition designed to reliably trigger self-correction (the model
  rarely produces ALL CAPS unprompted, so you'll see the retry loop fire)
- Composing multiple ai_functions into a simple pipeline
"""

import json
from typing import Literal

from ai_functions import ai_function
from ai_functions.types import PostConditionResult

model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


# ── Classification ──────────────────────────────────────────────────────────
# Literal return type constrains the model to one of the specified values.
# No Pydantic model or JSON schema needed — the type annotation is the schema.


@ai_function(model=model)
def classify_sentiment(review: str) -> Literal["positive", "negative", "neutral"]:
    """
    Classify the sentiment of the following customer review.

    Review: {review}
    """


# ── Summarization with post-condition ───────────────────────────────────────
# The ALL CAPS constraint is deliberately chosen because the model almost never
# produces uppercase output unprompted. This makes the self-correction loop
# visible when running the example — watch for the retry in the console output.


def check_uppercase(summary: str) -> PostConditionResult:
    if summary != summary.upper():
        return PostConditionResult(passed=False, message="The summary must be in ALL CAPS. Rewrite it in uppercase.")
    return PostConditionResult(passed=True)


@ai_function(model=model, post_conditions=[check_uppercase], max_attempts=3)
def summarize_review(review: str) -> str:
    """
    Summarize the following customer review in one sentence, written in ALL CAPS.

    Review: {review}
    """


# ── Pipeline ────────────────────────────────────────────────────────────────
# ai_functions are regular callables — compose them with plain Python.
# These two calls are independent, so they could also be run in parallel
# by defining the ai_functions as async and using asyncio.gather():
#   sentiment, summary = await asyncio.gather(classify_sentiment(review), summarize_review(review))


def analyze_review(review: str) -> dict:
    sentiment = classify_sentiment(review)
    summary = summarize_review(review)
    return {"review": review, "sentiment": sentiment, "summary": summary}


if __name__ == "__main__":
    reviews = [
        "Absolutely loved this product! It arrived quickly and works perfectly. Would buy again.",
        "Terrible experience. The item broke after two days and customer support never responded.",
        "It's fine I guess. Does what it says, nothing more nothing less.",
    ]

    results = [analyze_review(r) for r in reviews]
    print(json.dumps(results, indent=2))
