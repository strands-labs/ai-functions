"""One-shot execution — the simplest usage pattern.

An @ai_function is a template: prompt builder + output type + config.
Calling it directly creates a temporary handle, runs one cycle, and
returns the typed result.
"""

import asyncio

from example_helpers import models
from example_helpers.utils import display
from pydantic import BaseModel

from ai_functions import ai_function

# Simple: docstring prompt, primitive output.


@ai_function(model=models.medium)
def calculator(expression: str) -> float:
    """Evaluate the mathematical expression: {expression}"""


# Structured: explicit prompt, Pydantic output.


class TranslationResult(BaseModel):
    translated: str
    confidence: float


@ai_function(model=models.medium)
def translate(text: str, target_language: str) -> TranslationResult:
    return f"Translate the following to {target_language}:\n\n{text}"


# With post-conditions: retry until the result meets a runtime check.


def confidence_above_threshold(result: TranslationResult, **kwargs: object):
    assert result.confidence > 0.8, f"Confidence {result.confidence} below 0.8"


@ai_function(model=models.medium, post_conditions=[confidence_above_threshold], max_attempts=3)
def reliable_translate(text: str, target_language: str) -> TranslationResult:
    return f"Translate to {target_language} (be confident):\n\n{text}"


async def main():
    # Each call is independent — no shared history.
    result = await calculator(expression="(3 + 5) * 2")
    display("Calculator", str(result))

    translation = await translate(text="Hello world", target_language="French")
    display("Translation", f"{translation.translated} ({translation.confidence})")

    # Sync variant for scripts.
    result = calculator.run_sync(expression="7 * 6")
    display("Sync", str(result))


if __name__ == "__main__":
    asyncio.run(main())
