"""One-shot execution — the simplest usage pattern.

An @ai_function is a template: prompt builder + output type + config.
Calling it directly creates a temporary handle, runs one cycle, and
returns the typed result. No coordinator needed.
"""

import asyncio

from pydantic import BaseModel

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult

# ── Simple: docstring prompt, primitive output ───────────────────


@ai_function(float)
def calculator(expression: str):
    """Evaluate the mathematical expression: {expression}"""


# ── Structured: explicit prompt, pydantic output ────────────────


class TranslationResult(BaseModel):
    translated: str
    confidence: float


@ai_function(TranslationResult)
def translate(text: str, target_language: str) -> str:
    return f"Translate the following to {target_language}:\n\n{text}"


# ── With post-conditions ────────────────────────────────────────


def confidence_above_threshold(result: TranslationResult, **kwargs: object) -> PostConditionResult | None:
    if result.confidence > 0.8:
        return None
    return PostConditionResult(passed=False, message=f"Confidence {result.confidence} below 0.8")


@ai_function(
    TranslationResult,
    post_conditions=[confidence_above_threshold],
    max_attempts=3,
)
def reliable_translate(text: str, target_language: str) -> str:
    return f"Translate to {target_language} (be confident):\n\n{text}"


async def main():
    # Each call is independent — no shared history
    result = await calculator(expression="(3 + 5) * 2")
    print(f"Calculator: {result}")

    translation = await translate(text="Hello world", target_language="French")
    print(f"Translation: {translation.translated} ({translation.confidence})")

    # Sync variant for scripts
    result = calculator.run_sync(expression="7 * 6")
    print(f"Sync: {result}")


if __name__ == "__main__":
    asyncio.run(main())
