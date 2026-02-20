"""
Example showing the use of async to AI Functions to translate a sentence to multiple languages in parallel.
"""

import asyncio

from ai_functions import ai_function
from ai_functions.types import PostConditionResult

model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Post-condition to prevent a common failure case where the model outputs
# a transliteration for non-latin scripts
@ai_function(model=model)
def check_translation(text: str) -> PostConditionResult:
    """
    Check that the following text is written in the native script of the language and does not contain any romanization.
    ```
    {text}
    ```
    Answer immediately without explaining your thinking.
    """

# Simply define the AI Function as async to use it as any other async function
@ai_function(model=model, post_conditions=[check_translation])
async def translate_text(text: str, lang: str) -> str:
    """
    Translate the text below to the following language: `{lang}`.
    ```
    {text}
    ```
    """


async def main():
    text = 'It was the best of times, it was the worst of times'
    languages = ['fr', 'ja', 'it', 'zh']
    # run multiple functions in parallel to translate to each language and wait for all of them to terminate
    translations = await asyncio.gather(*(translate_text(text, lang) for lang in languages))

    print()
    print(f'=== {text} ===')
    for lang, translation in zip(languages, translations, strict=True):
        print(f'({lang}) {translation}')


if __name__ == '__main__':
    asyncio.run(main())
