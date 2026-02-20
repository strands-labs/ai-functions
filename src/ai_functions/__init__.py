"""AI Functions - Enhanced AI Function interface.

This module provides the @ai_function decorator for transforming Python
functions into AI-powered functions using the Strands Agents SDK.

Usage:
    from ai_functions import ai_function
    from ai_functions.types import AIFunctionConfig, CodeExecutionMode, ValidationError

    @ai_function
    def summarize(text: str) -> str:
        '''Summarize: {text}'''
"""

from .decorator import ai_function
from .types import AIFunctionConfig, CodeExecutionMode, PostConditionResult

__all__ = [
    "AIFunctionConfig",
    "CodeExecutionMode",
    "PostConditionResult",
    "ai_function",
]
