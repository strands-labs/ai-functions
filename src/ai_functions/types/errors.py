"""Error types for AI Functions.

This module defines the structured error hierarchy for AI Function errors.
All errors inherit from AIFunctionError and include context about the failure.
"""


class AIFunctionError(Exception):
    """Base exception for all AI function errors.

    All AI function errors inherit from this class, providing a consistent
    interface for error handling and context information.

    Attributes:
        function_name: Name of the AI function that failed
        message: Human-readable error message
    """

    def __init__(
        self,
        message: str,
        function_name: str,
    ):
        """Initialize AIFunctionError.

        Args:
            message: Human-readable error message
            function_name: Name of the AI function that failed
        """
        self.message = message
        self.function_name = function_name
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        return f"[{self.function_name}] {self.message}"


class ValidationError(AIFunctionError):
    """Post-condition validation failed.

    This error is raised when one or more validation conditions fail.

    Attributes:
        validation_errors: Mapping of condition name to error message
    """

    def __init__(
        self,
        function_name: str,
        validation_errors: dict[str, str],
    ):
        """Initialize ValidationError.

        Args:
            function_name: Name of the AI function that failed
            validation_errors: Mapping of condition name to error message
        """
        self.validation_errors = validation_errors
        super().__init__(
            message=self._format_validation_message(),
            function_name=function_name,
        )

    def _format_errors(self) -> str:
        """Format validation errors as a message."""
        return "\n".join(f"- {name}: {error}" for name, error in self.validation_errors.items())

    def _format_validation_message(self) -> str:
        """Format the validation error message."""
        return f"Validation failed:\n{self._format_errors()}"
