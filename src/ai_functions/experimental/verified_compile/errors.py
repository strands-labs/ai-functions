"""Internal failure categories; callers can catch the existing AIFunctionError."""

from ...ai_thread.errors import AIFunctionError


class ContractError(AIFunctionError):
    """A Python contract is outside the supported deterministic subset."""


class CompilerError(AIFunctionError):
    """The private compiler or native build could not run."""


class ModelSetupError(AIFunctionError):
    """The model provider requires configuration before synthesis can start."""


class CandidateError(Exception):
    """A generated implementation or proof failed validation and can be retried."""


class SynthesisError(AIFunctionError):
    """Synthesis exhausted its budget without producing a verified artifact."""

    def __init__(self, function_name: str, diagnostics: list[str]) -> None:
        self.diagnostics = tuple(diagnostics)
        self.attempts = len(diagnostics)
        super().__init__(
            f"Could not generate a verified implementation of {function_name!r} "
            f"after {self.attempts} attempt(s). No implementation was installed.",
            function_name=function_name,
        )
