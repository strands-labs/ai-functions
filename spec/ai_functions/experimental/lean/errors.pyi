"""Errors shared by Lean tooling."""

class LeanError(RuntimeError):
    """A Lean operation failed."""

class LeanSetupError(LeanError):
    """A toolchain could not be prepared."""

class LeanTimeoutError(LeanError):
    """A Lean operation exceeded its deadline."""
