"""Errors shared by Lean consumers."""


class LeanError(RuntimeError):
    """A Lean operation failed."""


class LeanSetupError(LeanError):
    """A toolchain or project could not be prepared."""


class LeanTimeoutError(LeanError):
    """A Lean operation exceeded its deadline."""
