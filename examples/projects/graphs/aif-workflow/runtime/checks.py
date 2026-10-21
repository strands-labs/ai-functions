"""Condition mechanics — copied into every .aif/<slug>/ package beside runtime.py.

Plumbing only: running a script, reading a file, comparing a number, hashing an input,
walking an AST. **The assertion itself stays in your `conditions.py`**, written out in
full, because the UX shows each condition's source so a human can audit what was
verified. A condition that reads `return checks.json_field_equals(...)` hides the very
thing the panel exists to show.

So: use these for the fiddly parts, and keep the comparison visible in your own
function body.

    def metrics_total_is_right(result: str) -> PostConditionResult | None:
        '''metrics.json reports the true total revenue.'''
        del result
        data = checks.read_json(METRICS) or {}           # plumbing
        wanted = expected_total()                        # computed from the input
        if not checks.close(data.get("total_revenue"), wanted):   # comparison, visible
            return PostConditionResult(
                passed=False,
                message=f"total_revenue is {data.get('total_revenue')!r}; it should be {wanted}.",
            )
        return None

Every reader here hits the filesystem on each call, deliberately: conditions are
re-evaluated once per attempt, and a cached read would grade attempt 2 against
attempt 1's content.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ai_functions.ai_thread.postcondition import PostConditionResult

CURRENCY = "${:,.2f}"
"""One definition of money formatting, so a workflow and its report agree."""


def currency(value: float) -> str:
    """Format a number as `$1,234.56`.

    Args:
        value: The amount to format.

    Returns:
        The amount with a dollar sign, thousands separators and two decimals.
    """
    return CURRENCY.format(value)


# ── Running things ────────────────────────────────────────────────────────────


def run_script(path: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run a Python script with this interpreter, from its own directory.

    Uses ``sys.executable`` so the script sees the same environment as the workflow,
    and never raises on a non-zero exit: the caller decides what a failure means.

    Args:
        path: The script to run.
        *args: Command-line arguments passed to the script.
        timeout: Seconds before the run is abandoned.

    Returns:
        The completed process, with ``stdout`` and ``stderr`` captured as text.

    Raises:
        subprocess.TimeoutExpired: The script ran longer than ``timeout``.
    """
    # Run from the script's folder so its relative paths resolve as the agent intended.
    return subprocess.run(
        [sys.executable, path.name, *args],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def run(command: list[str], *, cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    """Run an arbitrary command in ``cwd`` and capture its output.

    Args:
        command: The program and its arguments.
        cwd: Directory to run in.
        timeout: Seconds before the run is abandoned.

    Returns:
        The completed process, with ``stdout`` and ``stderr`` captured as text.

    Raises:
        subprocess.TimeoutExpired: The command ran longer than ``timeout``.
    """
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def must_exit_zero(path: Path, *args: str, timeout: int = 120) -> PostConditionResult | None:
    """Fail unless the script exists and exits zero; the message carries its stderr.

    Args:
        path: The script the step was asked to write.
        *args: Command-line arguments passed to the script.
        timeout: Seconds before the run is abandoned.

    Returns:
        ``None`` when the script exits zero, otherwise a failed ``PostConditionResult``
        whose message tells the agent what to fix.
    """
    # A missing script is its own failure, with its own instruction.
    if not path.is_file():
        return PostConditionResult(passed=False, message=f"{path.name} was not created; write it.")
    # Hand the agent the tail of stderr: enough to locate the error.
    proc = run_script(path, *args, timeout=timeout)
    if proc.returncode != 0:
        return PostConditionResult(
            passed=False,
            message=f"`python {path.name}` exited {proc.returncode}. Fix it:\n{tail(proc.stderr)}",
        )
    return None


def must_pass_command(command: list[str], *, cwd: Path, timeout: int = 180) -> PostConditionResult | None:
    """Fail unless the command exits zero; the message carries its output.

    Args:
        command: The program and its arguments, e.g. ``["pytest", "-q"]``.
        cwd: Directory to run in.
        timeout: Seconds before the run is abandoned.

    Returns:
        ``None`` when the command exits zero, otherwise a failed ``PostConditionResult``
        carrying the tail of its stdout (or stderr when stdout is empty).
    """
    proc = run(command, cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        return PostConditionResult(
            passed=False,
            message=(f"`{' '.join(command)}` exited {proc.returncode}. Fix it:\n{tail(proc.stdout or proc.stderr)}"),
        )
    return None


def tail(text: str, limit: int = 1200) -> str:
    """Return the last ``limit`` characters — enough context for a fix, not a wall.

    Args:
        text: Output to shorten.
        limit: Maximum characters kept from the end.

    Returns:
        The final ``limit`` characters of ``text``.
    """
    return text[-limit:]


# ── Reading things ────────────────────────────────────────────────────────────


def read_text(path: Path) -> str:
    """Return a file's text, or empty when it does not exist.

    Args:
        path: The file to read, fresh on every call.

    Returns:
        The file's UTF-8 text, or ``""`` when it is absent.
    """
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any] | None:
    """Return a JSON object, or ``None`` when the file is absent or malformed.

    Args:
        path: The JSON file to read, fresh on every call.

    Returns:
        The parsed object, or ``None`` when the file is missing, does not parse, or
        holds something other than an object.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    # A list or scalar is not the object a condition expects to index into.
    return data if isinstance(data, dict) else None


def read_json_list(path: Path) -> list[Any] | None:
    """Return a JSON array, or ``None`` when the file is absent or malformed.

    Args:
        path: The JSON file to read, fresh on every call.

    Returns:
        The parsed list, or ``None`` when the file is missing, does not parse, or
        holds something other than an array.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def read_csv(path: Path) -> list[dict[str, str]] | None:
    """Return CSV rows as dicts, or ``None`` when the file is absent.

    Args:
        path: The CSV file to read; its first row is the header.

    Returns:
        One dict per data row, keyed by header, with every value a string; ``None``
        when the file does not exist.
    """
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def missing_strings(path: Path, wanted: Iterable[str]) -> list[str]:
    """Return which of ``wanted`` do not appear in the file — empty means all present.

    Args:
        path: The file to search; a missing file contains nothing.
        wanted: Substrings that must appear.

    Returns:
        The items of ``wanted`` absent from the file, in their given order.
    """
    text = read_text(path)
    return [item for item in wanted if item not in text]


# ── Comparing things ──────────────────────────────────────────────────────────


def close(value: object, wanted: float, *, tolerance: float = 0.01) -> bool:
    """Whether a reported number matches within ``tolerance``.

    Use this rather than ``==`` on floats: an agent that computes in a different order
    lands a fraction of a cent away and is not wrong.

    Args:
        value: The value the agent reported, of any type.
        wanted: The expected number, computed from the inputs.
        tolerance: Largest difference still counted as a match.

    Returns:
        True when ``value`` is a real number (not a bool) within ``tolerance`` of ``wanted``.
    """
    # ``bool`` is an ``int`` subclass, so exclude it explicitly: ``True`` is not 1.0 here.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and abs(
            float(value) - wanted,
        )
        < tolerance
    )


def matches(value: object, wanted: object) -> bool:
    """Compare a reported value with an expected one, tolerantly for floats.

    The comparison is chosen by the type of ``wanted``: floats within a cent, bools
    by identity, ints exactly (rejecting bools), and anything else as text.

    Args:
        value: The value the agent reported.
        wanted: The expected value; its type selects the comparison.

    Returns:
        True when ``value`` matches ``wanted`` under that comparison.
    """
    if isinstance(wanted, float):
        return close(value, wanted)
    # Check bool before int: ``isinstance(True, int)`` is true.
    if isinstance(wanted, bool):
        return value is wanted
    if isinstance(wanted, int):
        return isinstance(value, int) and not isinstance(value, bool) and value == wanted
    return str(value) == str(wanted)


def is_number(value: object) -> bool:
    """Whether a JSON value is a number rather than a formatted string.

    Args:
        value: A value read from JSON.

    Returns:
        True for an int or float, False for a bool, a string such as ``"$1,234.56"``,
        or anything else.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ── Guarding inputs ───────────────────────────────────────────────────────────


def sha256(path: Path) -> str:
    """Return a file's SHA-256, or empty when it does not exist.

    Args:
        path: The file to hash.

    Returns:
        The hex digest of the file's bytes, or ``""`` when it is absent.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def file_unchanged(path: Path, *, because: str = "It is the input") -> Callable[[str], PostConditionResult | None]:
    """Build a condition asserting ``path`` is byte-identical to now.

    Call this at module level in ``conditions.py``, so the digest is captured *before*
    the agent runs. This is the negative condition that stops a task being satisfied
    by editing the thing it was supposed to satisfy — a failing test file, a source
    CSV, a spec.

    Args:
        path: The file that must stay unchanged.
        because: Why it must not change; included in the failure message.

    Returns:
        A post-condition named ``<stem>_unchanged`` that fails when the file was
        deleted or its contents differ from the digest taken now.
    """
    # Hash now, at build time — the whole point is a digest from before any attempt.
    original = sha256(path)

    def condition(result: str) -> PostConditionResult | None:
        del result
        if not path.is_file():
            return PostConditionResult(passed=False, message=f"{path.name} was deleted; restore it.")
        if sha256(path) != original:
            return PostConditionResult(
                passed=False,
                message=f"{path.name} was modified. {because} — restore it exactly and change your own output.",
            )
        return None

    # Name and describe the closure, since the runtime shows both as the ticket.
    condition.__name__ = f"{path.stem}_unchanged"
    condition.__doc__ = f"{path.name} is byte-identical to how it started."
    return condition


# ── Walking Python source ─────────────────────────────────────────────────────


def _tree(path: Path) -> ast.Module | None:
    """Parse a Python file, or ``None`` when it is absent or unparseable.

    Args:
        path: The Python source file.

    Returns:
        The module's AST, or ``None`` when the file is missing or has a syntax error.
    """
    if not path.is_file():
        return None
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None


def functions_in(path: Path) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every function defined in a Python file, nested ones included.

    Empty when the file is missing or does not parse — the condition that cares should
    say so itself, with a message the agent can act on.

    Args:
        path: The Python source file.

    Returns:
        Every ``def`` and ``async def`` in the file, in AST walk order.
    """
    tree = _tree(path)
    if tree is None:
        return []
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]


def function_names(path: Path) -> set[str]:
    """Return the names of every function defined in a Python file.

    Args:
        path: The Python source file.

    Returns:
        The set of function names; empty when the file is missing or does not parse.
    """
    return {node.name for node in functions_in(path)}


def undocumented_functions(path: Path) -> list[str]:
    """Return the functions with no docstring.

    Args:
        path: The Python source file.

    Returns:
        Names of functions lacking a docstring.
    """
    return [node.name for node in functions_in(path) if not ast.get_docstring(node)]


def unannotated_functions(path: Path) -> list[str]:
    """Return the functions missing a type hint on a parameter or on their return.

    Args:
        path: The Python source file.

    Returns:
        Names of functions with an unannotated parameter (other than ``self``) or no
        return annotation. ``*args`` and ``**kwargs`` are not inspected.
    """
    offenders: list[str] = []
    for node in functions_in(path):
        # Positional-only, regular and keyword-only parameters; ``self`` needs no hint.
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.returns is None or any(arg.annotation is None for arg in arguments if arg.arg != "self"):
            offenders.append(node.name)
    return offenders


def long_functions(path: Path, limit: int) -> dict[str, int]:
    """Return the functions whose body exceeds ``limit`` statements, with their length.

    Length counts top-level statements in the body, not lines, so a nested block
    counts once.

    Args:
        path: The Python source file.
        limit: Largest body length allowed.

    Returns:
        A mapping from each over-long function's name to its statement count.
    """
    return {node.name: len(node.body) for node in functions_in(path) if len(node.body) > limit}
