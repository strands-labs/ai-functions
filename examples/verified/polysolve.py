"""Counting the solutions of a polynomial system too big for its solver.

The system has five equations in five unknowns. msolve counts its solutions
exactly, but needs about 1500 s for a system of this shape, and it is given 420 s.
Eliminating one unknown gives an equivalent system it finishes in under 300 s.
So the agent does not count anything itself: it rewrites the system in Lean,
hands the smaller one to msolve, and proves the rewrite did not change the answer.

The certificate states the trust boundary exactly: msolve's count for the system
it was actually given is trusted, the elimination is proved, and the assumption
that eliminating a variable keeps the count is a hypothesis of the contract.
msolve counts solutions with multiplicity, so the answer is valid only if no solution repeats.

Needs msolve (https://msolve.lip6.fr) on `PATH`, or `MSOLVE_BIN` set to it.

Demonstrates:
- Reformulating a problem in Lean so a trusted tool can solve it
- A tool whose result carries a guarantee (`Certified`)
- Lean options for proofs that compute (`LeanProject(options=...)`)
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from example_helpers import models

from ai_functions import scope
from ai_functions.cli import print_event
from ai_functions.experimental import verified
from ai_functions.experimental.verified.function import Certified
from ai_functions.experimental.verified.lean import LeanProject

# The checks are `rfl` over polynomials with hundreds of terms, past Lean's default limits.
project = LeanProject(
    Path(__file__).parent / "lean",
    imports=["PolySolve"],
    options={"maxRecDepth": 4_000_000, "maxHeartbeats": 40_000_000},
)
PolySolve = project.symbols.PolySolve

SOLVER_TIMEOUT = 420.0
"""Seconds msolve gets: enough for the reduced system, not for the original."""

RawPoly = list[tuple[int, int, int, int, int, int]]
"""Terms `(e1, e2, e3, e4, e5, coefficient)`: the exponents of x1 … x5, then the coefficient."""


def msolve_input(system: list[RawPoly], k: int) -> str:
    """The system in msolve's input format, over x1 … xk."""
    names = [f"x{i + 1}" for i in range(k)]
    lines = []
    for poly in system:
        terms = []
        for *exponents, coefficient in poly:
            if any(exponents[k:]):
                raise ValueError(f"a term mentions a variable after x{k}: {tuple(exponents)}")
            if coefficient == 0:
                continue
            powers = zip(names, exponents[:k], strict=True)
            monomial = "".join(f"*{n}^{e}" if e > 1 else f"*{n}" for n, e in powers if e)
            terms.append(f"{'+' if coefficient > 0 else '-'}{abs(coefficient)}{monomial}")
        lines.append("".join(terms).lstrip("+") or "0")
    return ",".join(names) + "\n0\n" + ",\n".join(lines) + "\n"


def msolve_count(system: list[RawPoly], k: int) -> int:
    """Run msolve and read the solution count off its rational parametrisation."""
    binary = os.environ.get("MSOLVE_BIN") or shutil.which("msolve")
    if binary is None:
        raise RuntimeError("msolve not found: put it on PATH or set MSOLVE_BIN")
    with tempfile.TemporaryDirectory() as scratch:
        source, target = Path(scratch) / "system.ms", Path(scratch) / "out.txt"
        source.write_text(msolve_input(system, k))
        try:
            run = subprocess.run(
                [binary, "-f", str(source), "-o", str(target), "-P", "2", "-t", str(os.cpu_count() or 1)],
                capture_output=True,
                text=True,
                timeout=SOLVER_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"msolve did not finish within {SOLVER_TIMEOUT:.0f} s on a system in {k} unknowns"
            ) from None
        if run.returncode != 0 or not target.exists():
            raise RuntimeError(f"msolve failed: {(run.stderr or run.stdout).strip()[:400]}")
        output = target.read_text().strip()
    # `-P 2` prints `[0, [0, nvars, degree, …`: the degree is the number of solutions.
    match = re.match(r"\[\s*0\s*,\s*\[\s*0\s*,\s*\d+\s*,\s*(\d+)\s*,", output)
    if match is None:
        raise RuntimeError(f"msolve found no finite solution set: {output[:200]!r}")
    return int(match.group(1))


@verified.tool(PolySolve.msolveCount, stem="msolve")
def solve_with_msolve(k: int, system: list[RawPoly]) -> Certified:
    """Count the solutions of a polynomial system in x1 … xk with msolve, within 420 s.

    Returns the number of solutions, with the guarantee `HasSolutionCount k system n`.
    `system` must not mention a variable after xk. Refuses any system it cannot finish
    in time; a refused call records nothing.
    """
    count = msolve_count(system, k)
    return Certified(count, guarantees=lambda value, k, system: f"PolySolve.HasSolutionCount {k} {system} {value}")


# The contract `PolySolve.CountCorrect n system` is about the original system, which
# msolve cannot finish. The helpers for building polynomials, the certificate, and
# the theorems assembling the answer are in the project source.
@verified.ai_function(
    contract=PolySolve.CountCorrect,
    tools=[solve_with_msolve],
    model=models.large,
    max_attempts=12,
    timeout=600,
)
def count_solutions(system: list[RawPoly]) -> int:
    """How many solutions does the polynomial system H.system in x1 … x5 have?

    Each polynomial is a list of terms `(e1, e2, e3, e4, e5, c)`. The solver is exact
    but slow, and has a time budget: a system in fewer unknowns is easier for it.
    """


async def main() -> None:
    data = json.loads(Path(__file__).with_name("polysolve_system.json").read_text())
    system = [[tuple(term) for term in poly] for poly in data]
    print("Preparing Lean and counting solutions...", flush=True)
    async with scope(on_event=print_event):
        count, certificate = await count_solutions.with_certificate(system)
    print(f"count_solutions(5 equations in 5 unknowns) = {count}")
    print(f"Proved: {certificate.goal}")
    # The one thing the proof trusts: msolve's count for the system it was given.
    for fact in certificate.used.values():
        print(f"  trusted: {fact.symbol} in x1 … x{fact.arguments[0]} = {fact.value}")
    path = certificate.write(Path(__file__).parent / "data" / "polysolve.lean")
    print(f"Certificate: {path}")


if __name__ == "__main__":
    asyncio.run(main())
