"""Compute an integral with sympy inside a code-executing AI function.

The result comes back as a native ``sympy.Expr``, not a string, so it can be
evaluated numerically in plain Python afterwards.
"""

import sympy
from example_helpers import models
from example_helpers.utils import display

from ai_functions import ai_function


# `code_executor_additional_imports` widens the sandbox allowlist so the agent
# may `import sympy` inside the python_executor; the FinalAnswer wrapper allows
# arbitrary types so the answer can be a native ``sympy.Expr``.
@ai_function(model=models.medium, code_execution_mode="local", code_executor_additional_imports=["sympy"])
def compute_integral(integral: str) -> sympy.Expr:
    """
    Please compute the following integral symbolically and return its value as a sympy expression:
    ---
    {integral}
    """


if __name__ == "__main__":
    # AI functions are async by default; run_sync drives one from sync code.
    answer = compute_integral.run_sync(integral=r"\int_{-\inf}^\inf e^{-x^2} dx")
    display("Symbolic value", str(answer), lang="python")
    display("Numeric value", str(answer.evalf()), lang="text")
