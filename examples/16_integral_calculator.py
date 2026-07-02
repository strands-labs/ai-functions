"""
Example of using an AI Functions with Python integration to compute an integral using sympy and
return its symbolic value as a native Python object.
"""

import sympy

from ai_functions import ai_function


# `code_executor_additional_imports` widens the sandbox allowlist so the agent
# may `import sympy` inside the python_executor; the FinalAnswer wrapper allows
# arbitrary types so the answer can be a native ``sympy.Expr``.
@ai_function(sympy.Expr, code_execution_mode="local", code_executor_additional_imports=["sympy"])
def compute_integral(integral: str) -> sympy.Expr:
    """
    Please compute the following integral symbolically and return its value as a sympy expression:
    ---
    {integral}
    """


if __name__ == "__main__":
    # AI functions are async by default; run_sync drives one from sync code.
    answer = compute_integral.run_sync(integral=r"\int_{-\inf}^\inf e^{-x^2} dx")
    print("The symbolic value of the integral is:", answer)
    print("The numeric value is:", answer.evalf())
