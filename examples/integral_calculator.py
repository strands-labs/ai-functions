"""
Example of using an AI Functions with Python integration to compute an integral using sympy and
return its symbolic value as a native Python object.
"""

from ai_functions import ai_function
import sympy

@ai_function(code_execution_mode="local", code_executor_additional_imports=["sympy"])
def compute_integral(integral: str) -> sympy.Expr:
    """
    Please compute the following integral symbolically and return its value as a sympy expression:
    ---
    {integral}
    """


if __name__ == '__main__':
    answer = compute_integral(r"\int_{-\inf}^\inf e^{-x^2} dx")

    print("The symbolic value of the integral is:", answer)
    print("The numeric value is:", answer.evalf())
