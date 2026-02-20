"""
Basic example of running an AI Function with the Python execution environment enabled.
"""

from ai_functions import ai_function

@ai_function(code_execution_mode="local")
def natural_language_calculator(expression: str) -> float:
    """
    Evaluate the following expression: {expression}
    """

if __name__ == '__main__':
    expression = r"\sum_{i=1}^231 i**2"
    answer = natural_language_calculator(expression)
    print(f'{expression} = {answer}')
