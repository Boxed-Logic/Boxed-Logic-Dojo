"""Arithmetic tool for GSM8K math reasoning.

Exposes a single @tool-decorated function, calculate(), that evaluates
an arithmetic expression and returns the result as a string.  Two layers
of sandboxing prevent arbitrary code execution:

  1. Regex whitelist — only digits, whitespace, and the operators
     +  -  *  /  (  )  .  ,  are allowed through.
  2. Builtins-free eval — even if a malformed expression somehow passed
     the regex, eval runs with an empty builtins dict so no built-in
     names (open, import, etc.) are accessible.
"""
from __future__ import annotations
import re

from dojo.tools import tool

# Whitelist: digits, whitespace, arithmetic operators, parentheses, decimal point, comma.
# re.match checks from the start; the $ anchor requires the pattern to cover the entire string.
_SAFE_RE = re.compile(r'^[\d\s\+\-\*\/\(\)\.\,]+$')


@tool(
    name="calculate",
    description="Evaluate an arithmetic expression and return the numeric result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "An arithmetic expression using +, -, *, /, parentheses, and numbers. "
                    "Commas in numbers are stripped automatically (e.g. '1,000 * 3' is valid)."
                ),
            }
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result as a string.

    Integer results are formatted without a decimal point ('42' not '42.0')
    so they are easier for the model to match against whole-number answers.
    Errors (unsafe input, division by zero, syntax errors) are returned as
    an 'Error: ...' string so the model can adjust and retry.

    Args:
        expression: Arithmetic expression, e.g. '(12 + 8) * 3'.

    Returns:
        The numeric result as a string, or an 'Error: ...' message.
    """
    clean = expression.replace(",", "").strip()
    if not _SAFE_RE.match(clean):
        return "Error: unsafe expression"
    try:
        result = eval(clean, {"__builtins__": {}}, {})
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Error: {e}"
