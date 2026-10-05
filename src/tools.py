"""
    Temp tools for now
"""

from langchain_core.tools import tool

@tool
def meow() -> str:
    """Returns string Meow
    """

    return f"Meow"

@tool
def woof() -> str:
    """Returns woof
    """
    return f"Woof"