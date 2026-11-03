"""
This module distinguishes integer inputs from booleans while accepting integer subclasses.

Python booleans inherit from int. Numeric input checks use is_integer() to accept integer values
without interpreting True and False as one and zero.
"""

from typing import TypeGuard


def is_integer(value: object) -> TypeGuard[int]:
    """
    Return whether a value is an int instance other than a boolean.

    Args:
        value (object): Value to classify; int subclasses are accepted.

    Returns:
        TypeGuard[int]: True for integer instances excluding True and False.
            False for every other type; no conversion is performed.
    """
    return isinstance(value, int) and not isinstance(value, bool)
