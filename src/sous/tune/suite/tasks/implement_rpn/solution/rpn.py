"""Reverse Polish notation over integers.

tokenize(text) splits the text on whitespace and returns the tokens as a
list of strings; empty or blank text gives an empty list.

evaluate(tokens) evaluates the tokens as a reverse Polish expression: a
numeric token (an int literal, possibly negative) is pushed on a stack; an
operator token ("+", "-", "*", "/") pops the right operand, then the left
one, applies the operator and pushes the result. "/" is floor division (the
// operator). It returns the single value left on the stack. It raises
ValueError for a token that is neither a number nor an operator, for an
operator with fewer than two values on the stack, and when anything but
exactly one value is left at the end.
"""

_OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b,
}


def tokenize(text: str) -> list[str]:
    return text.split()


def evaluate(tokens: list[str]) -> int:
    stack: list[int] = []
    for token in tokens:
        if token in _OPERATORS:
            if len(stack) < 2:
                raise ValueError(f"operator {token!r} needs two operands")
            right, left = stack.pop(), stack.pop()
            stack.append(_OPERATORS[token](left, right))
        else:
            try:
                stack.append(int(token))
            except ValueError:
                raise ValueError(f"unknown token {token!r}") from None
    if len(stack) != 1:
        raise ValueError(f"{len(stack)} values left on the stack")
    return stack[0]
