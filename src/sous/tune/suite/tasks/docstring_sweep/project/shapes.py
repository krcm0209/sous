"""Areas and perimeters of simple shapes."""

import math


def _check_positive(*values: float) -> None:
    for value in values:
        if value <= 0:
            raise ValueError("dimensions must be positive")


def area_rect(width: float, height: float) -> float:
    """Area of a rectangle."""
    _check_positive(width, height)
    return width * height


def perimeter_rect(width: float, height: float) -> float:
    _check_positive(width, height)
    return 2 * (width + height)


def area_circle(radius: float) -> float:
    """Area of a circle."""
    _check_positive(radius)
    return math.pi * radius * radius


def perimeter_circle(radius: float) -> float:
    _check_positive(radius)
    return 2 * math.pi * radius


def area_triangle(base: float, height: float) -> float:
    _check_positive(base, height)
    return base * height / 2


def scale(value: float, factor: float) -> float:
    _check_positive(value, factor)
    return value * factor
