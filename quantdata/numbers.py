"""Numeric helper functions used to validate packaging, testing, and docs."""

from __future__ import annotations


def _build_summary_docstring() -> str:
    return (
        "Return a compact summary for a sequence of numbers.\n\n"
        "Parameters\n"
        "----------\n"
        "values:\n"
        "    Sequence of numeric values to summarize.\n\n"
        "Returns\n"
        "-------\n"
        "dict[str, float]\n"
        "    Dictionary containing the count, total, and arithmetic mean.\n"
    )


def first_n_triangular_numbers(count: int) -> list[int]:
    """Compute the first ``count`` triangular numbers."""
    if count < 0:
        raise ValueError("count must be non-negative")
    return [index * (index + 1) // 2 for index in range(1, count + 1)]


def summarize_numbers(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values must not be empty")

    total = float(sum(values))
    count = float(len(values))
    return {
        "count": count,
        "total": total,
        "mean": total / count,
    }


summarize_numbers.__doc__ = _build_summary_docstring()
