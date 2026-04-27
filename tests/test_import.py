import os



import quantdata
from quantdata.utils.numbers import first_n_triangular_numbers, summarize_numbers

def test_import_package() -> None:
    assert quantdata.__version__ == "0.1.0"


def test_triangular_numbers_and_summary() -> None:
    values = first_n_triangular_numbers(5)

    assert values == [1, 3, 6, 10, 15]
    assert summarize_numbers(values) == {
        "count": 5.0,
        "total": 35.0,
        "mean": 7.0,
    }


def test_summary_docstring_is_generated() -> None:
    assert summarize_numbers.__doc__ is not None
    assert "Return a compact summary" in summarize_numbers.__doc__
