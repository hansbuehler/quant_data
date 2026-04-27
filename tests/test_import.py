import quantdata


def test_import_package() -> None:
    assert quantdata.__version__ == "0.1.0"


def test_triangular_numbers_and_summary() -> None:
    values = quantdata.first_n_triangular_numbers(5)

    assert values == [1, 3, 6, 10, 15]
    assert quantdata.summarize_numbers(values) == {
        "count": 5.0,
        "total": 35.0,
        "mean": 7.0,
    }


def test_summary_docstring_is_generated() -> None:
    assert quantdata.summarize_numbers.__doc__ is not None
    assert "Return a compact summary" in quantdata.summarize_numbers.__doc__
