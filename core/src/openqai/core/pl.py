"""
Polars Utilities
================

"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from cdxcore.err import verify_inp
from cdxcore.util import fmt_list

from polars.testing import assert_frame_equal as pl_assert_frame_equal
import hashlib

def create_empty_df(schema: Mapping[str, Any]) -> pl.DataFrame:
    """
    Creates an empty :class:`polars.DataFrame` from a schema.

    A schema is a dictionary of the form::

        {
            "col_a": pl.Float32,
            "col_b": pl.Int64,
        }
    """
    assert isinstance(schema, Mapping), "schema must be a mapping of column name -> dtype"
    return pl.DataFrame(schema=dict(schema))


def _as_polars_dtype(dtype: Any, *, col: str) -> pl.DataType:
    try:
        return pl.Series(col, [], dtype=dtype).dtype
    except (TypeError, ValueError) as exc:
        verify_inp(False, f"Invalid dtype for column {col!r}: {dtype!r}. Error: {exc}")


def verify_schema(
    df: pl.DataFrame,
    schema: Mapping[str, Any],
    header: str | callable[[], str] | None = None,
    *,
    allow_extra_columns: bool = False,
) -> pl.DataFrame:
    """
    Validate a :class:`polars.DataFrame` against a schema of the form::

        {
            "col_a": pl.Float32,
            "col_b": pl.Int64,
        }

    Checks:
    - all schema columns exist
    - no extra columns, unless allow_extra_columns=True
    - exact dtype match

    Returns df unchanged, so it can be used inline.
    Raises :class:`ValueError` if validation fails. The error message will be
    preceeded by ``header`` which can either be a string or a callable returning
    a string (to avoid expensive message construction if validation passes).

    Parameters
    ----------
    df : :class:`polars.DataFrame`
        Data frame to validate.
    schema : Mapping[str, Any]
        Mapping of column name to expected dtype.
    header : str|callable[[],str]| None, optional
        Header for error messages. Can be a string or a callable returning a string. Using a callable delays string composition to avoid unnecessary overhead when validation passes.
    allow_extra_columns : bool, default False
        Whether to allow columns in the DataFrame that are not in the schema.
    """

    def real_header():
        if header is None:
            return ""
        elif isinstance(header, str):
            return header + ": "
        else:
            return header() + ": "

    assert isinstance(df, pl.DataFrame), f"{real_header()} must be a polars DataFrame"
    assert isinstance(
        schema, Mapping
    ), f"{real_header()}schema must be a mapping of column name -> dtype"

    expected_cols = set(schema.keys())
    actual_cols = set(df.columns)

    missing = sorted(expected_cols - actual_cols)
    extra = sorted(actual_cols - expected_cols)

    if len(missing) > 0:
        if len(extra) > 0:
            raise ValueError(
                f"{real_header()}is missing columns: {fmt_list(missing)} and has extra columns: {fmt_list(extra)}."
            )
        raise ValueError(f"{real_header()}is missing columns {fmt_list(missing)}.")
    if len(extra) > 0 and not allow_extra_columns:
        raise ValueError(f"{real_header()}has extra columns {fmt_list(extra)}.")

    dtype_errors: list[str] = []
    for col, expected_dtype_raw in schema.items():
        if col not in df.columns:
            continue

        expected_dtype = _as_polars_dtype(expected_dtype_raw, col=col)
        actual_dtype = df.schema[col]

        if actual_dtype != expected_dtype:
            dtype_errors.append(f"{col!r}: expected {expected_dtype!r}, got {actual_dtype!r}")

    if dtype_errors:
        raise ValueError(f"{real_header()} has schema mismatches: {fmt_list(dtype_errors)}.")

    return df


def assert_pl_frame_equal(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    sort_by: str | list[str] | None = None,
    check_dtypes: bool = True,
    check_column_order: bool = True,
    check_exact: bool = False,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-12,
) -> None:
    """
    Assert that two Polars DataFrames are equal, with options for sorting and tolerance.
    """
    if sort_by is not None:
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        actual = actual.sort(sort_by)
        expected = expected.sort(sort_by)

    pl_assert_frame_equal(
        actual,
        expected,
        check_row_order=True,
        check_column_order=check_column_order,
        check_dtypes=check_dtypes,
        check_exact=check_exact,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def hash_polars_df(
    df: pl.DataFrame,
    *,
    include_schema: bool = True,
    digest_size: int = 16,
) -> str:
    """ Compute a hash of a Polars DataFrame, including its data and optionally its schema. """
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(b"polars-dataframe-v1")

    if include_schema:
        h.update(repr(tuple((name, str(dtype)) for name, dtype in df.schema.items())).encode("utf-8"))

    row_hashes = df.hash_rows(
        seed=0,
        seed_1=1,
        seed_2=2,
        seed_3=3,
    ).to_numpy()

    h.update(row_hashes.tobytes())
    return h.hexdigest()


