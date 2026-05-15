"""
Panda Utilities
===============
"""

from __future__ import annotations

import pandas as pd
from collections.abc import Mapping
from typing import Any
import numpy as np
from cdxcore.err import verify_inp
from cdxcore.util import fmt_list


def create_empty_df( schema ):
    """
    Creates an empty :class:`pandas.DataFrame` from a schema.

    A schema is a dictionary of the form::        

        {
            "col_a": np.dtype("float32"),
            "col_b": np.dtype("int64"),
        } 
    """
    return pd.DataFrame({c: pd.Series(dtype=dtype) for c,dtype in schema.items()})

def _as_pandas_dtype(dtype: Any, *, col: str) -> Any:
    try:
        return pd.api.types.pandas_dtype(dtype)
    except (TypeError, ValueError) as exc:
        verify_inp(False, f"Invalid dtype for column {col!r}: {dtype!r}. Error: {exc}")

def verify_schema(
    df: pd.DataFrame,
    schema: Mapping[str, Any],
    header : str|callable[[],str]| None = None,
    *,
    allow_extra_columns: bool = False,
) -> pd.DataFrame:
    """
    Validate a :class:`pandas.DataFrame` against a schema of the form::

        {
            "col_a": np.dtype("float32"),
            "col_b": np.dtype("int64"),
        }

    Checks:
    - all schema columns exist
    - no extra columns, unless allow_extra_columns=True
    - exact dtype match

    Returns df unchanged, so it can be used inline.
    Raises :class:`ValueError` if validation fails. The error message will be preeded by ``header`` which can either
    be a string or a callable returning a string (to avoid expensive message construction if validation passes).

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
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

    assert isinstance(df, pd.DataFrame), f"{real_header()} must be a pandas DataFrame"
    assert isinstance(schema, Mapping), f"{real_header()}schema must be a mapping of column name -> dtype"

    expected_cols = set(schema.keys())
    actual_cols = set(df.columns)

    missing = sorted(expected_cols - actual_cols)
    extra = sorted(actual_cols - expected_cols)

    if len(missing) > 0:
        if len(extra) > 0:
            raise ValueError( f"{real_header()} is missing columns {fmt_list(missing)} and has extra columns {fmt_list(extra)}.")
        raise ValueError( f"{real_header()} is missing columns {fmt_list(missing)}.")
    if len(extra) > 0 and not allow_extra_columns:
        raise ValueError( f"{real_header()} has extra columns {fmt_list(extra)}.")
        
    dtype_errors: list[str] = []
    for col, expected_dtype_raw in schema.items():
        if col not in df.columns:
            continue

        expected_dtype = _as_pandas_dtype(expected_dtype_raw, col=col)
        actual_dtype = _as_pandas_dtype(df[col].dtype, col=col)

        if actual_dtype != expected_dtype:
            dtype_errors.append(
                f"{col!r}: expected {expected_dtype!r}, got {actual_dtype!r}"
            )

    if dtype_errors:
        raise ValueError( f"{real_header()} has schema mismatches: {fmt_list(dtype_errors)}.")

    return df