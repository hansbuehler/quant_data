"""
Utility functions for testing data
"""

import polars as pl
import pandas as pd
import numpy as np
import hashlib
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

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

def assert_pd_frame_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: str | list[str] | None = None,
    reset_index: bool = True,
    check_dtype: bool = True,
    check_exact: bool = False,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """
    Assert that two Pandas DataFrames are equal, with options for sorting and tolerance.
    """
    if sort_by is not None:
        actual = actual.sort_values(sort_by)
        expected = expected.sort_values(sort_by)

    if reset_index:
        actual = actual.reset_index(drop=True)
        expected = expected.reset_index(drop=True)

    pd_assert_frame_equal(
        actual,
        expected,
        check_dtype=check_dtype,
        check_exact=check_exact,
        rtol=rtol,
        atol=atol,
    )

def hash_pandas_df(
    df: pd.DataFrame,
    *,
    include_index: bool = True,
    include_schema: bool = True,
    digest_size: int = 16,
) -> str:
    """ Compute a hash of a pandas DataFrame, including its data and optionally its schema and index. """
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(b"pandas-dataframe-v1")

    if include_schema:
        h.update(repr(tuple(df.columns)).encode("utf-8"))
        h.update(repr(tuple(map(str, df.dtypes))).encode("utf-8"))
        if include_index:
            h.update(str(df.index.dtype).encode("utf-8"))

    row_hashes = pd.util.hash_pandas_object(
        df,
        index=include_index,
        hash_key="0123456789123456",
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)

    h.update(row_hashes.tobytes())
    return h.hexdigest()

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