from __future__ import annotations

import polars as pl
import pytest

from quantdata.utils.pl import create_empty_df, verify_schema


def test_create_empty_df_preserves_schema_columns_and_dtypes() -> None:
    schema = {
        "name": pl.String,
        "count": pl.Int32,
        "weight": pl.Float32,
        "ts": pl.Datetime("ns", "America/New_York"),
    }

    df = create_empty_df(schema)

    assert df.is_empty()
    assert df.schema == schema


def test_verify_schema_returns_same_dataframe_for_matching_schema() -> None:
    schema = {
        "count": pl.Int32,
        "weight": pl.Float32,
    }
    df = pl.DataFrame(
        {
            "count": [1, 2],
            "weight": [1.5, 2.5],
        },
        schema=schema,
    )

    validated = verify_schema(df, schema)

    assert validated is df


def test_verify_schema_header_callable_is_lazy_on_success() -> None:
    df = pl.DataFrame({"count": [1]}, schema={"count": pl.Int32})
    calls: list[str] = []

    def header() -> str:
        calls.append("called")
        return "sample"

    verify_schema(df, {"count": pl.Int32}, header)

    assert calls == []


def test_verify_schema_rejects_missing_and_extra_columns_with_header() -> None:
    df = pl.DataFrame(
        {
            "count": [1],
            "extra": [2],
        },
        schema={
            "count": pl.Int32,
            "extra": pl.Int32,
        },
    )

    with pytest.raises(ValueError, match="sample: .*missing columns.*weight.*extra columns.*extra"):
        verify_schema(
            df,
            {
                "count": pl.Int32,
                "weight": pl.Float32,
            },
            "sample",
        )


def test_verify_schema_rejects_extra_columns_unless_explicitly_allowed() -> None:
    df = pl.DataFrame(
        {
            "count": [1],
            "extra": [2],
        },
        schema={
            "count": pl.Int32,
            "extra": pl.Int32,
        },
    )
    schema = {"count": pl.Int32}

    with pytest.raises(ValueError, match="extra columns.*extra"):
        verify_schema(df, schema)

    validated = verify_schema(df, schema, allow_extra_columns=True)
    assert validated is df


def test_verify_schema_rejects_dtype_mismatch() -> None:
    df = pl.DataFrame({"count": [1]}, schema={"count": pl.Int64})

    with pytest.raises(ValueError, match="schema mismatches.*'count'.*Int32.*Int64"):
        verify_schema(df, {"count": pl.Int32})


def test_verify_schema_invokes_header_callable_only_when_needed() -> None:
    df = pl.DataFrame({"count": [1]}, schema={"count": pl.Int32})
    calls: list[str] = []

    def header() -> str:
        calls.append("called")
        return "sample"

    with pytest.raises(ValueError, match="sample: .*missing columns.*weight"):
        verify_schema(
            df,
            {
                "count": pl.Int32,
                "weight": pl.Float32,
            },
            header,
        )

    assert calls == ["called"]


def test_verify_schema_rejects_invalid_inputs() -> None:
    with pytest.raises(AssertionError, match="must be a polars DataFrame"):
        verify_schema([], {"count": pl.Int32})

    with pytest.raises(AssertionError, match="schema must be a mapping"):
        verify_schema(pl.DataFrame({"count": [1]}), ["count"])


def test_verify_schema_rejects_invalid_schema_dtype() -> None:
    df = pl.DataFrame({"count": [1]}, schema={"count": pl.Int32})

    with pytest.raises(ValueError, match="Invalid dtype for column 'count'"):
        verify_schema(df, {"count": object()})
