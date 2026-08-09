import pandas as pd
import pytest

from portfolio_risk_model.data.normalization import (
    normalize_column_names,
    normalize_date_column,
    normalize_market_data,
    normalize_numeric_column,
    normalize_text_column,
    sort_rows,
)


def test_normalize_column_names_renames_source_columns() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "Ticker": ["SPY"],
            "Adj Close": [100.0],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset",
        "Adj Close": "adjusted_close",
    }

    result = normalize_column_names(df, column_mapping)

    assert list(result.columns) == [
        "date",
        "asset",
        "adjusted_close",
    ]


def test_normalize_column_names_does_not_modify_input() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "Adj Close": [100.0],
        }
    )

    column_mapping = {
        "Date": "date",
        "Adj Close": "adjusted_close",
    }

    normalize_column_names(df, column_mapping)

    assert list(df.columns) == [
        "Date",
        "Adj Close",
    ]


def test_normalize_column_names_preserves_unmapped_columns() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "Adj Close": [100.0],
            "Volume": [1_000_000],
        }
    )

    column_mapping = {
        "Date": "date",
        "Adj Close": "adjusted_close",
    }

    result = normalize_column_names(df, column_mapping)

    assert list(result.columns) == [
        "date",
        "adjusted_close",
        "Volume",
    ]


def test_normalize_column_names_raise_when_mapping_creates_duplicate_columns() -> None:
    df = pd.DataFrame(
        {
            "Adj Close": [100.0],
            "adjusted_close": [101.0],
        }
    )

    column_mapping = {
        "Adj Close": "adjusted_close",
    }

    with pytest.raises(
        ValueError,
        match=r"Column normalization created duplicate columns: adjusted_close",
    ):
        normalize_column_names(df, column_mapping)


def test_normalize_date_column_converts_strings_to_datetime() -> None:
    df = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
            ]
        }
    )

    result = normalize_date_column(df)

    assert pd.api.types.is_datetime64_ns_dtype(result["date"])


def test_normalize_date_column_raises_for_invalid_dates() -> None:
    df = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "not-a-date",
                "2025-01-03",
            ]
        }
    )

    with pytest.raises(ValueError):
        normalize_date_column(df)


def test_normalize_date_column_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-01-02",
            ]
        }
    )

    original = df.copy()

    normalize_date_column(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_normalize_numeric_column_converts_strings_to_float() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": [
                "100.5",
                "101.25",
                "102.0",
            ]
        }
    )

    result = normalize_numeric_column(
        df,
        column="adjusted_close",
    )

    assert result["adjusted_close"].dtype == "float64"


def test_normalize_numeric_column_raises_for_invalid_values() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": [
                "100.5",
                "not-a-number",
                "102.0",
            ]
        }
    )

    with pytest.raises(ValueError):
        normalize_numeric_column(
            df,
            column="adjusted_close",
        )


def test_normalize_numeric_column_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": [
                "100.5",
                "101.25",
            ]
        }
    )

    original = df.copy()

    normalize_numeric_column(
        df,
        column="adjusted_close",
    )

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_normalize_numeric_column_preserves_numeric_values() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": [
                100.5,
                101.25,
                102.0,
            ]
        }
    )

    result = normalize_numeric_column(
        df,
        column="adjusted_close",
    )

    assert result["adjusted_close"].tolist() == [
        100.5,
        101.25,
        102.0,
    ]


def test_normalize_text_column_strips_whitespace() -> None:
    df = pd.DataFrame(
        {
            "asset_id": [
                " SPY",
                "GLD ",
                " TLT ",
            ]
        }
    )

    result = normalize_text_column(
        df,
        column="asset_id",
    )

    assert result["asset_id"].tolist() == [
        "SPY",
        "GLD",
        "TLT",
    ]


def test_normalize_text_column_converts_to_string_dtype() -> None:
    df = pd.DataFrame(
        {
            "asset_id": [
                "SPY",
                "GLD",
            ]
        }
    )

    result = normalize_text_column(
        df,
        column="asset_id",
    )

    assert str(result["asset_id"].dtype) == "string"


def test_normalize_text_column_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "asset_id": [
                " SPY ",
                " GLD ",
            ]
        }
    )

    original = df.copy()

    normalize_text_column(
        df,
        column="asset_id",
    )

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_normalize_market_data_applies_all_normalization_steps() -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-01",
                "2025-01-02",
            ],
            "Ticker": [
                " SPY ",
                " GLD ",
            ],
            "Adj Close": [
                "100.5",
                "200.25",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    result = normalize_market_data(
        df,
        column_mapping=column_mapping,
    )

    assert result.columns.tolist() == [
        "date",
        "asset_id",
        "adjusted_close",
    ]

    assert str(result["date"].dtype) == "datetime64[ns]"
    assert str(result["asset_id"].dtype) == "string"
    assert str(result["adjusted_close"].dtype) == "float64"

    assert result["asset_id"].tolist() == [
        "SPY",
        "GLD",
    ]


def test_normalize_market_data_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "Ticker": [" SPY "],
            "Adj Close": ["100.5"],
        }
    )

    original = df.copy()

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    normalize_market_data(
        df,
        column_mapping=column_mapping,
    )

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_sort_rows_sorts_by_multiple_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-01",
                    "2025-01-01",
                ]
            ).as_unit("ns"),
            "asset_id": pd.Series(
                [
                    "SPY",
                    "SPY",
                    "GLD",
                ],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [
                    102.0,
                    101.0,
                    200.0,
                ],
                dtype="float64",
            ),
        }
    )

    result = sort_rows(
        df,
        columns=("date", "asset_id"),
    )

    assert result["asset_id"].tolist() == [
        "GLD",
        "SPY",
        "SPY",
    ]

    assert result["date"].tolist() == list(
        pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-01",
                "2025-01-02",
            ]
        ).as_unit("ns")
    )


def test_sort_rows_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-01",
                ]
            ).as_unit("ns"),
            "asset_id": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
        }
    )

    original = df.copy()

    sort_rows(
        df,
        columns=("date", "asset_id"),
    )

    pd.testing.assert_frame_equal(
        df,
        original,
    )
