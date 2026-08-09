from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from portfolio_risk_model.data.pipeline import (
    build_interim_market_data,
    build_processed_market_data,
    fetch_market_data,
    ingest_and_save_raw_market_data,
    ingest_market_data,
    prepare_interim_market_data,
    prepare_processed_market_data,
    run_market_data_pipeline,
)
from portfolio_risk_model.data.providers import (
    InMemoryMarketDataProvider,
    MarketDataRequest,
)


def test_prepare_interim_market_data_returns_valid_normalized_data() -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-01",
                "2025-01-02",
            ],
            "Ticker": [
                " SPY ",
                " SPY ",
            ],
            "Adj Close": [
                "100.5",
                "101.25",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    result = prepare_interim_market_data(
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


def test_prepare_interim_market_data_rejects_null_required_values() -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-01",
                "2025-01-02",
            ],
            "Ticker": [
                "SPY",
                None,
            ],
            "Adj Close": [
                "100.5",
                "101.25",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    with pytest.raises(ValueError):
        prepare_interim_market_data(
            df,
            column_mapping=column_mapping,
        )


def test_prepare_interim_market_data_rejects_duplicate_primary_keys() -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-01",
                "2025-01-01",
            ],
            "Ticker": [
                "SPY",
                "SPY",
            ],
            "Adj Close": [
                "100.0",
                "101.0",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    with pytest.raises(ValueError):
        prepare_interim_market_data(
            df,
            column_mapping=column_mapping,
        )


def test_prepare_interim_market_data_returns_sorted_rows() -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-02",
                "2025-01-01",
                "2025-01-01",
            ],
            "Ticker": [
                "SPY",
                "SPY",
                "GLD",
            ],
            "Adj Close": [
                "102.0",
                "101.0",
                "200.0",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    result = prepare_interim_market_data(
        df,
        column_mapping=column_mapping,
    )

    expected = [
        ("2025-01-01", "GLD"),
        ("2025-01-01", "SPY"),
        ("2025-01-02", "SPY"),
    ]

    actual = [
        (date.strftime("%Y-%m-%d"), asset_id)
        for date, asset_id in zip(
            result["date"],
            result["asset_id"],
            strict=True,
        )
    ]

    assert actual == expected


def test_prepare_interim_market_data_rejects_non_positive_prices() -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-01",
                "2025-01-02",
            ],
            "Ticker": [
                "SPY",
                "SPY",
            ],
            "Adj Close": [
                "100.0",
                "0.0",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    with pytest.raises(ValueError):
        prepare_interim_market_data(
            df,
            column_mapping=column_mapping,
        )


def test_prepare_interim_market_data_rejects_blank_asset_id() -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-01",
                "2025-01-02",
            ],
            "Ticker": [
                "SPY",
                "   ",
            ],
            "Adj Close": [
                "100.0",
                "101.0",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    with pytest.raises(ValueError):
        prepare_interim_market_data(
            df,
            column_mapping=column_mapping,
        )


def test_build_interim_market_data_saves_parquet(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-02",
                "2025-01-01",
            ],
            "Ticker": [
                "SPY",
                "SPY",
            ],
            "Adj Close": [
                "101.0",
                "100.0",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    output_path = tmp_path / "interim" / "market_data.parquet"

    result = build_interim_market_data(
        df,
        column_mapping=column_mapping,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(result) == 2


def test_build_interim_market_data_saves_expected_data(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-02",
                "2025-01-01",
            ],
            "Ticker": [
                " SPY ",
                " SPY ",
            ],
            "Adj Close": [
                "101.0",
                "100.0",
            ],
        }
    )

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    output_path = tmp_path / "market_data.parquet"

    result = build_interim_market_data(
        df,
        column_mapping=column_mapping,
        output_path=output_path,
    )

    saved = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(
        saved,
        result,
    )


def test_ingest_market_data_uses_provider() -> None:
    source_data = pd.DataFrame(
        {
            "date": pd.Series(
                [
                    "2024-01-01",
                    "2024-01-02",
                ],
                dtype="datetime64[ns]",
            ),
            "asset_id": pd.Series(
                [
                    "equity",
                    "equity",
                ],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [
                    100.0,
                    101.0,
                ],
                dtype="float64",
            ),
        }
    )

    provider = InMemoryMarketDataProvider(source_data)

    requests = [
        MarketDataRequest(
            asset_id="equity",
            symbol="TEST_EQUITY",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
        )
    ]

    result = ingest_market_data(
        provider=provider,
        requests=requests,
    )

    expected = pd.DataFrame(
        {
            "date": pd.Series(
                [
                    "2024-01-01",
                    "2024-01-02",
                ],
                dtype="datetime64[ns]",
            ),
            "asset_id": pd.Series(
                [
                    "equity",
                    "equity",
                ],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [
                    100.0,
                    101.0,
                ],
                dtype="float64",
            ),
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_ingest_and_save_raw_market_data_creates_raw_file(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "Date": [
                "2025-01-02",
                "2025-01-01",
            ],
            "Ticker": [
                " SPY ",
                "SPY",
            ],
            "Adj Close": [
                "101.0",
                "100.0",
            ],
        }
    )

    provider = InMemoryMarketDataProvider(df)

    output_path = tmp_path / "raw" / "market_data.parquet"

    result = ingest_and_save_raw_market_data(
        provider=provider,
        asset_ids=["SPY"],
        output_path=output_path,
    )

    assert output_path.exists()

    pd.testing.assert_frame_equal(
        result,
        df,
    )


def test_raw_market_data_is_saved_without_normalization(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "Ticker": [" SPY "],
            "Adj Close": ["100.0"],
        }
    )

    provider = InMemoryMarketDataProvider(df)

    output_path = tmp_path / "raw.parquet"

    ingest_and_save_raw_market_data(
        provider=provider,
        asset_ids=["SPY"],
        output_path=output_path,
    )

    saved = pd.read_parquet(output_path)

    assert saved.columns.tolist() == [
        "Date",
        "Ticker",
        "Adj Close",
    ]

    assert saved.loc[0, "Ticker"] == " SPY "


def test_prepare_processed_market_data_adds_return_column() -> None:
    interim_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ).as_unit("ns"),
            "asset_id": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [100.0, 102.0],
                dtype="float64",
            ),
        }
    )

    result = prepare_processed_market_data(
        interim_df,
    )

    assert "return" in result.columns
    assert result.loc[1, "return"] == pytest.approx(0.02)


def test_build_processed_market_data_saves_parquet(
    tmp_path: Path,
) -> None:
    interim_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ).as_unit("ns"),
            "asset_id": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [100.0, 102.0],
                dtype="float64",
            ),
        }
    )

    output_path = tmp_path / "processed" / "market_data.parquet"

    result = build_processed_market_data(
        interim_df,
        output_path=output_path,
    )

    assert output_path.exists()
    assert "return" in result.columns
    assert result.loc[1, "return"] == pytest.approx(0.02)


def test_build_processed_market_data_saves_expected_data(
    tmp_path: Path,
) -> None:
    interim_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ).as_unit("ns"),
            "asset_id": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [100.0, 102.0],
                dtype="float64",
            ),
        }
    )

    output_path = tmp_path / "processed.parquet"

    result = build_processed_market_data(
        interim_df,
        output_path=output_path,
    )

    saved = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(
        saved,
        result,
    )


def test_run_market_data_pipeline_creates_all_layers(
    tmp_path: Path,
) -> None:
    raw_df = pd.DataFrame(
        {
            "Date": [
                "2025-01-02",
                "2025-01-01",
            ],
            "Ticker": [
                " SPY ",
                " SPY ",
            ],
            "Adj Close": [
                "102.0",
                "100.0",
            ],
        }
    )

    provider = InMemoryMarketDataProvider(raw_df)

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    raw_path = tmp_path / "raw" / "market_data.parquet"
    interim_path = tmp_path / "interim" / "market_data.parquet"
    processed_path = tmp_path / "processed" / "market_data.parquet"

    result = run_market_data_pipeline(
        provider=provider,
        asset_ids=["SPY"],
        column_mapping=column_mapping,
        raw_output_path=raw_path,
        interim_output_path=interim_path,
        processed_output_path=processed_path,
    )

    assert raw_path.exists()
    assert interim_path.exists()
    assert processed_path.exists()

    assert "return" in result.columns


def test_run_market_data_pipeline_preserves_layer_semantics(
    tmp_path: Path,
) -> None:
    raw_df = pd.DataFrame(
        {
            "Date": [
                "2025-01-02",
                "2025-01-01",
            ],
            "Ticker": [
                " SPY ",
                " SPY ",
            ],
            "Adj Close": [
                "102.0",
                "100.0",
            ],
        }
    )

    provider = InMemoryMarketDataProvider(raw_df)

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    raw_path = tmp_path / "raw.parquet"
    interim_path = tmp_path / "interim.parquet"
    processed_path = tmp_path / "processed.parquet"

    run_market_data_pipeline(
        provider=provider,
        asset_ids=["SPY"],
        column_mapping=column_mapping,
        raw_output_path=raw_path,
        interim_output_path=interim_path,
        processed_output_path=processed_path,
    )

    raw = pd.read_parquet(raw_path)
    interim = pd.read_parquet(interim_path)
    processed = pd.read_parquet(processed_path)

    assert raw.columns.tolist() == [
        "Date",
        "Ticker",
        "Adj Close",
    ]

    assert interim.columns.tolist() == [
        "date",
        "asset_id",
        "adjusted_close",
    ]

    assert processed.columns.tolist() == [
        "date",
        "asset_id",
        "adjusted_close",
        "return",
    ]


def test_run_market_data_pipeline_calculates_expected_return(
    tmp_path: Path,
) -> None:
    raw_df = pd.DataFrame(
        {
            "Date": [
                "2025-01-02",
                "2025-01-01",
            ],
            "Ticker": [
                "SPY",
                "SPY",
            ],
            "Adj Close": [
                "102.0",
                "100.0",
            ],
        }
    )

    provider = InMemoryMarketDataProvider(raw_df)

    column_mapping = {
        "Date": "date",
        "Ticker": "asset_id",
        "Adj Close": "adjusted_close",
    }

    result = run_market_data_pipeline(
        provider=provider,
        asset_ids=["SPY"],
        column_mapping=column_mapping,
        raw_output_path=tmp_path / "raw.parquet",
        interim_output_path=tmp_path / "interim.parquet",
        processed_output_path=tmp_path / "processed.parquet",
    )

    assert result.loc[0, "adjusted_close"] == 100.0
    assert result.loc[1, "adjusted_close"] == 102.0

    assert pd.isna(result.loc[0, "return"])
    assert result.loc[1, "return"] == pytest.approx(0.02)


def test_fetch_market_data_combines_requests() -> None:
    source_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "asset_id": [
                "equity",
                "equity",
                "gold",
                "gold",
            ],
            "adjusted_close": [
                100.0,
                101.0,
                500.0,
                510.0,
            ],
        }
    )

    provider = InMemoryMarketDataProvider(source_data)

    requests = [
        MarketDataRequest(
            asset_id="equity",
            symbol="TEST_EQUITY",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
        ),
        MarketDataRequest(
            asset_id="gold",
            symbol="TEST_GOLD",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
        ),
    ]

    result = fetch_market_data(
        provider=provider,
        requests=requests,
    )

    assert len(result) == 4

    assert set(result["asset_id"]) == {
        "equity",
        "gold",
    }

    assert result["adjusted_close"].tolist() == [
        100.0,
        500.0,
        101.0,
        510.0,
    ]
