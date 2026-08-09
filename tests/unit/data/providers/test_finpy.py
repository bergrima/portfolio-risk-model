from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from portfolio_risk_model.data.providers import (
    FinpyMarketDataProvider,
    MarketDataProviderError,
    MarketDataRequest,
)


def test_finpy_provider_returns_canonical_data() -> None:
    raw_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "Close": [101, 102, 103],
            "Final": [100, 101, 102],
            "Adj Close": [201, 202, 203],
            "Adj Final": [200, 201, 202],
        }
    )

    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    provider = FinpyMarketDataProvider()

    with patch(
        "portfolio_risk_model.data.providers.finpy.fpy.Get_Price_History",
        return_value=raw_data,
    ):
        result = provider.fetch(request)

    assert list(result.columns) == [
        "date",
        "asset_id",
        "adjusted_close",
    ]

    assert result["asset_id"].tolist() == [
        "equity",
        "equity",
        "equity",
    ]

    assert result["adjusted_close"].tolist() == [
        200.0,
        201.0,
        202.0,
    ]


def test_finpy_provider_filters_requested_date_range() -> None:
    raw_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2023-12-31",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "Adj Final": [
                90,
                100,
                101,
                102,
                110,
            ],
        }
    )

    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    provider = FinpyMarketDataProvider()

    with patch(
        "portfolio_risk_model.data.providers.finpy.fpy.Get_Price_History",
        return_value=raw_data,
    ):
        result = provider.fetch(request)

    assert result["adjusted_close"].tolist() == [
        100.0,
        101.0,
        102.0,
    ]


def test_finpy_provider_raises_on_empty_data() -> None:
    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    provider = FinpyMarketDataProvider()

    with patch(
        "portfolio_risk_model.data.providers.finpy.fpy.Get_Price_History",
        return_value=pd.DataFrame(),
    ):
        with pytest.raises(
            MarketDataProviderError,
            match="returned no data",
        ):
            provider.fetch(request)


def test_finpy_provider_wraps_download_error() -> None:
    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    provider = FinpyMarketDataProvider()

    with patch(
        "portfolio_risk_model.data.providers.finpy.fpy.Get_Price_History",
        side_effect=RuntimeError("connection failed"),
    ):
        with pytest.raises(
            MarketDataProviderError,
            match="finpy failed",
        ):
            provider.fetch(request)
