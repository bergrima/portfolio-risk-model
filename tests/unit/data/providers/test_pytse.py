from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from portfolio_risk_model.data.providers import (
    MarketDataProviderError,
    MarketDataRequest,
    PytseMarketDataProvider,
)


def test_pytse_provider_returns_canonical_data() -> None:
    raw_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 105.0],
            "adjClose": [102.0, 103.0, 104.0],
            "volume": [1000, 2000, 3000],
        }
    )

    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    provider = PytseMarketDataProvider()

    with patch(
        "portfolio_risk_model.data.providers.pytse.tse.download",
        return_value={"فولاد": raw_data},
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
        102.0,
        103.0,
        104.0,
    ]


def test_pytse_provider_filters_requested_date_range() -> None:
    raw_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-12-31",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "adjClose": [
                90.0,
                100.0,
                101.0,
                102.0,
                110.0,
            ],
        }
    )

    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    provider = PytseMarketDataProvider()

    with patch(
        "portfolio_risk_model.data.providers.pytse.tse.download",
        return_value={"فولاد": raw_data},
    ):
        result = provider.fetch(request)

    assert result["date"].tolist() == list(
        pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ]
        )
    )


def test_pytse_provider_raises_when_symbol_is_missing() -> None:
    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    provider = PytseMarketDataProvider()

    with patch(
        "portfolio_risk_model.data.providers.pytse.tse.download",
        return_value={},
    ):
        with pytest.raises(
            MarketDataProviderError,
            match="returned no data",
        ):
            provider.fetch(request)


def test_pytse_provider_wraps_download_error() -> None:
    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    provider = PytseMarketDataProvider()

    with patch(
        "portfolio_risk_model.data.providers.pytse.tse.download",
        side_effect=RuntimeError("network failure"),
    ):
        with pytest.raises(
            MarketDataProviderError,
            match="pytse failed",
        ):
            provider.fetch(request)
