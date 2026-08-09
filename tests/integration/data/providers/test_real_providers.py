import os
from datetime import date

import pandas as pd
import pytest

from portfolio_risk_model.data.providers import (
    FinpyMarketDataProvider,
    MarketDataRequest,
    PytseMarketDataProvider,
)

RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set RUN_INTEGRATION_TESTS=1 to run real market-data tests"),
    ),
]


def make_test_request() -> MarketDataRequest:
    return MarketDataRequest(
        asset_id="foolad",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
    )


def assert_valid_market_data(
    data: pd.DataFrame,
) -> None:
    assert not data.empty

    assert list(data.columns) == [
        "date",
        "asset_id",
        "adjusted_close",
    ]

    assert data["date"].dtype == "datetime64[ns]"

    assert data["asset_id"].dtype.name == "string"

    assert data["adjusted_close"].dtype == "float64"

    assert data["date"].is_monotonic_increasing

    assert not data[["date", "asset_id"]].duplicated().any()

    assert data["date"].min() >= pd.Timestamp("2024-01-01")

    assert data["date"].max() <= pd.Timestamp("2024-03-31")

    assert (data["adjusted_close"] > 0).all()


def test_real_pytse_provider() -> None:
    provider = PytseMarketDataProvider()

    data = provider.fetch(make_test_request())

    assert_valid_market_data(data)


def test_real_finpy_provider() -> None:
    provider = FinpyMarketDataProvider()

    data = provider.fetch(make_test_request())

    assert_valid_market_data(data)
