import pandas as pd

from portfolio_risk_model.data.providers import (
    InMemoryMarketDataProvider,
)


def test_in_memory_provider_returns_dataframe() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "Ticker": ["SPY"],
            "Adj Close": [100.0],
        }
    )

    provider = InMemoryMarketDataProvider(df)

    result = provider.fetch(
        asset_ids=["SPY"],
    )

    pd.testing.assert_frame_equal(
        result,
        df,
    )


def test_in_memory_provider_returns_copy() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "Ticker": ["SPY"],
            "Adj Close": [100.0],
        }
    )

    provider = InMemoryMarketDataProvider(df)

    result = provider.fetch(
        asset_ids=["SPY"],
    )

    result.loc[0, "Adj Close"] = 999.0

    second_result = provider.fetch(
        asset_ids=["SPY"],
    )

    assert second_result.loc[0, "Adj Close"] == 100.0
