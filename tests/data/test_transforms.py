import pandas as pd
import pytest

from portfolio_risk_model.data.transforms import (
    add_simple_returns,
)


def test_add_simple_returns_calculates_returns_per_asset() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-01",
                    "2025-01-02",
                ]
            ).as_unit("ns"),
            "asset_id": pd.Series(
                [
                    "SPY",
                    "SPY",
                    "GLD",
                    "GLD",
                ],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [
                    100.0,
                    102.0,
                    200.0,
                    210.0,
                ],
                dtype="float64",
            ),
        }
    )

    result = add_simple_returns(df)

    assert result.loc[1, "return"] == pytest.approx(0.02)
    assert result.loc[3, "return"] == pytest.approx(0.05)


def test_add_simple_returns_first_observation_is_missing() -> None:
    df = pd.DataFrame(
        {
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

    result = add_simple_returns(df)

    assert pd.isna(result.loc[0, "return"])


def test_add_simple_returns_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
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

    original = df.copy()

    add_simple_returns(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )
