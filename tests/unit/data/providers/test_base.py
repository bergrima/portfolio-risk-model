from datetime import date

import pytest

from portfolio_risk_model.data.providers import MarketDataRequest


def test_market_data_request_accepts_valid_dates() -> None:
    request = MarketDataRequest(
        asset_id="equity",
        symbol="فولاد",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )

    assert request.asset_id == "equity"
    assert request.symbol == "فولاد"
    assert request.start_date == date(2024, 1, 1)
    assert request.end_date == date(2024, 12, 31)


def test_market_data_request_rejects_empty_asset_id() -> None:
    with pytest.raises(ValueError, match="asset_id"):
        MarketDataRequest(
            asset_id="",
            symbol="فولاد",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )


def test_market_data_request_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        MarketDataRequest(
            asset_id="equity",
            symbol="",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )


def test_market_data_request_rejects_reversed_dates() -> None:
    with pytest.raises(
        ValueError,
        match="start_date must be on or before end_date",
    ):
        MarketDataRequest(
            asset_id="equity",
            symbol="فولاد",
            start_date=date(2024, 12, 31),
            end_date=date(2024, 1, 1),
        )
