from datetime import date

from portfolio_risk_model.data.universe import (
    MVP_UNIVERSE,
    build_market_data_requests,
)


def test_mvp_universe_contains_expected_assets() -> None:
    asset_ids = {asset.asset_id for asset in MVP_UNIVERSE}

    assert asset_ids == {
        "gold",
        "equity",
        "fixed_income",
    }


def test_mvp_universe_contains_expected_symbols() -> None:
    symbols = {asset.symbol for asset in MVP_UNIVERSE}

    assert symbols == {
        "طلا",
        "آگاس",
        "اعتماد",
    }


def test_build_market_data_requests() -> None:
    start_date = date(2017, 1, 1)
    end_date = date(2026, 8, 10)

    requests = build_market_data_requests(
        assets=MVP_UNIVERSE,
        start_date=start_date,
        end_date=end_date,
    )

    assert len(requests) == 3

    assert {request.asset_id for request in requests} == {
        "gold",
        "equity",
        "fixed_income",
    }

    assert all(request.start_date == start_date for request in requests)

    assert all(request.end_date == end_date for request in requests)
