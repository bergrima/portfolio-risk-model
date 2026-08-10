from datetime import date
from pathlib import Path

from portfolio_risk_model.data.pipeline import run_market_data_pipeline
from portfolio_risk_model.data.providers import (
    MarketDataRequest,
    PytseMarketDataProvider,
)


def main() -> None:
    provider = PytseMarketDataProvider()

    requests = [
        MarketDataRequest(
            asset_id="gold",
            symbol="طلا",
            start_date=date(2010, 1, 1),
            end_date=date.today(),
        ),
        MarketDataRequest(
            asset_id="equity",
            symbol="آگاس",
            start_date=date(2010, 1, 1),
            end_date=date.today(),
        ),
        MarketDataRequest(
            asset_id="fixed_income",
            symbol="لبخند",
            start_date=date(2010, 1, 1),
            end_date=date.today(),
        ),
    ]

    result = run_market_data_pipeline(
        provider=provider,
        requests=requests,
        column_mapping={
            "date": "date",
            "asset_id": "asset_id",
            "adjusted_close": "adjusted_close",
        },
        raw_output_path=Path("data/raw/market_data.parquet"),
        interim_output_path=Path("data/interim/market_data.parquet"),
        processed_output_path=Path("data/processed/market_data.parquet"),
    )

    print(
        result.groupby("asset_id").agg(
            rows=("date", "size"),
            start=("date", "min"),
            end=("date", "max"),
        )
    )


if __name__ == "__main__":
    main()
