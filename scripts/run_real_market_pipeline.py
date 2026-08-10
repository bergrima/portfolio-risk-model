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
            asset_id="foolad",
            symbol="فولاد",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        )
    ]

    column_mapping = {
        "date": "date",
        "asset_id": "asset_id",
        "adjusted_close": "adjusted_close",
    }

    processed_data = run_market_data_pipeline(
        provider=provider,
        requests=requests,
        column_mapping=column_mapping,
        raw_output_path=Path("data/raw/foolad.parquet"),
        interim_output_path=Path("data/interim/foolad.parquet"),
        processed_output_path=Path("data/processed/foolad.parquet"),
    )

    print(processed_data.head())
    print()
    print(processed_data.tail())
    print()
    print(f"Rows: {len(processed_data)}")
    print(f"Start: {processed_data['date'].min()}")
    print(f"End: {processed_data['date'].max()}")


if __name__ == "__main__":
    main()
