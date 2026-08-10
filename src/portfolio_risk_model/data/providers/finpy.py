import finpy_tse as fpy
import pandas as pd

from portfolio_risk_model.data.providers.base import (
    MarketDataProviderError,
    MarketDataRequest,
)


class FinpyMarketDataProvider:
    def fetch(
        self,
        request: MarketDataRequest,
    ) -> pd.DataFrame:
        try:
            raw = fpy.Get_Price_History(
                stock=request.symbol,
                ignore_date=True,
                adjust_price=True,
                show_weekday=False,
                double_date=True,
            )
        except Exception as exc:
            raise MarketDataProviderError(
                f"finpy failed for symbol {request.symbol!r}"
            ) from exc

        if raw is None or raw.empty:
            raise MarketDataProviderError(
                f"finpy returned no data for symbol {request.symbol!r}"
            )

        required_columns = {
            "Date",
            "Adj Final",
        }

        missing_columns = required_columns - set(raw.columns)

        if missing_columns:
            raise MarketDataProviderError(
                f"finpy response is missing columns: {sorted(missing_columns)}"
            )

        data = raw[
            [
                "Date",
                "Adj Final",
            ]
        ].copy()

        data = data.rename(
            columns={
                "Date": "date",
                "Adj Final": "adjusted_close",
            }
        )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="raise",
        ).astype("datetime64[ns]")

        data["adjusted_close"] = pd.to_numeric(
            data["adjusted_close"],
            errors="raise",
        ).astype("float64")

        start = pd.Timestamp(request.start_date)
        end = pd.Timestamp(request.end_date)

        data = data[
            data["date"].between(
                start,
                end,
                inclusive="both",
            )
        ].copy()

        data["asset_id"] = request.asset_id

        data["asset_id"] = data["asset_id"].astype("string")

        data = data[
            [
                "date",
                "asset_id",
                "adjusted_close",
            ]
        ]

        data = data.sort_values(["date", "asset_id"]).reset_index(drop=True)

        return data
