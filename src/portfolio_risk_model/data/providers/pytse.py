import pandas as pd
import pytse_client as tse

from portfolio_risk_model.data.providers.base import (
    MarketDataProviderError,
    MarketDataRequest,
)


class PytseMarketDataProvider:
    def fetch(
        self,
        request: MarketDataRequest,
    ) -> pd.DataFrame:
        try:
            downloaded = tse.download(
                symbols=request.symbol,
                write_to_csv=False,
                include_jdate=False,
                adjust=True,
            )
        except Exception as exc:
            raise MarketDataProviderError(
                f"pytse failed for symbol {request.symbol!r}"
            ) from exc

        if request.symbol not in downloaded:
            raise MarketDataProviderError(
                f"pytse returned no data for symbol {request.symbol!r}"
            )

        raw = downloaded[request.symbol]

        if raw.empty:
            raise MarketDataProviderError(
                f"pytse returned an empty dataset for {request.symbol!r}"
            )

        required_columns = {
            "date",
            "adjClose",
        }

        missing_columns = required_columns - set(raw.columns)

        if missing_columns:
            raise MarketDataProviderError(
                f"pytse response is missing columns: {sorted(missing_columns)}"
            )

        data = raw[
            [
                "date",
                "adjClose",
            ]
        ].copy()

        data = data.rename(
            columns={
                "adjClose": "adjusted_close",
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
