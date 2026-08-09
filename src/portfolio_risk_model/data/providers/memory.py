import pandas as pd

from portfolio_risk_model.data.providers.base import MarketDataRequest


class InMemoryMarketDataProvider:
    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data.copy()

    def fetch(
        self,
        request: MarketDataRequest | None = None,
        *,
        asset_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        data = self._data.copy()

        # Legacy interface used by the current pipeline.
        if request is None:
            if asset_ids is None:
                raise ValueError("Either request or asset_ids must be provided")

            return data.copy()

        # New interface used by the new provider architecture.
        data["date"] = pd.to_datetime(
            data["date"],
            errors="raise",
        )

        start = pd.Timestamp(request.start_date)
        end = pd.Timestamp(request.end_date)

        mask = (data["asset_id"] == request.asset_id) & data["date"].between(
            start,
            end,
            inclusive="both",
        )

        result = data.loc[mask].copy()
        result["date"] = result["date"].astype("datetime64[ns]")

        result["asset_id"] = result["asset_id"].astype("string")

        result["adjusted_close"] = pd.to_numeric(
            result["adjusted_close"],
            errors="raise",
        ).astype("float64")

        return result.sort_values(["date", "asset_id"]).reset_index(drop=True)
