from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    def fetch(
        self,
        asset_ids: list[str],
    ) -> pd.DataFrame: ...


class InMemoryMarketDataProvider:
    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self._data = data.copy()

    def fetch(
        self,
        asset_ids: list[str],
    ) -> pd.DataFrame:
        return self._data.copy()
