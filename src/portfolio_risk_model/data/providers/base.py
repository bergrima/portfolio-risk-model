from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd


@dataclass(frozen=True, slots=True)
class MarketDataRequest:
    asset_id: str
    symbol: str
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must not be empty")

        if not self.symbol:
            raise ValueError("symbol must not be empty")

        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")


class MarketDataProvider(Protocol):
    def fetch(
        self,
        request: MarketDataRequest,
    ) -> pd.DataFrame: ...


class MarketDataProviderError(RuntimeError):
    """Raised when a provider cannot return usable market data."""
