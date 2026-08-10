from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from portfolio_risk_model.data.providers import MarketDataRequest


@dataclass(frozen=True, slots=True)
class AssetDefinition:
    asset_id: str
    symbol: str


MVP_UNIVERSE: tuple[AssetDefinition, ...] = (
    AssetDefinition(
        asset_id="gold",
        symbol="طلا",
    ),
    AssetDefinition(
        asset_id="equity",
        symbol="آگاس",
    ),
    AssetDefinition(
        asset_id="fixed_income",
        symbol="اعتماد",
    ),
)


def build_market_data_requests(
    assets: Sequence[AssetDefinition],
    start_date: date,
    end_date: date,
) -> list[MarketDataRequest]:
    return [
        MarketDataRequest(
            asset_id=asset.asset_id,
            symbol=asset.symbol,
            start_date=start_date,
            end_date=end_date,
        )
        for asset in assets
    ]
