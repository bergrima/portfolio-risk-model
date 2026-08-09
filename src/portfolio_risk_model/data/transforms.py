import pandas as pd


def add_simple_returns(
    df: pd.DataFrame,
    price_column: str = "adjusted_close",
    asset_column: str = "asset_id",
    return_column: str = "return",
) -> pd.DataFrame:
    result = df.copy()

    result[return_column] = result.groupby(asset_column)[price_column].pct_change()

    return result
