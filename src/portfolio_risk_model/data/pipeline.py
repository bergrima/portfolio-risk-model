from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from portfolio_risk_model.data.contracts import (
    INTERIM_MARKET_DATA_CONTRACT,
    PROCESSED_MARKET_DATA_CONTRACT,
)
from portfolio_risk_model.data.normalization import normalize_market_data, sort_rows
from portfolio_risk_model.data.providers import MarketDataProvider, MarketDataRequest
from portfolio_risk_model.data.storage import save_dataframe
from portfolio_risk_model.data.transforms import (
    add_simple_returns,
)
from portfolio_risk_model.data.validation import (
    validate_column_dtypes,
    validate_non_empty_strings,
    validate_nullability,
    validate_positive_values,
    validate_primary_key_uniqueness,
    validate_required_columns,
    validate_table,
)


def prepare_interim_market_data(
    df: pd.DataFrame,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    result = normalize_market_data(
        df,
        column_mapping=column_mapping,
    )

    validate_required_columns(
        result,
        INTERIM_MARKET_DATA_CONTRACT,
    )

    validate_column_dtypes(
        result,
        INTERIM_MARKET_DATA_CONTRACT,
    )

    validate_nullability(
        result,
        INTERIM_MARKET_DATA_CONTRACT,
    )

    validate_primary_key_uniqueness(
        result,
        INTERIM_MARKET_DATA_CONTRACT,
    )

    validate_positive_values(
        result,
        columns=("adjusted_close",),
    )

    validate_non_empty_strings(
        result,
        columns=("asset_id",),
    )

    result = sort_rows(
        result,
        columns=INTERIM_MARKET_DATA_CONTRACT.sort_by,
    )

    return result


def build_interim_market_data(
    df: pd.DataFrame,
    column_mapping: dict[str, str],
    output_path: Path,
) -> pd.DataFrame:
    result = prepare_interim_market_data(
        df,
        column_mapping=column_mapping,
    )

    save_dataframe(
        result,
        output_path,
    )

    return result


def ingest_market_data(
    provider: MarketDataProvider,
    requests: Sequence[MarketDataRequest] | None = None,
    *,
    asset_ids: list[str] | None = None,
) -> pd.DataFrame:
    # New provider interface.
    if requests is not None:
        return fetch_market_data(
            provider=provider,
            requests=requests,
        )

    # Temporary compatibility with the existing pipeline.
    if asset_ids is not None:
        return provider.fetch(asset_ids=asset_ids)  # type: ignore[call-arg]

    raise ValueError("Either requests or asset_ids must be provided")


def ingest_and_save_raw_market_data(
    provider: MarketDataProvider,
    asset_ids: list[str],
    output_path: Path,
) -> pd.DataFrame:
    raw_data = ingest_market_data(
        provider=provider,
        asset_ids=asset_ids,
    )

    save_dataframe(
        raw_data,
        output_path,
    )

    return raw_data


def prepare_processed_market_data(
    interim_df: pd.DataFrame,
) -> pd.DataFrame:
    result = add_simple_returns(
        interim_df,
        price_column="adjusted_close",
        asset_column="asset_id",
        return_column="return",
    )

    validate_table(
        result,
        PROCESSED_MARKET_DATA_CONTRACT,
    )

    return result


def build_processed_market_data(
    interim_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    result = prepare_processed_market_data(
        interim_df,
    )

    save_dataframe(
        result,
        output_path,
    )

    return result


def run_market_data_pipeline(
    provider: MarketDataProvider,
    asset_ids: list[str],
    column_mapping: dict[str, str],
    raw_output_path: Path,
    interim_output_path: Path,
    processed_output_path: Path,
) -> pd.DataFrame:
    raw_data = ingest_and_save_raw_market_data(
        provider=provider,
        asset_ids=asset_ids,
        output_path=raw_output_path,
    )

    interim_data = build_interim_market_data(
        raw_data,
        column_mapping=column_mapping,
        output_path=interim_output_path,
    )

    processed_data = build_processed_market_data(
        interim_data,
        output_path=processed_output_path,
    )

    return processed_data


def fetch_market_data(
    provider: MarketDataProvider,
    requests: Sequence[MarketDataRequest],
) -> pd.DataFrame:
    if not requests:
        raise ValueError("At least one market data request is required")

    frames = [provider.fetch(request) for request in requests]

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    return data.sort_values(["date", "asset_id"]).reset_index(drop=True)
