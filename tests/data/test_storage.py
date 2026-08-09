from pathlib import Path

import pandas as pd
import pytest

from portfolio_risk_model.data.storage import load_dataframe, save_dataframe


def test_save_dataframe_creates_parquet_file(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "asset_id": ["SPY", "GLD"],
            "adjusted_close": [100.0, 200.0],
        }
    )

    path = tmp_path / "interim" / "market_data.parquet"

    save_dataframe(
        df,
        path,
    )

    assert path.exists()


def test_saved_dataframe_can_be_loaded_back(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "asset_id": pd.Series(
                ["SPY", "GLD"],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [100.0, 200.0],
                dtype="float64",
            ),
        }
    )

    path = tmp_path / "market_data.parquet"

    save_dataframe(
        df,
        path,
    )

    result = pd.read_parquet(path)

    pd.testing.assert_frame_equal(
        result,
        df,
    )


def test_load_dataframe_reads_saved_parquet(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "asset_id": pd.Series(
                ["SPY", "GLD"],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [100.0, 200.0],
                dtype="float64",
            ),
        }
    )

    path = tmp_path / "market_data.parquet"

    save_dataframe(
        df,
        path,
    )

    result = load_dataframe(path)

    pd.testing.assert_frame_equal(
        result,
        df,
    )


def test_load_dataframe_raises_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        load_dataframe(path)
