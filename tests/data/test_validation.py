import numpy as np
import pandas as pd
import pytest

from portfolio_risk_model.data.contracts import (
    INTERIM_MARKET_DATA_CONTRACT,
    ColumnSpec,
    LogicalDType,
    TableContract,
)
from portfolio_risk_model.data.validation import (
    validate_column_dtypes,
    validate_non_empty_strings,
    validate_nullability,
    validate_positive_values,
    validate_price_values,
    validate_primary_key_uniqueness,
    validate_required_columns,
    validate_sort_order,
    validate_table,
)


@pytest.fixture
def price_contract() -> TableContract:
    return TableContract(
        name="prices",
        columns=(
            ColumnSpec("date", LogicalDType.DATETIME, False),
            ColumnSpec("asset", LogicalDType.STRING, False),
            ColumnSpec("adjusted_close", LogicalDType.FLOAT, False),
        ),
        primary_key=("date", "asset"),
        sort_by=("asset", "date"),
    )


def test_required_columns_pass_when_all_columns_exist(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "asset": ["SPY"],
            "adjusted_close": [100.0],
        }
    )

    validate_required_columns(df, price_contract)


def test_required_columns_raise_when_column_is_missing(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "asset": ["SPY"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns: adjusted_close",
    ):
        validate_required_columns(df, price_contract)


def test_column_dtypes_pass_when_types_are_correct(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "asset": pd.Series(["SPY"], dtype="string"),
            "adjusted_close": [100.0],
        }
    )

    validate_column_dtypes(df, price_contract)


def test_column_dtypes_raise_when_type_is_wrong(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "asset": pd.Series(["SPY"], dtype="string"),
            "adjusted_close": ["100.0"],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Column 'adjusted_close'.*expected 'float64'",
    ):
        validate_column_dtypes(df, price_contract)


def test_nullability_pass_when_non_nullable_columns_have_no_missing_values(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "asset": pd.Series(["SPY", "SPY"], dtype="string"),
            "adjusted_close": [100.0, 101.0],
        }
    )

    validate_nullability(df, price_contract)


def test_nullability_raise_when_non_nullable_column_has_missing_value(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "asset": pd.Series(["SPY", "SPY"], dtype="string"),
            "adjusted_close": [100.0, None],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Column 'adjusted_close' contains 1 missing values",
    ):
        validate_nullability(df, price_contract)


def test_nullability_allows_missing_values_when_column_is_nullable() -> None:
    contract = TableContract(
        name="prices",
        columns=(
            ColumnSpec("date", LogicalDType.DATETIME, False),
            ColumnSpec("volume", LogicalDType.FLOAT, True),
        ),
        primary_key=("date",),
        sort_by=("date",),
    )

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "volume": [1000.0, None],
        }
    )

    validate_nullability(df, contract)


def test_primary_key_pass_when_rows_are_unique(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "asset": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": [100.0, 101.0],
        }
    )

    validate_primary_key_uniqueness(df, price_contract)


def test_primary_key_raise_when_duplicate_exists(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "asset": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": [100.0, 101.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Primary key \(date, asset\) contains 2 duplicate rows",
    ):
        validate_primary_key_uniqueness(df, price_contract)


def test_primary_key_allows_same_date_for_different_assets(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "asset": pd.Series(
                ["SPY", "GLD"],
                dtype="string",
            ),
            "adjusted_close": [100.0, 200.0],
        }
    )

    validate_primary_key_uniqueness(df, price_contract)


def test_sort_order_pass_when_data_is_sorted(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "asset": pd.Series(
                ["GLD", "GLD", "SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": [
                200.0,
                201.0,
                100.0,
                101.0,
            ],
        }
    )

    validate_sort_order(df, price_contract)


def test_sort_order_raise_when_dates_are_unsorted(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-01",
                ]
            ),
            "asset": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": [
                101.0,
                100.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Data is not sorted by \(asset, date\)",
    ):
        validate_sort_order(df, price_contract)


def test_sort_order_raise_when_assets_are_unsorted(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-01",
                ]
            ),
            "asset": pd.Series(
                ["SPY", "GLD"],
                dtype="string",
            ),
            "adjusted_close": [
                100.0,
                200.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Data is not sorted by \(asset, date\)",
    ):
        validate_sort_order(df, price_contract)


def test_price_values_pass_when_prices_are_valid() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": [
                100.0,
                101.5,
                99.25,
            ]
        }
    )

    validate_price_values(
        df,
        price_columns=("adjusted_close",),
    )


def test_price_values_raise_when_price_is_negative() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": [
                100.0,
                -10.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Column 'adjusted_close' contains 1 invalid price values",
    ):
        validate_price_values(
            df,
            price_columns=("adjusted_close",),
        )


def test_price_values_raise_when_price_is_infinite() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": [
                100.0,
                np.inf,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Column 'adjusted_close' contains 1 invalid price values",
    ):
        validate_price_values(
            df,
            price_columns=("adjusted_close",),
        )


def test_price_values_ignore_missing_values() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": [
                100.0,
                np.nan,
            ]
        }
    )

    validate_price_values(
        df,
        price_columns=("adjusted_close",),
    )


def test_validate_table_pass_when_all_checks_pass(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "asset": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": [
                100.0,
                101.0,
            ],
        }
    )

    validate_table(
        df,
        price_contract,
        price_columns=("adjusted_close",),
    )


def test_validate_table_raise_when_price_is_invalid(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "asset": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": [
                100.0,
                -10.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Column 'adjusted_close' contains 1 invalid price values",
    ):
        validate_table(
            df,
            price_contract,
            price_columns=("adjusted_close",),
        )


def test_validate_table_fails_first_on_missing_column(
    price_contract: TableContract,
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "asset": pd.Series(["SPY"], dtype="string"),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Missing required columns: adjusted_close",
    ):
        validate_table(
            df,
            price_contract,
            price_columns=("adjusted_close",),
        )


def test_primary_key_uniqueness_passes_when_keys_are_unique() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ).as_unit("ns"),
            "asset_id": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [100.0, 101.0],
                dtype="float64",
            ),
        }
    )

    validate_primary_key_uniqueness(
        df,
        INTERIM_MARKET_DATA_CONTRACT,
    )


def test_primary_key_uniqueness_raises_for_duplicate_keys() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-01",
                ]
            ).as_unit("ns"),
            "asset_id": pd.Series(
                ["SPY", "SPY"],
                dtype="string",
            ),
            "adjusted_close": pd.Series(
                [100.0, 101.0],
                dtype="float64",
            ),
        }
    )

    with pytest.raises(ValueError):
        validate_primary_key_uniqueness(
            df,
            INTERIM_MARKET_DATA_CONTRACT,
        )


def test_positive_values_pass_when_all_values_are_positive() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": pd.Series(
                [100.0, 101.5, 99.25],
                dtype="float64",
            )
        }
    )

    validate_positive_values(
        df,
        columns=("adjusted_close",),
    )


def test_positive_values_raise_when_zero_exists() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": pd.Series(
                [100.0, 0.0, 101.0],
                dtype="float64",
            )
        }
    )

    with pytest.raises(ValueError):
        validate_positive_values(
            df,
            columns=("adjusted_close",),
        )


def test_positive_values_raise_when_negative_exists() -> None:
    df = pd.DataFrame(
        {
            "adjusted_close": pd.Series(
                [100.0, -5.0, 101.0],
                dtype="float64",
            )
        }
    )

    with pytest.raises(ValueError):
        validate_positive_values(
            df,
            columns=("adjusted_close",),
        )


def test_non_empty_strings_pass_when_values_are_valid() -> None:
    df = pd.DataFrame(
        {
            "asset_id": pd.Series(
                ["SPY", "GLD", "TLT"],
                dtype="string",
            )
        }
    )

    validate_non_empty_strings(
        df,
        columns=("asset_id",),
    )


def test_non_empty_strings_raise_when_empty_string_exists() -> None:
    df = pd.DataFrame(
        {
            "asset_id": pd.Series(
                ["SPY", "", "GLD"],
                dtype="string",
            )
        }
    )

    with pytest.raises(ValueError):
        validate_non_empty_strings(
            df,
            columns=("asset_id",),
        )
