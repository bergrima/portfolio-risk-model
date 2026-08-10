import numpy as np
import pandas as pd

from portfolio_risk_model.data.contracts import LogicalDType, TableContract


def validate_required_columns(
    df: pd.DataFrame,
    contract: TableContract,
) -> None:
    required_columns = {column.name for column in contract.columns}
    actual_columns = set(df.columns)

    missing_columns = required_columns - actual_columns

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")


def validate_column_dtypes(
    df: pd.DataFrame,
    contract: TableContract,
) -> None:
    dtype_checks = {
        LogicalDType.DATETIME: pd.api.types.is_datetime64_any_dtype,
        LogicalDType.STRING: pd.api.types.is_string_dtype,
        LogicalDType.FLOAT: pd.api.types.is_float_dtype,
        LogicalDType.BOOLEAN: pd.api.types.is_bool_dtype,
    }

    for column in contract.columns:
        if column.name not in df.columns:
            continue

        series = df[column.name]
        check = dtype_checks[column.dtype]

        if not check(series):
            raise ValueError(
                f"Column '{column.name}' has dtype '{series.dtype}', "
                f"expected '{column.dtype.value}'"
            )


def validate_nullability(
    df: pd.DataFrame,
    contract: TableContract,
) -> None:
    for column in contract.columns:
        if column.name not in df.columns:
            continue

        if column.nullable:
            continue

        missing_count = df[column.name].isna().sum()

        if missing_count > 0:
            raise ValueError(
                f"Column '{column.name}' contains {missing_count} missing values "
                "but is not nullable"
            )


def validate_primary_key_uniqueness(
    df: pd.DataFrame,
    contract: TableContract,
) -> None:
    missing_key_columns = [
        column for column in contract.primary_key if column not in df.columns
    ]

    if missing_key_columns:
        return

    duplicate_mask = df.duplicated(
        subset=list(contract.primary_key),
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count > 0:
        key = ", ".join(contract.primary_key)
        raise ValueError(
            f"Primary key ({key}) contains {duplicate_count} duplicate rows"
        )


def validate_sort_order(
    df: pd.DataFrame,
    contract: TableContract,
) -> None:
    missing_sort_columns = [
        column for column in contract.sort_by if column not in df.columns
    ]

    if missing_sort_columns:
        return

    sort_columns = list(contract.sort_by)

    actual_order = df[sort_columns].reset_index(drop=True)

    expected_order = actual_order.sort_values(
        by=sort_columns, kind="mergesort"
    ).reset_index(drop=True)

    if not actual_order.equals(expected_order):
        sort_key = ", ".join(contract.sort_by)
        raise ValueError(f"Data is not sorted by ({sort_key})")


def validate_price_values(
    df: pd.DataFrame,
    price_columns: tuple[str, ...],
) -> None:
    for column_name in price_columns:
        if column_name not in df.columns:
            continue

        series = df[column_name]

        invalid_mask = series.notna() & ((series <= 0) | ~np.isfinite(series))

        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            raise ValueError(
                f"Column '{column_name}' contains {invalid_count} invalid price values"
            )


def validate_table(
    df: pd.DataFrame,
    contract: TableContract,
    price_columns: tuple[str, ...] = (),
) -> None:
    validate_required_columns(df, contract)
    validate_column_dtypes(df, contract)
    validate_nullability(df, contract)
    validate_primary_key_uniqueness(df, contract)
    validate_sort_order(df, contract)
    validate_price_values(df, price_columns)


def validate_positive_values(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    for column in columns:
        invalid = df[column] <= 0

        if invalid.any():
            raise ValueError(f"Column '{column}' contains non-positive values.")


def validate_non_empty_strings(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    for column in columns:
        invalid = df[column].str.len() == 0

        if invalid.any():
            raise ValueError(f"Column '{column}' contains empty strings.")
