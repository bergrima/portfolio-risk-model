import pandas as pd


def normalize_column_names(
    df: pd.DataFrame,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    normalized = df.rename(columns=column_mapping).copy()

    if normalized.columns.duplicated().any():
        duplicate_columns = normalized.columns[
            normalized.columns.duplicated(keep=False)
        ].unique()

        duplicates = ", ".join(sorted(duplicate_columns))

        raise ValueError(
            f"Column normalization created duplicate columns: {duplicates}"
        )

    return normalized


def normalize_date_column(
    df: pd.DataFrame,
    column: str = "date",
) -> pd.DataFrame:
    result = df.copy()

    result[column] = pd.to_datetime(
        result[column],
        errors="raise",
    ).dt.as_unit("ns")

    return result


def normalize_numeric_column(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    result = df.copy()

    result[column] = pd.to_numeric(
        result[column],
        errors="raise",
    ).astype("float64")

    return result


def normalize_text_column(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    result = df.copy()

    result[column] = result[column].astype("string").str.strip()

    return result


def normalize_market_data(
    df: pd.DataFrame,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    result = normalize_column_names(
        df,
        column_mapping=column_mapping,
    )

    result = normalize_date_column(
        result,
        column="date",
    )

    result = normalize_numeric_column(
        result,
        column="adjusted_close",
    )

    result = normalize_text_column(
        result,
        column="asset_id",
    )

    return result


def sort_rows(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    result = df.copy()

    result = result.sort_values(by=list(columns)).reset_index(drop=True)

    return result
