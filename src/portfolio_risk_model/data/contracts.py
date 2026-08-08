from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class DataLayer(str, Enum):
    RAW = "raw"
    INTERIM = "interim"
    PROCESSED = "processed"


class Frequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


class LogicalDType(str, Enum):
    DATETIME = "datetime64[ns]"
    STRING = "string"
    FLOAT = "float64"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    dtype: LogicalDType
    nullable: bool


@dataclass(frozen=True, slots=True)
class TableContract:
    name: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...]
    sort_by: tuple[str, ...]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        column_names = [column.name for column in self.columns]

        if len(column_names) != len(set(column_names)):
            raise ValueError(f"Duplicate columns in contract: {self.name}")

        missing_primary_key = set(self.primary_key) - set(column_names)

        if missing_primary_key:
            raise ValueError(
                f"Primary key columns missing from contract: "
                f"{sorted(missing_primary_key)}"
            )

        missing_sort_columns = set(self.sort_by) - set(column_names)

        if missing_sort_columns:
            raise ValueError(
                f"Sort columns missing from contract: "
                f"{sorted(missing_sort_columns)}"
            )

        nullable_columns = {
            column.name
            for column in self.columns
            if column.nullable
        }

        invalid_primary_key = set(self.primary_key) & nullable_columns

        if invalid_primary_key:
            raise ValueError(
                f"Primary key columns cannot be nullable: "
                f"{sorted(invalid_primary_key)}"
            )

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            column.name
            for column in self.columns
            if not column.nullable
        )


INTERIM_MARKET_DATA_CONTRACT = TableContract(
    name="interim_market_data",
    columns=(
        ColumnSpec("date", LogicalDType.DATETIME, False),
        ColumnSpec("asset_id", LogicalDType.STRING, False),
        ColumnSpec("open", LogicalDType.FLOAT, True),
        ColumnSpec("high", LogicalDType.FLOAT, True),
        ColumnSpec("low", LogicalDType.FLOAT, True),
        ColumnSpec("close", LogicalDType.FLOAT, True),
        ColumnSpec("adjusted_close", LogicalDType.FLOAT, False),
        ColumnSpec("volume", LogicalDType.FLOAT, True),
        ColumnSpec("source", LogicalDType.STRING, False),
        ColumnSpec("currency", LogicalDType.STRING, True),
    ),
    primary_key=("date", "asset_id"),
    sort_by=("asset_id", "date"),
)


PROCESSED_MARKET_DATA_CONTRACT = TableContract(
    name="processed_market_data",
    columns=(
        ColumnSpec("date", LogicalDType.DATETIME, False),
        ColumnSpec("asset_id", LogicalDType.STRING, False),
        ColumnSpec("adjusted_close", LogicalDType.FLOAT, False),
        ColumnSpec("is_observed", LogicalDType.BOOLEAN, False),
    ),
    primary_key=("date", "asset_id"),
    sort_by=("asset_id", "date"),
)


@dataclass(frozen=True, slots=True)
class RawDataManifest:
    asset_id: str
    source: str
    provider_symbol: str
    frequency: Frequency
    fetched_at_utc: datetime
    requested_start: date | None
    requested_end: date | None
    row_count: int
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id cannot be empty")

        if not self.source:
            raise ValueError("source cannot be empty")

        if not self.provider_symbol:
            raise ValueError("provider_symbol cannot be empty")

        if self.row_count < 0:
            raise ValueError("row_count cannot be negative")

        if (
            self.requested_start is not None
            and self.requested_end is not None
            and self.requested_start > self.requested_end
        ):
            raise ValueError(
                "requested_start cannot be after requested_end"
            )