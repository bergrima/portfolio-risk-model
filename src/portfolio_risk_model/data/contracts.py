from dataclasses import dataclass
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
            missing = ", ".join(sorted(missing_primary_key))
            raise ValueError(f"Primary key columns not found in contract: {missing}")

        missing_sort_columns = set(self.sort_by) - set(column_names)

        if missing_sort_columns:
            missing = ", ".join(sorted(missing_sort_columns))
            raise ValueError(f"Sort columns not found in contract: {missing}")

        columns_by_name = {column.name: column for column in self.columns}

        nullable_primary_keys = [
            column_name
            for column_name in self.primary_key
            if columns_by_name[column_name].nullable
        ]

        if nullable_primary_keys:
            nullable = ", ".join(sorted(nullable_primary_keys))
            raise ValueError(f"Primary key columns cannot be nullable: {nullable}")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if not column.nullable)


INTERIM_MARKET_DATA_CONTRACT = TableContract(
    name="interim_market_data",
    columns=(
        ColumnSpec(
            name="date",
            dtype=LogicalDType.DATETIME,
            nullable=False,
        ),
        ColumnSpec(
            name="asset_id",
            dtype=LogicalDType.STRING,
            nullable=False,
        ),
        ColumnSpec(
            name="adjusted_close",
            dtype=LogicalDType.FLOAT,
            nullable=False,
        ),
    ),
    primary_key=("date", "asset_id"),
    sort_by=("asset_id", "date"),
)


PROCESSED_MARKET_DATA_CONTRACT = TableContract(
    name="processed_market_data",
    columns=(
        ColumnSpec(
            name="date",
            dtype=LogicalDType.DATETIME,
            nullable=False,
        ),
        ColumnSpec(
            name="asset_id",
            dtype=LogicalDType.STRING,
            nullable=False,
        ),
        ColumnSpec(
            name="adjusted_close",
            dtype=LogicalDType.FLOAT,
            nullable=False,
        ),
        ColumnSpec(
            name="return",
            dtype=LogicalDType.FLOAT,
            nullable=True,
        ),
    ),
    primary_key=(
        "date",
        "asset_id",
    ),
    sort_by=(
        "date",
        "asset_id",
    ),
)
