import pytest

from portfolio_risk_model.data.contracts import (
    ColumnSpec,
    INTERIM_MARKET_DATA_CONTRACT,
    LogicalDType,
    TableContract,
)


def test_interim_contract_primary_key() -> None:
    assert INTERIM_MARKET_DATA_CONTRACT.primary_key == (
        "date",
        "asset_id",
    )


def test_interim_contract_contains_adjusted_close() -> None:
    assert (
        "adjusted_close"
        in INTERIM_MARKET_DATA_CONTRACT.column_names
    )


def test_primary_key_columns_are_required() -> None:
    required = INTERIM_MARKET_DATA_CONTRACT.required_columns

    assert "date" in required
    assert "asset_id" in required


def test_contract_rejects_duplicate_columns() -> None:
    with pytest.raises(ValueError):
        TableContract(
            name="invalid",
            columns=(
                ColumnSpec(
                    "date",
                    LogicalDType.DATETIME,
                    False,
                ),
                ColumnSpec(
                    "date",
                    LogicalDType.DATETIME,
                    False,
                ),
            ),
            primary_key=("date",),
            sort_by=("date",),
        )


def test_contract_rejects_nullable_primary_key() -> None:
    with pytest.raises(ValueError):
        TableContract(
            name="invalid",
            columns=(
                ColumnSpec(
                    "date",
                    LogicalDType.DATETIME,
                    True,
                ),
            ),
            primary_key=("date",),
            sort_by=("date",),
        )