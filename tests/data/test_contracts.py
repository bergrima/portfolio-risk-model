from portfolio_risk_model.data.contracts import PROCESSED_MARKET_DATA_CONTRACT


def test_processed_contract_contains_return() -> None:
    assert "return" in PROCESSED_MARKET_DATA_CONTRACT.column_names


def test_processed_contract_uses_same_primary_key() -> None:
    assert PROCESSED_MARKET_DATA_CONTRACT.primary_key == (
        "date",
        "asset_id",
    )


def test_processed_return_is_nullable() -> None:
    return_spec = next(
        column
        for column in PROCESSED_MARKET_DATA_CONTRACT.columns
        if column.name == "return"
    )

    assert return_spec.nullable is True
