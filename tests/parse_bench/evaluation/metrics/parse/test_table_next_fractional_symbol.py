import pytest

from parse_bench.evaluation.metrics.parse.table_next import score_table_next


@pytest.mark.parametrize("value", ["1/3 $ V_{DD} $", "1/3V_{DD}", "$1/3V_{DD}$", "$1/3V_{DD} $", "1/3 VDD"])
def test_fractional_subscript_notation_does_not_penalize_equivalent_label(value: str) -> None:
    table = "| Pin | Format |\n|---|---|\n| 1/3V<sub>DD</sub> | Binary |"
    assert score_table_next(table, table.replace("1/3V<sub>DD</sub>", value)).score == 1


@pytest.mark.parametrize("value", ["2/3V_{DD}", "1/3V_{SS}", "-1/3V_{DD}", "1/3A_{DD}", "1/3V_{DD} mV"])
def test_fractional_symbol_value_subscript_sign_and_unit_errors_remain(value: str) -> None:
    table = "| Pin | Format |\n|---|---|\n| 1/3V<sub>DD</sub> | Binary |"
    assert score_table_next(table, table.replace("1/3V<sub>DD</sub>", value)).score < 1
