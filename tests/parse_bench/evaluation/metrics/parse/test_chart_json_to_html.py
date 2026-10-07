from parse_bench.evaluation.metrics.parse.chart_json_to_html import chart_json_to_html
from parse_bench.evaluation.metrics.parse.rules_chart import parse_chart_tables


def test_point_lists_preserve_repeated_x_values_and_series():
    chart = {"values": {"Sales": [{"x": "2024", "y": 10}, {"x": "2024", "y": 12}], "Profit": {"2024": 3}}}
    points, categories = parse_chart_tables(chart_json_to_html(chart))
    assert points.caption == "Sales"
    assert points.data.tolist() == [["x", "y"], ["2024", "10"], ["2024", "12"]]
    assert categories.data.tolist() == [["", "Profit"], ["2024", "3"]]
