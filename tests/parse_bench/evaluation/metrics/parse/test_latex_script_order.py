import pytest

from parse_bench.evaluation.metrics.parse.rules_formatting import LatexRule


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (r"\sigma^2_\eta", r"\sigma_\eta^2"),
        (r"\sigma^2_{\pi^0}", r"\sigma_{\pi^{0}}^{2}"),
        (r"x^{a_{i}^{2}}_{j}", r"x_j^{a^2_i}"),
        (r"x^2_i + y^3_j", r"x_i^2 + y_j^3"),
        (r"x^\alpha_i", r"x_{i}^{\alpha}"),
        (r"x^{\{a\}}_i", r"x_i^{\{a\}}"),
    ],
)
def test_equivalent_scripts(expected, actual):
    for left, right in [(expected, actual), (actual, expected)]:
        assert LatexRule({"type": "is_latex", "formula": left}).run("$" + right + "$")[0]


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (r"\chi^2_{\eta\eta\pi^0}", r"\chi^2_{\pi^0\eta\eta}"),
        (r"x_i^2", r"x_j^2"),
        (r"x_i^2", r"x_i^3"),
        (r"x_{i^2}", r"x_i^2"),
        (r"x_i y^2", r"x_i^2 y"),
        (r"x^2_\eta", r"x^2_e"),
        (r"x^2^3", r"x^3"),
    ],
)
def test_different_scripts_still_fail(expected, actual):
    assert not LatexRule({"type": "is_latex", "formula": expected}).run("$" + actual + "$")[0]


def test_case_0007_inline_formula_with_reversed_script_order():
    expected = (
        r"\chi^2_{\pi^0\eta\eta} = "
        r"\frac{(M(\gamma_1\gamma_2)-m_{\pi^0})^2}{\sigma^2_{\pi^0}} + "
        r"\frac{(M(\gamma_3\gamma_4)-m_\eta)^2}{\sigma^2_\eta} + "
        r"\frac{(M(\gamma_5\gamma_6)-m_\eta)^2}{\sigma^2_\eta}"
    )
    actual = expected.replace(r"\sigma^2_{\pi^0}", r"\sigma_{\pi^0}^2").replace(r"\sigma^2_\eta", r"\sigma_\eta^2")
    rule = LatexRule({"type": "is_latex", "formula": expected})
    assert rule.run("Candidates are selected by minimizing $" + actual + "$, where masses are nominal.")[0]
    wrong = expected.replace(r"\pi^0\eta\eta", r"\eta\eta\pi^0")
    assert not LatexRule({"type": "is_latex", "formula": wrong}).run("$" + actual + "$")[0]
