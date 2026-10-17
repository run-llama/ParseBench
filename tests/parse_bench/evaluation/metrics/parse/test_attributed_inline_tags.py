"""Inline formatting tags carrying attributes are stripped like their bare forms.

Parsers emit ``<mark style="background-color:#ffff00">`` (and the same for
``<u>``, ``<b>``, ...). Every normalizer that deletes the bare tag must delete
the attributed one too, or the leftover opening tag fails text comparisons
(e.g. an ``is_title`` rule whose heading is highlighted).
"""

import pytest

from parse_bench.evaluation.metrics.parse.rules_formatting import _strip_other_formatting
from parse_bench.evaluation.metrics.parse.utils import normalize_cell_text, normalize_text

TAGS = ["b", "i", "u", "ins", "mark", "s", "del", "strike"]


@pytest.mark.parametrize("tag", TAGS)
def test_normalize_text_strips_attributed_tag(tag: str) -> None:
    attributed = f'Annual <{tag} style="background-color:#ffff00">Report</{tag}>'
    assert normalize_text(attributed) == "annual report"


@pytest.mark.parametrize("tag", TAGS)
def test_normalize_text_attributed_matches_bare(tag: str) -> None:
    bare = f"Annual <{tag}>Report</{tag}> 2024"
    attributed = f'Annual <{tag} class="x">Report</{tag}> 2024'
    assert normalize_text(attributed) == normalize_text(bare)


def test_normalize_text_uppercase_mark() -> None:
    assert normalize_text('<MARK style="color:red">Title</MARK>') == "title"


def test_normalize_text_keeps_lookalike_tags() -> None:
    # \b keeps <b>/<s>/<u> from eating <br>, <sup>, <ul>; <sup> still drops content.
    assert normalize_text("a<br>b") == "a b"
    assert normalize_text("x<sup>2</sup>") == "x"
    assert normalize_text("<ul>item</ul>") == "<ul>item</ul>"


@pytest.mark.parametrize("tag", ["b", "strong", "i", "em", "u", "ins", "s", "del", "strike", "mark"])
def test_normalize_cell_text_strips_attributed_tag(tag: str) -> None:
    assert normalize_cell_text(f'<{tag} style="color:red">Total</{tag}>') == "Total"


@pytest.mark.parametrize(
    ("other", "keep"),
    [
        ("b", "mark"),
        ("strong", "mark"),
        ("i", "mark"),
        ("em", "mark"),
        ("u", "mark"),
        ("ins", "mark"),
        ("s", "mark"),
        ("del", "mark"),
        ("strike", "mark"),
        ("sup", "mark"),
        ("sub", "mark"),
        ("mark", "bold"),
    ],
)
def test_strip_other_formatting_strips_attributed_tag(other: str, keep: str) -> None:
    text = f'<{other} style="color:red">word</{other}>'
    assert _strip_other_formatting(text, keep) == "word"


def test_strip_other_formatting_keeps_attributed_kept_kind() -> None:
    # The kept kind's tags must survive with their attributes: MarkColorRule reads them.
    text = '<mark style="background-color:#ffff00"><u class="a">word</u></mark>'
    assert _strip_other_formatting(text, "mark") == '<mark style="background-color:#ffff00">word</mark>'
