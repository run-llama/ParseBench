from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.text_next import score_text_next, tokenize


@pytest.mark.parametrize("style", ["", "**", "*", "__"])
def test_struck_link_does_not_append_markdown_tildes_to_visible_url(style: str) -> None:
    reference = "Read www.example.org for details."
    output = f"Read ~~{style}[www.example.org](https://redirect.example/?id=52){style}~~ for details."
    assert score_text_next(reference, output).score == 1.0
    assert score_text_next(output, reference).score == 1.0


@pytest.mark.parametrize(
    ("reference", "output"),
    [
        ("https://www.example.org/en/Data", "https://www.example. org/en/Data"),
        ("https://example.org/-/media/report.pdf", "https://example.org/- /media/report.pdf"),
        ("www.example.org/investor-relations/announcements/", "www.example.org/investor-relations/ announcements/"),
        ("http://example.org/?published=on", "http://example.org/?pu\nblished=o\nn"),
        ("https://example.org/volume-54", "https://example.org/volume-\n54"),
        ("https://example.org/global-warming.", "https://example.org/global- warming"),
        ("www.example.org", "www. example.org"),
    ],
)
def test_url_wrap_keeps_every_character_and_uses_other_sides_complete_url(reference: str, output: str) -> None:
    assert score_text_next(f"Read {reference} for details.", f"Read {output} for details.").score == 1.0
    assert score_text_next(f"Read {output} for details.", f"Read {reference} for details.").score == 1.0


@pytest.mark.parametrize(
    "output",
    [
        "https://example.org/notes/data-2025.pdf?published=on",
        "https://example.org/notesdata-2024.pdf?published=on",
        "https://example.org/notes/data-2024.pdf?published=off",
        "https://example.org/notes/data-2024.pdf?published=\nof",
        "https://example.org/notes/data-2024.pdf",
        "[https://example.net/other](https://example.org/notes/data-2024.pdf?published=on)",
    ],
)
def test_url_reconstruction_does_not_hide_wrong_digits_hyphens_suffixes_or_labels(output: str) -> None:
    reference = "https://example.org/notes/data-2024.pdf?published=on"
    result = score_text_next(f"Read {reference} for details.", f"Read {output} for details.")
    assert result.score < 1.0
    assert result.events.critical + result.events.major > 0


def test_literal_url_tildes_are_not_strikeout_and_url_hyphens_are_not_word_hyphenation() -> None:
    assert [token.key for token in tokenize("https://example.org/a-b/~~draft~~")[0]] == [
        "#https://example.org/a-b/~~draft~~"
    ]
    assert score_text_next("https://example.org/a-b", "https://example.org/ab").score < 1.0


def test_wrapped_urls_still_charge_url_reordering_and_repetition() -> None:
    first = "https://example.org/alpha/first.pdf"
    second = "https://example.org/beta/second.pdf"
    reference = f"First source {first}\n\nSecond source {second}"
    output = f"First source {second.replace('/second', '/ second')}\n\nSecond source {first}"
    result = score_text_next(reference, output)
    assert result.score < 1.0
    assert result.order < 1.0 or result.critical < 1.0
    assert score_text_next(reference, f"{reference}\n\n{first}").score < 1.0


def test_link_inside_image_alt_does_not_invent_image_destination_text() -> None:
    reference = "The report is available today."
    output = reference + "\n![icon: [Example](https://example.org/channel/42)](page_1_image_7_v2.jpg)"
    assert score_text_next(reference, output).score == 1.0
    assert score_text_next(reference, output + "\npage_1_image_7_v2.jpg").score < 1.0


def test_linked_image_alt_still_recovers_printed_text_without_link_destinations() -> None:
    reference = "The annual report contains the complete regional results."
    output = "![image: [The annual report contains the complete regional results.](https://example.org)](page.jpg)"
    assert score_text_next(reference, output).score == 1.0
    assert score_text_next(reference, "![icon: [Logo](https://example.org)](page.jpg)").score < 1.0


def test_footer_separator_does_not_swallow_following_domain_or_split_url_paths() -> None:
    reference = "Call 70 / www.example.org for details."
    assert score_text_next(reference, "Call 70/www.example.org for details.").score == 1.0
    url = "https://example.org/70/www.example.net/path"
    assert [token.key for token in tokenize(url)[0]] == [f"#{url}"]


def test_multiline_link_label_does_not_expose_hidden_url_as_text() -> None:
    reference = "Read the annual report for details."
    output = "Read [the annual\nreport](https://example.org/secret/2024) for details."
    assert score_text_next(reference, output).score == 1.0


@pytest.mark.parametrize(
    "output",
    [
        "https://example.org/pub [report](https://example.org/publications)",
        "https://example.org/pub\n\nlications",
        "https://example.org/pub other text lications",
    ],
)
def test_url_wrap_does_not_borrow_hidden_suffix_or_cross_another_paragraph(output: str) -> None:
    assert score_text_next("https://example.org/publications", output).score < 1.0
