"""The V2 markdown payload uses ``null`` for absent header/footer/printed page number."""

from __future__ import annotations

from parse_bench.inference.providers.parse.llamaparse_v2_normalization import MarkdownPage, MarkdownResult


def test_null_page_text_fields_normalise_to_empty_strings() -> None:
    result = MarkdownResult.model_validate(
        {
            "pages": [
                {
                    "page_number": 1,
                    "md": "# Title",
                    "pageHeaderMarkdown": None,
                    "pageFooterMarkdown": None,
                    "printedPageNumber": None,
                },
                {"page_number": 2, "md": None},
            ]
        }
    )
    first, second = result.pages
    assert (first.header, first.footer, first.printed_page_number) == ("", "", "")
    assert first.markdown == "# Title"
    assert second.markdown == ""


def test_present_page_text_fields_are_kept() -> None:
    page = MarkdownPage.model_validate(
        {
            "page_number": 3,
            "md": "body",
            "pageHeaderMarkdown": "Hdr",
            "pageFooterMarkdown": "Ftr",
            "printedPageNumber": "iii",
        }
    )
    assert (page.header, page.footer, page.printed_page_number) == ("Hdr", "Ftr", "iii")
