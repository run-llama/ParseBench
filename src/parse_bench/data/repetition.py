"""Repetition tests for your own documents: write a ``repetition_overshoot`` test next to every PDF.

The budgets come from each document's text layer: every leader, rule or blank and how long it runs, every unit and
how often it occurs, and the number of text lines. A parser then fails a document only when its output repeats
something past what the document holds (see ``rules_repetition.py``).

A PDF with no text layer (a scan) has nothing to budget from. By default it is skipped; ``scans=True`` tests it as a
page that repeats nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from parse_bench.evaluation.metrics.parse.rules_repetition import FILL, fill_runs, strip_markup

# A run of one other symbol a symbol font may use for a leader (~, >, ․), spaced or not
_SYMBOL_RUN = re.compile(r"([^\w\s|])(?:[ \t]*\1){9,}")

SCAN_RULE: dict[str, Any] = {
    "type": "repetition_overshoot",
    "reference_text": "",
    "skip_units_absent_from_reference": False,
    "line_cap": 40,
    "max_chars": 5000,
}


def text_rule(text: str) -> dict[str, Any]:
    """The ``repetition_overshoot`` rule for a page or document whose text layer is ``text``."""
    symbols = [
        sum(1 for c in m.group(0) if not c.isspace()) for m in _SYMBOL_RUN.finditer(text) if m.group(1) not in FILL
    ]
    return {
        "type": "repetition_overshoot",
        "reference_text": text,
        "page_fill_runs": fill_runs(strip_markup(text)) + symbols,
        "page_lines": sum(1 for line in text.splitlines() if line.strip()),
    }


def text_layer(pdf: Path) -> str:
    try:
        import pymupdf
    except ImportError as e:
        raise ImportError("Reading text layers needs PyMuPDF: pip install 'parse-bench[local]'") from e

    with pymupdf.open(pdf) as doc:
        return "\n\n".join(page.get_text("text", sort=True) for page in doc)


def make_tests(input_dir: Path, overwrite: bool = False, scans: bool = False) -> dict[str, list[str]]:
    """Write ``<name>.test.json`` with one ``repetition_overshoot`` rule next to every PDF under ``input_dir``.

    Returns the PDFs written, kept (a test already there) and skipped (no text layer), by name.
    """
    done: dict[str, list[str]] = {"written": [], "kept": [], "skipped": []}
    for pdf in sorted(Path(input_dir).rglob("*.pdf")):
        out = pdf.with_name(f"{pdf.stem}.test.json")
        if out.exists() and not overwrite:
            done["kept"].append(str(pdf))
            continue
        text = text_layer(pdf)
        if sum(c.isalnum() for c in text) >= 20:
            rule, tags = text_rule(text), ["repetition", "text_layer"]
        elif scans:
            rule, tags = dict(SCAN_RULE), ["repetition", "scan"]
        else:
            done["skipped"].append(str(pdf))
            continue
        out.write_text(json.dumps({"test_rules": [rule], "tags": tags}, ensure_ascii=False, indent=1), encoding="utf-8")
        done["written"].append(str(pdf))
    return done


class RepetitionCLI:
    """Repetition tests: does a parser repeat itself more than the document does?

    Write a ``repetition_overshoot`` test, budgeted from the text layer, next to each PDF, then run any pipeline on
    the folder:

        parse-bench repetition make ./my_docs
        parse-bench run llamaparse_agentic --input_dir ./my_docs
    """

    def make(self, input_dir: str | Path, overwrite: bool = False, scans: bool = False) -> int:
        """Write a ``repetition_overshoot`` test (``<name>.test.json``) next to every PDF in a folder.

        Args:
            input_dir: Folder of PDFs (searched recursively)
            overwrite: Replace test.json files that already exist (default: keep them)
            scans: Also test PDFs with no text layer, as pages that repeat nothing (default: skip them)

        Example:
            parse-bench repetition make ./my_docs
            parse-bench repetition make ./my_docs --scans
        """
        root = Path(input_dir)
        if not root.is_dir():
            print(f"Not a folder: {root}")
            return 1
        try:
            done = make_tests(root, overwrite=overwrite, scans=scans)
        except ImportError as e:
            print(e)
            return 1
        written, kept, skipped = (len(done[k]) for k in ("written", "kept", "skipped"))
        print(f"Repetition tests written: {written}, kept: {kept}, skipped: {skipped}")
        for pdf in done["skipped"]:
            print(f"  skipped (no text layer, pass --scans to test it as a page that repeats nothing): {pdf}")
        if not done["written"] and not done["kept"]:
            print(f"No PDF to test under {root}")
            return 1
        print(f"Next: parse-bench run <pipeline> --input_dir {root}")
        return 0
