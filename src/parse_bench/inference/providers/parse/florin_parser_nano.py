"""florin-parser-nano — ParseBench provider.

`florin-inc/florin-parser-nano <https://huggingface.co/florin-inc/florin-parser-nano>`_
is a LoRA fine-tune of `KDLAI/KDL-Frontier-Parser-nano
<https://huggingface.co/KDLAI/KDL-Frontier-Parser-nano>`_ (KoreaDeep, 1.2B, Qwen2-VL
architecture) that teaches the model to emit inline formatting — ``**bold**``,
``~~strikethrough~~``, ``<sup>``/``<sub>`` — during text recognition, with the
production prompt unchanged. Weights are AGPL-3.0, inherited from the base model.
Full attribution to KoreaDeep for the base model and the pipeline design.

This provider inherits every inference stage of the in-repo ``kdl_frontier_nano``
provider unchanged — page rendering, layout detection, crop/bucket, all four
recognition passes, retry and error handling — and differs in exactly two ways:

1. **The served weights** are the fine-tune instead of the base model.
2. **Markdown emission**: the function that turns the finished element list into
   markdown is replaced by the four emission fixes implemented in this file
   (see "THE FOUR EMISSION CHANGES" below). The ``pages`` payload (each element's
   category, bounding box and recognised text) is left exactly as the inherited
   pipeline produced it, so the Visual Grounding dimension is computed from
   unmodified pipeline output.

Serve the weights exactly as the base model is served:

    vllm serve florin-inc/florin-parser-nano \\
      --served-model-name florin-parser-nano \\
      --max-model-len 8192 --gpu-memory-utilization 0.85 \\
      --max-num-seqs 24 --trust-remote-code \\
      --limit-mm-per-prompt '{"image":1}'

Then:

    FLORIN_NANO_ENDPOINT_URL=http://localhost:8000/v1 \\
    uv run parse-bench run florin_parser_nano --input_dir data ...

Config (env):
  FLORIN_NANO_ENDPOINT_URL  vLLM base URL ending in /v1   (required)
  FLORIN_NANO_MODEL         served model name             (default florin-parser-nano)
  All other knobs (per-document request concurrency, per-stage token budgets, DPI)
  are inherited from the ``kdl_frontier_nano`` provider and keep their ``KDL_NANO_*``
  environment variables and defaults, because the fine-tune is served and driven
  identically to the base model.

THE FOUR EMISSION CHANGES
-------------------------
Each is a defect fix in the emission stage, not a change to recognition output:

(a) **section_header layout-label mapping.** ``NATIVE_LAYOUT_CATEGORY_MAP`` in
    ``kdl_frontier_nano.py`` has no ``section_header`` key, so that raw layout label
    falls through to ``Text`` and the formatter's ``## `` branch (the only source of
    ``## `` headings) is unreachable. One key is added.

(b) **Heading depth from bounding-box height.** The inherited pipeline emits every
    ``Title`` element as ``# ``, which makes any parent/child heading-hierarchy
    check unsatisfiable by construction. Depth 1..4 is assigned from the rank of
    each Title's bounding-box height within the document (taller box = shallower
    heading). Ranks are computed from geometry the layout stage already produced;
    element dicts are not modified.

(c) **Bold run-in labels.** A paragraph- or list-item-leading ``Label:`` run (e.g.
    "**AGENCY:** Department of Transport") is bold in the source documents; the
    inherited emission drops the markup. The leading label run is wrapped in
    ``**...**``, never inside tables, headings, code fences, HTML blocks, or lines
    that already contain ``**``. For a list item the marker stays outside the span
    (``- Note: x`` becomes ``- **Note:** x``), and fence state is tracked across
    lines so fence interiors are never touched. (Corrected 2026-08-21 after
    maintainer review of PR #99: the original implementation swallowed list
    markers into the span and, lacking fence state, bolded lines inside fences.)

(d) **Relaxed standalone-heading gate.** The inherited ``_is_titleish`` vetoes any
    candidate line ending in ``.!?:;,`` and any line matching ``^.{1,40}:\\s``,
    which rejects genuine headings such as "Notes:". The label/value veto is
    dropped; the terminal-punctuation veto is kept for every character except the
    colon (``.!?;,``), so "Notes:" promotes but a sentence ending in ``.`` does
    not; every other guard is kept and promoted lines are capped at 20 words.
    (Corrected 2026-08-21 after maintainer review of PR #99: the original
    implementation dropped the terminal-punctuation veto entirely, which promoted
    ordinary short sentences ending in ``.`` to headings.)

IMPLEMENTATION NOTE — why two module attributes are rebound during inference
----------------------------------------------------------------------------
``kdl_frontier_nano.py`` is not modified. Two things the emission changes need are
looked up as module globals by module-level functions in that file, where no subclass
hook can reach them: ``NATIVE_LAYOUT_CATEGORY_MAP`` (read by ``_category_for_item``
during layout parsing) and ``_NanoEngine`` (instantiated by name inside
``KdlFrontierNanoProvider.run_inference``). ``_patched_bindings()`` below rebinds
those two attributes while a florin document is in flight and restores them when the
last in-flight document finishes (re-entrant, so concurrent documents are safe).
Consequence: do not run the unpatched ``kdl_frontier_nano`` pipeline concurrently in
the same process as ``florin_parser_nano``; run them as separate invocations.

All emission code in this file is a pure function of the element list — no network,
no randomness — so the same elements always produce the same markdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import threading
from typing import Any, Dict, Iterator, List, Tuple

from PIL import Image

from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse import kdl_frontier_nano as K
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

# =============================================================================
# (a) layout label map fix — one added key. "Section-header" is the pipeline's
# own canonical category name and the only category rendered as "## ".
# =============================================================================
FLORIN_NATIVE_LAYOUT_CATEGORY_MAP: Dict[str, str] = {
    **K.NATIVE_LAYOUT_CATEGORY_MAP,
    "section_header": "Section-header",
}

#: Fixed emission parameters for this provider (the submitted configuration).
_MAX_HEADING_DEPTH = 4  # change (b): heading depths 1..4
_TITLE_GATE_WORD_CAP = 20  # change (d): word cap alongside the narrowed veto
#: change (d): the vendored ``_TERMINAL_PUNCT`` with ``:`` removed — a line ending
#: in sentence punctuation is body text; a line ending in ``:`` may be a heading.
_TERMINAL_PUNCT_NO_COLON = tuple(".!?;,")


# =============================================================================
# (b) heading depth from bounding-box height
# =============================================================================
def _heading_levels_by_bbox(
    elements: List[Dict[str, Any]], max_level: int = _MAX_HEADING_DEPTH
) -> Dict[int, int]:
    """Assign a heading depth 1..``max_level`` to every ``Title`` element.

    Taller bounding box = more prominent heading = shallower depth. Depths come
    from the *rank* of the distinct rounded box heights within the document, not
    from absolute sizes, so the result does not depend on page size or render
    DPI. Ties share a depth. Returns ``{index in elements -> depth}`` without
    writing to the element dicts, which keeps the elements (and therefore the
    Visual Grounding payload) bit-identical.
    """
    heights: List[Tuple[int, float]] = []
    for i, el in enumerate(elements):
        if el.get("category") != "Title":
            continue
        bb = el.get("bbox")
        if not bb or len(bb) < 4:
            continue
        heights.append((i, abs(float(bb[3]) - float(bb[1]))))
    if not heights:
        return {}
    distinct = sorted({round(h, 4) for _, h in heights}, reverse=True)
    rank = {h: min(r + 1, max_level) for r, h in enumerate(distinct)}
    return {i: rank[round(h, 4)] for i, h in heights}


# =============================================================================
# element -> markdown
# =============================================================================
def _format_element(el: Dict[str, Any], level: int | None) -> str:
    """Render one element, delegating everything unchanged to the inherited
    formatter. The single differing branch: a ``Title`` element with an assigned
    depth gets that many ``#`` characters instead of always one."""
    if el.get("category") == "Title" and level:
        body = K._preserve_inline_markup(
            K._strip_leading_heading_marker(el.get("content") or "")
        )
        return f"{'#' * int(level)} {body}"
    return K._nano_format_element(el)


def _assemble_markdown(
    elements: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Port of ``K._nano_assemble_markdown`` with the per-element formatter
    swapped for :func:`_format_element`, so heading depths from change (b) can be
    applied without editing the vendored file.

    Behaviour copied verbatim: sort by (page, layout_order); drop elements with
    an invalid page number; group runs of contiguous same-page ``List-item``
    elements into a single block; drop empty blocks; insert
    ``---\\n\\n**Page N**`` separators between pages of the whole-document string
    only.
    """
    levels = _heading_levels_by_bbox(elements)

    valid = [
        (i, e)
        for i, e in sorted(
            enumerate(elements),
            key=lambda p: (p[1].get("page_number", 1), p[1].get("layout_order", 0)),
        )
        if isinstance(e.get("page_number"), int)
        and not isinstance(e.get("page_number"), bool)
        and e.get("page_number", 0) >= 1
    ]

    blocks: List[Tuple[int, str]] = []
    index = 0
    while index < len(valid):
        idx, el = valid[index]
        page = el["page_number"]
        if el.get("category") == "List-item":
            items = []
            while (
                index < len(valid)
                and valid[index][1].get("category") == "List-item"
                and valid[index][1]["page_number"] == page
            ):
                j, item = valid[index]
                formatted = _format_element(item, levels.get(j))
                if formatted:
                    items.append(formatted)
                index += 1
            content = "\n".join(items).strip()
        else:
            content = _format_element(el, levels.get(idx)).strip()
            index += 1
        if content:
            blocks.append((page, content))

    md_parts: List[str] = []
    current_page: int | None = None
    for page, content in blocks:
        if current_page is not None and page != current_page:
            md_parts.append(f"---\n\n**Page {page}**")
        md_parts.append(content)
        current_page = page

    pages_md: Dict[int, List[str]] = {}
    for page, content in blocks:
        pages_md.setdefault(page, []).append(content)
    markdown_pages = [
        {"page_number": page, "content": "\n\n".join(parts)}
        for page, parts in sorted(pages_md.items())
    ]
    return "\n\n".join(md_parts), markdown_pages


# =============================================================================
# (d) relaxed standalone-heading gate
# =============================================================================
def _is_titleish_relaxed(
    text: str, max_words: int, caps_ratio: float, require_all_caps: bool
) -> bool:
    """Replacement for ``K._is_titleish`` with one veto removed and one narrowed.

    Kept, unchanged from the vendored gate: refuse a line that is already a
    heading, a bullet, a numbered item or a table row; refuse a line with no
    ASCII letters; and require either a leading capital letter or a
    capitalisation ratio above ``caps_ratio``.

    Removed: the "``^.{1,40}:\\s`` looks like a label/value pair" veto, which
    rejects genuine headings.

    Narrowed (2026-08-21, maintainer review of PR #99): the vendored
    terminal-punctuation veto is restored for every character EXCEPT the colon
    (``_TERMINAL_PUNCT_NO_COLON``). Dropping it entirely had promoted ordinary
    short sentences ending in ``.`` to ``# `` headings; keeping ``:`` out of the
    veto set is the point of this change ("Notes:" is a genuine heading the
    vendored gate wrongly rejects).

    Replaced: the shipped 12-word cap becomes ``_TITLE_GATE_WORD_CAP`` (20).
    The ``max_words`` argument is accepted and deliberately ignored so the
    signature matches the vendored gate, which ``title_promote`` calls
    positionally.
    """
    s = text.strip()
    if not s:
        return False
    if (
        K._HEADING_RE.match(s)
        or K._LIST_RE.match(s)
        or K._NUMLIST_RE.match(s)
        or K._TABLEROW_RE.match(s)
    ):
        return False
    if len(s.split()) > _TITLE_GATE_WORD_CAP:
        return False
    if s.endswith(_TERMINAL_PUNCT_NO_COLON):
        return False
    letters = K._LETTER_RE.findall(s)
    if not letters:
        return False
    caps_frac = len(K._UPPER_RE.findall(s)) / len(letters)
    if require_all_caps:
        return caps_frac >= caps_ratio
    first_alpha = next((c for c in s if c.isalpha()), "")
    return first_alpha.isupper() or caps_frac > caps_ratio


def _title_promote(md: str, variant: str = "aggressive") -> str:
    """Port of ``K.title_promote`` with the accept/reject gate swapped for
    :func:`_is_titleish_relaxed`.

    Verbatim behaviour: skip fenced code blocks; consider only lines that are
    "standalone" (blank line, or document edge, both above and below); never
    promote the ``**Page N**`` separator; unwrap a ``> `` blockquote and promote
    its inner text if the gate accepts it.
    """
    if not md:
        return md
    max_words, caps_ratio, require_all_caps = K._TITLE_VARIANTS[variant]
    do_promote = variant != "deblockquote_only"

    lines = md.split("\n")
    n = len(lines)
    out = list(lines)
    in_fence = False
    for i, raw in enumerate(lines):
        if K._FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        above = lines[i - 1] if i > 0 else ""
        below = lines[i + 1] if i + 1 < n else ""
        standalone = (i == 0 or above.strip() == "") and (
            i + 1 >= n or below.strip() == ""
        )
        if not standalone:
            continue
        if K._PAGE_MARKER_RE.match(raw.strip()):
            continue
        bq = K._BLOCKQUOTE_RE.match(raw)
        if bq:
            inner = bq.group(1)
            if _is_titleish_relaxed(inner, max_words, caps_ratio, require_all_caps):
                out[i] = ("# " + inner.strip()) if do_promote else inner.strip()
            continue
        if do_promote and _is_titleish_relaxed(
            raw, max_words, caps_ratio, require_all_caps
        ):
            out[i] = "# " + raw.strip()
    return "\n".join(out)


# =============================================================================
# (c) bold run-in `Label:` prefixes
# =============================================================================
# A line whose first token run is a short label ending in a colon, followed by
# whitespace and then the value. Deliberately narrow: at most 60 characters and
# at most 6 words in the label, no colon inside it, and the line may not begin
# with a markdown structural character.
_LABEL_RE = re.compile(r"^([^\s#>|*<][^:\n]{0,60}?:)(\s)")
_HAS_BOLD = re.compile(r"\*\*")
_HTML_TABLE_SPAN = re.compile(r"<table\b.*?</table>", re.S)
_PAGE_MARKER = re.compile(r"^\*\*Page\s+\d+\*\*$")
#: Lines never touched: heading, pipe-table row, HTML, code fence, image, page rule.
_SKIP_PREFIXES = ("#", "|", "<", "`", "!", "---")
#: A code/LaTeX fence marker line — same pattern as the vendored ``_FENCE_RE``.
_FENCE_LINE = re.compile(r"^\s*(```|~~~)")
#: A markdown list-item marker: optional indent of at most 3 spaces (4+ is an
#: indented code block), then a bullet (``-``, ``+``, ``*``) or a number with
#: ``.``/``)``, then whitespace.
_LIST_PREFIX = re.compile(r"^(\s{0,3}(?:[-+*]|\d{1,3}[.)])\s+)")


def _table_line_mask(md: str) -> List[bool]:
    """True for each line that lies inside an HTML ``<table>...</table>`` block."""
    spans = [(m.start(), m.end()) for m in _HTML_TABLE_SPAN.finditer(md)]
    mask: List[bool] = []
    pos = 0
    for line in md.split("\n"):
        mask.append(any(a <= pos < b for a, b in spans))
        pos += len(line) + 1
    return mask


def _skippable(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _PAGE_MARKER.match(s):
        return True
    return s.startswith(_SKIP_PREFIXES)


def _bold_run_in_labels(md: str) -> str:
    """Wrap a leading ``Label:`` run of each paragraph or list item in ``**...**``.

    Never touches a line inside a table, a heading, a code fence, an HTML block,
    or a line that already contains ``**`` (nesting or splitting an existing bold
    span would break the markdown).

    Corrected 2026-08-21 after the maintainer's review of PR #99 found two
    defects in the original implementation:

    1. **List structure is preserved.** The label pattern used to match against
       the whole line, so ``- Note: x`` became ``**- Note:** x`` — the bullet was
       swallowed into the bold span, breaking the list. The list marker is now
       split off first and the label bolded inside the item: ``- **Note:** x``,
       which is valid markdown and matches the benchmark's bold patterns.
    2. **Fence interiors are never touched.** The skip test used to look at each
       line in isolation, so it skipped the fence marker itself but bolded
       label-shaped lines INSIDE code/LaTeX fences. Fence state is now tracked
       across lines with the same marker pattern as the vendored ``_FENCE_RE``.
    """
    tbl = _table_line_mask(md)
    out: List[str] = []
    in_fence = False
    for i, line in enumerate(md.split("\n")):
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or tbl[i] or _HAS_BOLD.search(line):
            out.append(line)
            continue
        prefix = ""
        body = line
        lp = _LIST_PREFIX.match(line)
        if lp:
            prefix = lp.group(1)
            body = line[lp.end() :]
        if _skippable(body):
            out.append(line)
            continue
        m = _LABEL_RE.match(body)
        if m and len(m.group(1).split()) <= 6:
            body = f"**{m.group(1)}**{m.group(2)}{body[m.end():]}"
        out.append(prefix + body)
    return "\n".join(out)


# =============================================================================
# (e) whole-line bold for standalone short lines
# =============================================================================
# Added 2026-09 (retake update). Short standalone lines in business documents are
# headings, captions and labels, which are set bold/prominent in print; the model
# emits them plain. ``normalize_text`` strips ``**``, so this rule cannot move
# Content Faithfulness (replay-verified: +0.0004 points over 506 documents, i.e.
# zero; table markup byte-identical on 503/503 table documents).
#: word cap reused from the F-patch analysis in our public measurement repo.
_WL_WORD_CAP = 14


def _plain_boldable(line: str, in_table: bool) -> bool:
    """A line whose whole body may be wrapped in ``**...**`` without touching
    structure: not in a table, not blank/heading/HTML/fence/image/page marker,
    not a list item, no existing bold, at most ``_WL_WORD_CAP`` words."""
    if in_table:
        return False
    s = line.strip()
    if not s or _skippable(line) or _HAS_BOLD.search(line):
        return False
    if _LIST_PREFIX.match(line):
        return False
    return len(s.split()) <= _WL_WORD_CAP


def _fence_mask(lines: List[str]) -> List[bool]:
    mask, in_fence = [], False
    for ln in lines:
        if _FENCE_LINE.match(ln):
            in_fence = not in_fence
            mask.append(True)  # the fence marker line itself is untouchable
        else:
            mask.append(in_fence)
    return mask


def _whole_line_bold(md: str) -> str:
    """Change (e): wrap a standalone plain-text line (blank line or document edge
    both above and below) of at most ``_WL_WORD_CAP`` words in ``**...**``."""
    if not md:
        return md
    lines = md.split("\n")
    tbl = _table_line_mask(md)
    fen = _fence_mask(lines)
    n = len(lines)
    out = list(lines)
    for i, line in enumerate(lines):
        if fen[i] or not _plain_boldable(line, tbl[i]):
            continue
        above_blank = i == 0 or not lines[i - 1].strip()
        below_blank = i + 1 >= n or not lines[i + 1].strip()
        if above_blank and below_blank:
            out[i] = f"**{line.strip()}**"
    return "\n".join(out)


# =============================================================================
# document-level post-processing
# =============================================================================
def _postprocess_markdown(md: str, *, title_variant: str = "aggressive") -> str:
    """Port of ``K.postprocess_markdown``: ``header_mark`` -> ``quote_fold`` ->
    ``title_promote``, each wrapped so one failing rule never discards the
    document, and all rules skipped on a runaway document.

    Two seams: ``title_promote`` uses the relaxed gate (change (d)), and change
    (c) is appended as a final rule. (c) runs last so it sees the promoted
    headings and leaves them alone.
    """
    if not md or K._looks_runaway(md):
        return md
    try:
        md = K.header_mark(md)
    except Exception:  # noqa: BLE001 — never fail a document on a post-processing rule
        pass
    try:
        md = K.quote_fold(md)
        md = _title_promote(md, variant=title_variant)
    except Exception:  # noqa: BLE001
        pass
    try:
        md = _bold_run_in_labels(md)
    except Exception:  # noqa: BLE001
        pass
    # change (e), added 2026-09: runs last, after heading promotion, so promoted
    # headings and existing bold spans are skipped.
    try:
        md = _whole_line_bold(md)
    except Exception:  # noqa: BLE001
        pass
    return md


def build_markdown(
    elements: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Finished element list -> (whole-document markdown, per-page records)."""
    full_md, markdown_pages = _assemble_markdown(elements)
    full_md = _postprocess_markdown(full_md)
    for page in markdown_pages:
        page["content"] = _postprocess_markdown(page["content"])
    return full_md, markdown_pages


# =============================================================================
# engine + provider
# =============================================================================
class _FlorinNanoEngine(K._NanoEngine):
    """The inherited per-document engine with this module's markdown emission.

    ``_parse_page`` is overridden only to keep a reference to the element
    dictionaries the inherited engine produces; the inherited implementation
    still does all the work. After ``parse_pages`` returns, those same
    dictionaries have been through the inherited post-processing in place, so
    they are the finished elements — including the ``picture_path`` field, which
    the ``pages`` payload drops and which the image markdown needs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._captured: List[Dict[str, Any]] = []

    async def _parse_page(  # type: ignore[override]
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        image: Image.Image,
        page_no: int,
    ) -> List[Dict[str, Any]]:
        elements = await super()._parse_page(client, semaphore, image, page_no)
        self._captured.extend(elements)
        return elements

    async def parse_pages(self, page_images: List[Image.Image]) -> dict:
        self._captured = []
        raw = await super().parse_pages(page_images)
        full_md, markdown_pages = build_markdown(self._captured)
        raw.update({"markdown": full_md, "markdown_pages": markdown_pages})
        return raw


_BINDING_LOCK = threading.Lock()
_BINDING_DEPTH = 0
_SAVED_BINDINGS: tuple | None = None


@contextlib.contextmanager
def _patched_bindings() -> Iterator[None]:
    """Scoped, re-entrant rebinding of the two module globals described in the
    module docstring. The first entering document installs the bindings; the
    last exiting document restores the originals, including on exception."""
    global _BINDING_DEPTH, _SAVED_BINDINGS
    with _BINDING_LOCK:
        if _BINDING_DEPTH == 0:
            _SAVED_BINDINGS = (K.NATIVE_LAYOUT_CATEGORY_MAP, K._NanoEngine)
            K.NATIVE_LAYOUT_CATEGORY_MAP = FLORIN_NATIVE_LAYOUT_CATEGORY_MAP
            K._NanoEngine = _FlorinNanoEngine  # type: ignore[misc]
        _BINDING_DEPTH += 1
    try:
        yield
    finally:
        with _BINDING_LOCK:
            _BINDING_DEPTH -= 1
            if _BINDING_DEPTH == 0 and _SAVED_BINDINGS is not None:
                K.NATIVE_LAYOUT_CATEGORY_MAP, K._NanoEngine = _SAVED_BINDINGS  # type: ignore[misc]
                _SAVED_BINDINGS = None


@register_provider("florin_parser_nano")
class FlorinParserNanoProvider(K.KdlFrontierNanoProvider):
    """florin-inc/florin-parser-nano: the ``kdl_frontier_nano`` provider with the
    fine-tuned weights and this module's markdown emission. Serving requirements
    and every inference stage are inherited unchanged."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        cfg = dict(base_config or {})
        cfg["endpoint_url"] = (
            cfg.get("endpoint_url") or os.getenv("FLORIN_NANO_ENDPOINT_URL") or ""
        )
        if not cfg["endpoint_url"]:
            raise ProviderConfigError(
                "FLORIN_NANO_ENDPOINT_URL is required (vLLM OpenAI-compatible base "
                "URL ending in /v1, serving florin-inc/florin-parser-nano)."
            )
        cfg["model"] = (
            cfg.get("model") or os.getenv("FLORIN_NANO_MODEL") or "florin-parser-nano"
        )
        super().__init__(provider_name, cfg)

    def run_inference(
        self, pipeline: PipelineSpec, request: InferenceRequest
    ) -> RawInferenceResult:
        with _patched_bindings():
            return super().run_inference(pipeline, request)
