"""OCRia-1 (llamaindex/OCRia-1): an open-weight 4B document parser served with vLLM, with optional layout boxes.

Transcription: every page is rendered at 300 DPI with pypdfium2 (CropBox, /Rotate applied) and sent, as an image-only
user turn after the model's own system prompt (`prompt.txt` from the model repository), to an OpenAI-compatible vLLM
endpoint. Greedy decoding, thinking off, up to 24,576 output tokens in a 32,768-token context. The raw answer is
post-processed into markdown by `ocria_postprocess` (wrapper and comment removal, page furniture unwrapped, HTML
emphasis to markdown, markdown tables to HTML).

Layout boxes (the visual grounding dimension) come from the `ocria_grounding` package shipped with the model: Docling's
Heron layout detector (Apache-2.0) gives the regions, and each word of the answer is placed by its PDF text-layer box
(or by the model's attention on pages without a text layer). That step needs a GPU with PyTorch and Transformers, so
it runs in a separate server that ships with the model (`examples/serve_boxes.py`); this provider sends it the same
page image the answer was generated from, the raw answer, and the page's text-layer words.

Serve the model (vLLM 0.30 or later) and the box server:

    vllm serve llamaindex/OCRia-1 --served-model-name ocria-1 --dtype bfloat16 \
      --max-model-len 32768 --limit-mm-per-prompt '{"image": 1}'
    huggingface-cli download llamaindex/OCRia-1 --local-dir OCRia-1
    python OCRia-1/examples/serve_boxes.py --port 8100        # optional: layout boxes

Then:

    OCRIA_ENDPOINT_URL=http://localhost:8000/v1 OCRIA_GROUNDING_URL=http://localhost:8100 \
    uv run parse-bench run ocria_1 --dataset parsebench-2

Config (env, or the pipeline config):
  OCRIA_ENDPOINT_URL    vLLM base URL ending in /v1                         (required)
  OCRIA_MODEL           served model name                                   (default ocria-1)
  OCRIA_GROUNDING_URL   box server base URL; unset = no layout boxes        (optional)
  OCRIA_GROUNDING       layout | all: box only layout pages (the only dimension that reads boxes) or every page
                        (default layout)
  OCRIA_PROMPT_FILE     local system prompt instead of the model repository's prompt.txt
  OCRIA_MAX_CONCURRENT  pages of one document sent in parallel             (default 4)
"""

from __future__ import annotations

import base64
import io
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

MODEL_REPO = "llamaindex/OCRia-1"
DPI = 300
MAX_TOKENS = 24576
MAX_MODEL_LEN = 32768
PAGE_SEPARATOR = "\n\n"
_IMG_FRAC = (0.6, 0.4, 0.25)  # image downscale per retry when the page alone overflows the context
_OVER_INPUT_RE = re.compile(r"max_tokens must be at least 1, got (-?\d+)")
_OVER_TOTAL_RE = re.compile(r"passed (\d+) input tokens and requested (\d+) output")


# ======================================================================
# Post-processing: OCRia-1's raw answer -> scored markdown
# ======================================================================

_PDFIUM_LOCK = threading.Lock()  # PDFium is not thread-safe; documents run on parallel threads

_TABLE_SPLIT = re.compile(r"(<table\b.*?</table>)", re.IGNORECASE | re.DOTALL)
_PAGE_TAG_RE = re.compile(r"</?page(?:_full)?_transcription\s*>", re.IGNORECASE)
_TSV_THEAD = "[thead]"
_TSV_COLSPAN_RE = re.compile(r"^\[colspan=(\d+)\]")
_TSV_ROWSPAN = "ROW_SPAN_CONTINUE"


def _strip_markdown_wrapper(text: str) -> str:
    text = re.sub(r"^```(?:markdown)?\s*\n", "", text)
    return re.sub(r"\n```\s*$", "", text)


def _emphasis_segment_to_markdown(s: str) -> str:
    """HTML emphasis/heading tags -> the markdown forms the formatting rules credit (tables handled by the caller)."""
    s = re.sub(
        r"<h([1-6])\b[^>]*>(.*?)</h\1>",
        lambda m: f"\n{'#' * int(m.group(1))} {m.group(2).strip()}\n",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(r"</?(?:strong|b)\b[^>]*>", "**", s, flags=re.IGNORECASE)
    s = re.sub(r"</?(?:em|i)\b[^>]*>", "*", s, flags=re.IGNORECASE)
    s = re.sub(r"</?(?:s|strike|del)\b[^>]*>", "~~", s, flags=re.IGNORECASE)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</(?:p|div)>", "\n", s, flags=re.IGNORECASE)
    return re.sub(r"<(?:p|div)\b[^>]*>", "\n", s, flags=re.IGNORECASE)


def _html_emphasis_to_markdown(text: str) -> str:
    if not text:
        return text
    return "".join(
        part if i % 2 else _emphasis_segment_to_markdown(part) for i, part in enumerate(_TABLE_SPLIT.split(text))
    )


def _tsv_cell(cell: str) -> tuple[str, int]:
    text = cell[len(_TSV_THEAD) :] if cell.startswith(_TSV_THEAD) else cell
    colspan = 1
    m = _TSV_COLSPAN_RE.match(text)
    if m:
        colspan, text = int(m.group(1)), text[m.end() :]
    if text.strip() == "EMPTY":
        text = ""
    return text.strip(), colspan


def _tsv_rows_to_html(rows: list[list[str]]) -> str:
    header_rows = [r for r in rows if r and r[0].startswith(_TSV_THEAD)]
    body_rows = [r for r in rows if not (r and r[0].startswith(_TSV_THEAD))]
    emitted: list[list[dict]] = [[] for _ in body_rows]
    owner: dict[int, dict] = {}
    for ri, cells in enumerate(body_rows):
        col = 0
        for c in cells:
            if c == _TSV_ROWSPAN:
                o = owner.get(col)
                if o is not None:
                    o["rowspan"] += 1
                    col += o["colspan"]
                else:
                    emitted[ri].append({"text": "", "colspan": 1, "rowspan": 1, "tag": "td"})
                    col += 1
                continue
            text, colspan = _tsv_cell(c)
            d = {"text": text, "colspan": colspan, "rowspan": 1, "tag": "td"}
            emitted[ri].append(d)
            owner[col] = d
            for k in range(col + 1, col + colspan):
                owner.pop(k, None)
            col += colspan

    def tag(d: dict) -> str:
        attrs = (f' colspan="{d["colspan"]}"' if d["colspan"] > 1 else "") + (
            f' rowspan="{d["rowspan"]}"' if d["rowspan"] > 1 else ""
        )
        return f"<{d['tag']}{attrs}>{d['text']}</{d['tag']}>"

    parts = ["<table>"]
    if header_rows:
        parts.append("<thead>")
        for cells in header_rows:
            row = []
            for c in cells:
                text, colspan = _tsv_cell(c)
                row.append(tag({"text": text, "colspan": colspan, "rowspan": 1, "tag": "th"}))
            parts.append("<tr>" + "".join(row) + "</tr>")
        parts.append("</thead>")
    parts.append("<tbody>")
    parts += ["<tr>" + "".join(tag(d) for d in row) + "</tr>" for row in emitted]
    parts += ["</tbody>", "</table>"]
    return "\n".join(parts)


def _tsv_tables_to_html(md: str) -> str:
    """```tsv blocks (closed, or unclosed up to the next blank line) -> HTML tables."""
    if not md or "```tsv" not in md:
        return md
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        if lines[i].strip() == "```tsv":
            close = next((k for k in range(i + 1, len(lines)) if lines[k].strip() == "```"), None)
            if close is not None:
                body, nxt = lines[i + 1 : close], close + 1
            else:
                start = i + 1
                while start < len(lines) and lines[start].strip() == "":
                    start += 1
                end = next((k for k in range(start, len(lines)) if lines[k].strip() == ""), len(lines))
                body, nxt = lines[start:end], end
            rows = [ln.split("\t") for ln in body if ln.strip() != ""]
            if rows:
                out.append(_tsv_rows_to_html(rows))
            i = nxt
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _sanitize_html_attributes(text: str) -> str:
    """Quote bare HTML attribute values (as QwenProvider._sanitize_html_attributes)."""

    def _quote_attrs(match: re.Match) -> str:
        return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', match.group(0))

    return re.sub(r"<[^>]+>", _quote_attrs, text)


def _convert_md_tables_to_html(content: str) -> str:
    """Markdown pipe tables -> HTML <table> (as QwenProvider._convert_md_tables_to_html, without its aiohttp import)."""
    import markdown2

    def flush(table_lines: list[str], out: list[str]) -> None:
        if len(table_lines) >= 2:
            html = markdown2.markdown("\n".join(table_lines), extras=["tables"]).strip()
            out.extend([html] if "<table>" in html.lower() else table_lines)
        else:
            out.extend(table_lines)

    out: list[str] = []
    table: list[str] = []
    in_table = False
    for line in content.split("\n"):
        if "|" in line and line.strip().startswith("|"):
            table = [line] if not in_table else table + [line]
            in_table = True
        else:
            if in_table:
                flush(table, out)
                table, in_table = [], False
            out.append(line)
    if in_table:
        flush(table, out)
    return "\n".join(out)


def ocria_postprocess(raw: str) -> str:
    """OCRia-1's raw answer -> the markdown that is scored (the model card's post-processing)."""
    s = re.sub(r"<!--.*?-->", "", raw or "", flags=re.S).strip()  # analysis comments
    s = re.sub(r"<!--(?:(?!-->).)*$", "", s, flags=re.S).strip()  # an unclosed runaway comment
    s = _PAGE_TAG_RE.sub("", s).strip()  # <page_full_transcription> / <page_transcription> wrappers
    # page furniture: drop the tags, keep the text inside
    s = re.sub(r"</?(?:page_header|page_footer|page_number|watermark)\s*>", " ", s, flags=re.IGNORECASE)
    s = _tsv_tables_to_html(s)
    s = re.sub(r"(?:&nbsp;[ \t]*)+", " ", s)  # runaway &nbsp; padding
    s = re.sub(r"(?:\.[ \t]*){6,}", " ", s)  # runaway dot leaders
    s = _strip_markdown_wrapper(s).strip()  # the ```markdown page wrapper
    s = _html_emphasis_to_markdown(s)
    s = re.sub(r"<mark\b[^>]*>", "<mark>", s, flags=re.IGNORECASE)  # bare <mark>
    s = re.sub(r"<(page_header|page_footer|watermark)>.*?</\1\s*>\s*", "", s, flags=re.S | re.I)
    s = re.sub(r"</?page_number\s*>", "", s, flags=re.I)
    s = re.sub(r"!\[[^\]\n]{0,200}\]", "", s)  # image references
    s = re.sub(r"```mermaid\s*\n.*?\n```\s*", "", s, flags=re.S)  # diagrams drawn as mermaid
    s = re.sub(r"^[ \t]*<hr\s*/?>[ \t]*$\n?", "", s, flags=re.M | re.I)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if s:
        s = _convert_md_tables_to_html(s)
        s = _sanitize_html_attributes(s)
    return s


# ======================================================================
# Pages: rendering and text-layer words in the same pixel frame
# ======================================================================


def _to_image(left: float, b: float, r: float, t: float, crop, rot: int, s: float) -> list[float]:
    cx0, cy0, cx1, cy1 = crop

    def m(ux: float, uy: float) -> tuple[float, float]:
        return {
            0: (ux - cx0, cy1 - uy),
            90: (uy - cy0, ux - cx0),
            180: (cx1 - ux, uy - cy0),
            270: (cy1 - uy, cx1 - ux),
        }[rot]

    xs, ys = zip(m(left, b), m(r, t), strict=True)
    return [min(xs) * s, min(ys) * s, max(xs) * s, max(ys) * s]


def page_words(page, dpi: int = DPI) -> list[dict]:
    """Text-layer words of a pypdfium2 page in pixels of its `page.render(scale=dpi / 72)` image (the box server's
    `ocria_grounding.pdf.page_words`): characters joined while the gap is under 0.35 line heights on one line."""
    s, rot, crop = dpi / 72, page.get_rotation() % 360, page.get_cropbox()
    tp = page.get_textpage()
    out: list[dict] = []
    cur: dict | None = None
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch or ch.isspace() or ch in "\r\n\x00￾":
            if cur:
                out.append(cur)
                cur = None
            continue
        box = _to_image(*tp.get_charbox(i), crop, rot, s)
        hgt = box[3] - box[1]
        if cur and box[0] - cur["box"][2] < 0.35 * hgt and abs(box[1] - cur["box"][1]) < 0.5 * hgt:
            cur["text"] += ch
            cb = cur["box"]
            cur["box"] = [min(cb[0], box[0]), min(cb[1], box[1]), max(cb[2], box[2]), max(cb[3], box[3])]
        else:
            if cur:
                out.append(cur)
            cur = {"text": ch, "box": box}
    if cur:
        out.append(cur)
    return out


def _png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _downscale_b64(b64: str, factor: float) -> str:
    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im = im.resize((max(1, int(im.width * factor)), max(1, int(im.height * factor))), Image.LANCZOS)
    return _png_b64(im)


@lru_cache(maxsize=4)
def _system_prompt(path: str | None) -> str:
    if path:
        return Path(path).read_text()
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MODEL_REPO, "prompt.txt")).read_text()


@register_provider("ocria_1")
class OCRia1Provider(Provider):
    """OCRia-1 through a vLLM endpoint; layout boxes from the model's box server when configured."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        cfg = self.base_config
        self._endpoint_url = (cfg.get("endpoint_url") or os.getenv("OCRIA_ENDPOINT_URL") or "").rstrip("/")
        if not self._endpoint_url:
            raise ProviderConfigError(
                f"OCRIA_ENDPOINT_URL is required (vLLM OpenAI-compatible base URL ending in /v1, serving {MODEL_REPO})."
            )
        self._model = cfg.get("model") or os.getenv("OCRIA_MODEL") or "ocria-1"
        self._grounding_url = (cfg.get("grounding_url") or os.getenv("OCRIA_GROUNDING_URL") or "").rstrip("/")
        self._grounding = (cfg.get("grounding") or os.getenv("OCRIA_GROUNDING") or "layout").lower()
        self._prompt_file = cfg.get("prompt_file") or os.getenv("OCRIA_PROMPT_FILE") or None
        self._dpi = int(cfg.get("dpi", DPI))
        self._max_tokens = int(cfg.get("max_tokens", MAX_TOKENS))
        self._max_model_len = int(cfg.get("max_model_len", MAX_MODEL_LEN))
        self._timeout = float(cfg.get("timeout", 1800))
        self._max_pages = int(cfg.get("max_pages", 400))
        self._max_concurrent = int(os.getenv("OCRIA_MAX_CONCURRENT", cfg.get("max_concurrent", 4)))
        self._api_key = cfg.get("api_key") or os.getenv(cfg.get("api_key_env", "VLLM_API_KEY"), "") or "EMPTY"

    # ------------------------------------------------------------------ rendering

    def _pages(self, source_path: Path) -> list[tuple[Any, list[dict] | None]]:
        """(PIL image, text-layer words or None) per page."""
        from PIL import Image

        if source_path.suffix.lower() != ".pdf":
            return [(Image.open(source_path).convert("RGB"), None)]
        import pypdfium2 as pdfium

        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(source_path))
            try:
                if len(pdf) > self._max_pages:
                    raise ProviderPermanentError(f"Document has {len(pdf)} pages > max_pages={self._max_pages}.")
                out = []
                for i in range(len(pdf)):
                    pg = pdf[i]
                    out.append((pg.render(scale=self._dpi / 72).to_pil().convert("RGB"), page_words(pg, self._dpi)))
            finally:
                pdf.close()
        return out

    # ------------------------------------------------------------------ transcription

    def _transcribe(self, client, b64: str) -> dict:
        """One page -> {answer, finish_reason, completion_tokens, max_tokens}. A prompt that overflows the context
        first shrinks the output budget, then the image."""
        system = _system_prompt(self._prompt_file)
        mt, cur = self._max_tokens, b64
        for attempt in range(4):
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + cur}}],
                },
            ]
            try:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=mt,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as e:  # vLLM 400 when the image + output budget exceed the context
                msg = str(e)
                m_total = _OVER_TOTAL_RE.search(msg)
                if m_total and attempt < 3:
                    inp = int(m_total.group(1))
                    if self._max_model_len - int(inp * 1.5) < 512:
                        cur = _downscale_b64(b64, _IMG_FRAC[attempt])
                    mt = max(256, min(mt // 2, self._max_model_len - int(inp * 1.5)))
                    continue
                if _OVER_INPUT_RE.search(msg) and attempt < 3:
                    cur = _downscale_b64(b64, _IMG_FRAC[attempt])
                    mt = min(self._max_tokens, 2048)
                    continue
                raise
            usage = getattr(resp, "usage", None)
            return {
                "answer": resp.choices[0].message.content or "",
                "finish_reason": resp.choices[0].finish_reason,
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "max_tokens": mt,
            }
        raise ProviderTransientError("page did not fit the context after 4 attempts")

    # ------------------------------------------------------------------ boxes

    def _wants_boxes(self, request: InferenceRequest) -> bool:
        if not self._grounding_url:
            return False
        if self._grounding == "all":
            return True
        return str(request.example_id).split("/", 1)[0] == "layout"

    def _boxes(self, b64: str, answer: str, words: list[dict] | None) -> dict:
        r = httpx.post(
            self._grounding_url + "/ground",
            json={"image_png_b64": b64, "answer": answer, "words": words},
            timeout=self._timeout,
        )
        if r.status_code >= 500:
            raise ProviderTransientError(f"box server {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ provider API

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")
        started_at = datetime.now()
        try:
            pages = self._pages(source_path)
        except ProviderPermanentError:
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Failed to render document: {e}") from e
        if not pages:
            raise ProviderPermanentError("Document rendered to zero pages.")

        from openai import OpenAI

        client = OpenAI(base_url=self._endpoint_url, api_key=self._api_key, timeout=self._timeout, max_retries=3)
        boxes = self._wants_boxes(request)

        def one(page) -> dict:
            image, words = page
            b64 = _png_b64(image)
            gen = self._transcribe(client, b64)
            rec = {**gen, "width_px": image.width, "height_px": image.height}
            if boxes:
                rec["layout"] = self._boxes(b64, gen["answer"], words)
            return rec

        try:
            with ThreadPoolExecutor(max_workers=max(1, min(self._max_concurrent, len(pages)))) as ex:
                recs = list(ex.map(one, pages))
        except (ProviderPermanentError, ProviderTransientError):
            raise
        except Exception as e:
            raise ProviderTransientError(f"OCRia-1 inference failed: {e}") from e
        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"dpi": self._dpi, "pages": recs},
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError("OCRia1Provider only supports PARSE.")
        raw = raw_result.raw_output
        recs = raw.get("pages") or []
        scale = 72.0 / float(raw.get("dpi") or DPI)  # pixels -> PDF points
        pages = [PageIR(page_index=i, markdown=ocria_postprocess(r.get("answer", ""))) for i, r in enumerate(recs)]
        # the document's markdown post-processes the joined raw answers (one wrapper per document, as scored)
        markdown = ocria_postprocess(PAGE_SEPARATOR.join(r.get("answer", "") for r in recs))

        layout_pages: list[ParseLayoutPageIR] = []
        for i, r in enumerate(recs):
            lay = r.get("layout")
            if not lay:
                continue
            items = []
            for it in lay.get("items") or []:
                x0, y0, x1, y1 = (float(v) * scale for v in it["box"])
                if x1 <= x0 or y1 <= y0:
                    continue
                seg = LayoutSegmentIR(x=x0, y=y0, w=x1 - x0, h=y1 - y0, label=it.get("label"))
                md = it.get("md", "")
                items.append(
                    LayoutItemIR(type=it.get("type", "Text"), md=md, value=md, bbox=seg, layout_segments=[seg])
                )
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=i + 1,
                    width=float(lay.get("width") or r.get("width_px", 0)) * scale,
                    height=float(lay.get("height") or r.get("height_px", 0)) * scale,
                    md=pages[i].markdown,
                    items=items,
                )
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=markdown,
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
