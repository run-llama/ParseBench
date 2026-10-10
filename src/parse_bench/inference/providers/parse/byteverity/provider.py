"""synthesa-codex: Codex CLI VLM proposer + deterministic PDF floor + sealed routing oracle.

The VLM (driven headless through ``codex exec`` on a rendered page image) only
*proposes* a layout-annotated transcription. Everything the PDF can answer
deterministically — word geometry, font flags, the text layer itself — is
taken from PyMuPDF, and a sealed synthesa-decide oracle decides per page which
source to trust. The model is never the authority on anything the file proves.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import (
    SYSTEM_PROMPT_LAYOUT,
    USER_PROMPT_LAYOUT,
    build_layout_pages,
    items_to_markdown,
    parse_layout_blocks,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

from . import floor

EXTRA_RULES = (
    "\n\nAdditional rules (important):\n"
    "- Keep visible text formatting inline: **bold**, ~~strikethrough~~, "
    "<sup>superscript</sup>, <sub>subscript</sub>. Label the document title 'Title' and "
    "section/sub-section headings 'Section-header' (not Text).\n"
    "- Running headers/footers and page numbers: label them Page-header / Page-footer.\n"
    "- Tables: always HTML with the header row(s) in <thead> using <th>; use rowspan/colspan "
    "for merged cells; one value per cell; do not drop rows.\n"
    "- Charts/graphs: inside the chart's Picture div, give the chart title, then an HTML "
    "<table> of its data: <thead> whose first <th> names the category axis and then one <th> "
    "per series using the exact legend text; then one <tr> per category with the category "
    "label exactly as printed, and one plain number per cell (no units, no % sign, no "
    "thousands separators). Use printed data labels exactly when present; otherwise estimate "
    "each value carefully from the gridlines. Include every category and every series; for a "
    "single-series chart use the value-axis title (or 'Value') as the series header.\n"
    "- Transcribe every word on the page exactly, in reading order; never summarize.\n"
)

CODEX_PREAMBLE = (
    "You are acting purely as a vision transcription model. Do NOT run any shell "
    "commands or tools; just look at the attached page image and answer.\n\n"
)


_RB = ""
ORACLE_DIR = "page_route"
_ORACLE_LOCK = threading.Lock()
_USAGE = threading.local()


def _usage_log() -> list:
    if not hasattr(_USAGE, "calls"):
        _USAGE.calls = []
    return _USAGE.calls


def _usage_summary(num_pages: int) -> dict[str, Any]:
    calls = list(_usage_log())
    total = sum(c.get("cost_usd") or 0.0 for c in calls)
    return {
        "usage_calls": calls,
        "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
        "thinking_tokens": sum(c.get("reasoning_tokens", 0) for c in calls),
        "cost_usd": total,
        "cost_per_page_usd": total / num_pages if num_pages else 0.0,
    }


_ORACLE_CLIENT: Any = None


class _TableDecide:
    """Engine-free evaluator for the provider-level oracles (page_route, chart_repair_gate)."""

    def evaluate(self, oracle: str, facts: dict[str, Any]) -> dict[str, Any]:
        from .tables import table

        return table(os.path.basename(os.path.normpath(oracle))).evaluate(facts)


def _oracle() -> Any:
    global _ORACLE_CLIENT
    with _ORACLE_LOCK:
        if _ORACLE_CLIENT is None:
            _ORACLE_CLIENT = _TableDecide()
    return _ORACLE_CLIENT


@register_provider("byteverity")
class ByteVerityProvider(Provider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._model = self.base_config.get("model", "gpt-6-luna")
        self._effort = self.base_config.get("effort", "low")
        self._dpi = int(self.base_config.get("dpi", 150))
        self._timeout = int(self.base_config.get("timeout", 600))
        self._retries = int(self.base_config.get("retries", 2))
        self._stage = self.base_config.get("stage", "full")
        self._transport = self.base_config.get("transport", os.environ.get("BYTEVERITY_TRANSPORT", "codex"))
        self._codex = None
        if self._transport == "codex":  # the Codex CLI is needed only for the codex transport
            self._codex = shutil.which("codex") or os.path.expanduser("~/.local/bin/codex")
            if not os.path.exists(self._codex):
                raise ProviderConfigError("transport=codex needs the codex CLI on PATH (or use transport=openai)")
        self._api = None
        self._escalate_model = self.base_config.get("escalate_model", "gpt-6-sol")

    # -- sealed routing oracle -------------------------------------------------
    def _route(self, facts: dict[str, Any]) -> dict[str, Any]:
        return _oracle().evaluate(ORACLE_DIR, facts)

    def _full_page(self, page: Any, i: int, img: str, w: int, h: int) -> dict[str, Any]:
        """Propose (VLM) -> measure (PDF floor) -> dispose (sealed oracle), at most one escalation."""
        words = floor.page_words(page)
        tl = floor.text_layer_class(words, page)
        dets = None
        if os.environ.get("SX_LAYOUT_LIVE", "1") == "1":
            from .layout import detect_page

            dets = detect_page(page)
        if self._model == "none":  # byteverity_parse_novlm: no proposer at all
            route = "fallback_textlayer" if tl != "none" else "give_up_empty"
            return {
                "page_index": i,
                "raw_content": "",
                "items": [],
                "width": w,
                "height": h,
                "route": route,
                "trail": [{"model": "none", "route": route}],
                "text_layer": tl,
                "layout_dets": dets,
            }
        prompt = CODEX_PREAMBLE + SYSTEM_PROMPT_LAYOUT + EXTRA_RULES + "\n\n" + USER_PROMPT_LAYOUT
        trail = []
        escalated = False
        model = self._model
        while True:
            try:
                raw = self._codex_call(img, prompt, model=model)
            except ProviderTransientError:
                raw = ""
            items = parse_layout_blocks(raw) if raw else []
            status = "empty" if not raw.strip() else ("ok" if items else "unparsed")
            vlm_text = " ".join(it.get("text", "") for it in items) if items else raw
            rc = floor.recall_class(words, vlm_text) if tl == "full" else "na"
            facts = {"vlm_status": status, "text_layer": tl, "recall": rc, "escalated": escalated}
            d = self._route(facts)
            route = (d.get("decision") or {}).get("route", "HOLD") if d.get("status") == "DECIDED" else "HOLD"
            trail.append({"model": model, "facts": facts, "route": route, "result_digest": d.get("result_digest")})
            if route == "escalate" and not escalated:
                escalated, model = True, self._escalate_model
                continue
            break
        page_out = {
            "page_index": i,
            "raw_content": raw,
            "items": items,
            "width": w,
            "height": h,
            "route": route,
            "trail": trail,
            "text_layer": tl,
        }
        if tl != "full" and items and os.environ.get("SX_EMPHASIS_LIVE", "1") == "1":
            page_out["emphasis"] = self._emphasis_query(img, items)
        if items and os.environ.get("SX_TABLE_ZOOM_LIVE", "1") == "1":
            page_out["table_requery"] = self._table_zoom(page, items)
        page_out["layout_dets"] = dets
        if self.base_config.get("chart_repair") and tl == "full" and items:
            self._chart_repair(page, img, page_out)
        return page_out

    def _chart_repair(self, page: Any, img: str, pd: dict[str, Any]) -> None:
        """Residual chart repair (hybrid): re-ask the escalation model when printed chart text is unexplained;
        the sealed chart_repair_gate accepts only a strictly better residual band."""
        from .chart_repair import band, repair_note, residual

        r0, miss = residual(page, pd["items"])
        if band(r0) not in ("high", "no_table"):
            return
        prompt = CODEX_PREAMBLE + SYSTEM_PROMPT_LAYOUT + EXTRA_RULES + repair_note(miss) + "\n\n" + USER_PROMPT_LAYOUT
        try:
            out = self._codex_call(img, prompt, model=self._escalate_model)
        except ProviderTransientError:
            return
        items = parse_layout_blocks(out)
        r1 = residual(page, items)[0] if items else None
        facts = {"before": band(r0), "after": band(r1)}
        d = _oracle().evaluate(os.path.join(_RB, "chart_repair_gate"), facts)
        verdict = (d.get("decision") or {}).get("verdict", "keep_original")
        pd["repair"] = {"facts": facts, "verdict": verdict, "result_digest": d.get("result_digest"), "r0": r0, "r1": r1}
        if verdict == "accept_repair":
            pd["raw_content"], pd["items"] = out, items

    def _table_zoom(self, page: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Zoom pass (proposer model, effort high).

        Re-asks each of the proposer's own tables on a high-dpi crop; O13 decides admission.
        """
        import pymupdf

        from .prompts_table import PROMPT as _TPROMPT

        out = []
        W, H = page.rect.width, page.rect.height
        for k, it in enumerate(items):
            bb = it.get("bbox")
            if (it.get("label") or "").lower() != "table" or not (isinstance(bb, list) and len(bb) == 4):
                continue
            clip = (
                pymupdf.Rect(bb[0] / 1000 * W - 15, bb[1] / 1000 * H - 15, bb[2] / 1000 * W + 15, bb[3] / 1000 * H + 15)
                & page.rect
            )
            if clip.is_empty or clip.width < 20 or clip.height < 10:
                continue
            with tempfile.TemporaryDirectory(prefix="sx_tz_") as td:
                img = os.path.join(td, "t.png")
                z = max(2.0, min(4.0, 1600 / max(clip.width, 1)))
                page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=clip).save(img)
                try:
                    html = self._codex_call(
                        img, _TPROMPT, model=self._model, effort=self.base_config.get("zoom_effort", "high")
                    )
                except ProviderTransientError:
                    continue
            m = re.search(r"<table[\s\S]*</table>", html, re.I)
            if m:
                out.append({"item": k, "html": m.group(0)})
        return out

    def _emphasis_query(self, img: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Residual query (proposer model only): which of the proposer's own lines are headings / bold. O11 decides."""
        import json as _json

        from .prompts_emphasis import PROMPT, lines_of

        lines = lines_of(_sx_markdown([dict(it, text=_prose_breaks(it.get("text", ""))) for it in items]))
        if not lines:
            return {"headings": [], "bold": []}
        try:
            out = self._codex_call(
                img, PROMPT + "\n".join(f"{k + 1}: {li}" for k, li in enumerate(lines)), model=self._model
            )
        except ProviderTransientError as e:
            return {"error": str(e)[:200]}
        m = re.search(r"\{[\s\S]*\}", out)
        try:
            js = _json.loads(m.group(0)) if m else {}
        except Exception:
            return {"error": "unparseable"}
        heads = [
            {"text": lines[h["line"] - 1], "level": int(h.get("level", 2))}
            for h in js.get("headings", [])
            if isinstance(h, dict) and isinstance(h.get("line"), int) and 1 <= h["line"] <= len(lines)
        ]
        bolds = [
            {"text": b["text"]}
            for b in js.get("bold", [])
            if isinstance(b, dict) and isinstance(b.get("text"), str) and b["text"].strip()
        ]
        return {"headings": heads, "bold": bolds}

    # -- VLM proposer ---------------------------------------------------------
    def _codex_call(self, image_path: str, prompt: str, model: str | None = None, effort: str | None = None) -> str:
        model = model or self._model
        effort = effort or self._effort
        if self._transport == "openai":
            if self._api is None:
                from .transport import OpenAITransport

                self._api = OpenAITransport(
                    max_tokens=int(self.base_config.get("max_tokens", 32768)),
                    base_url=self.base_config.get("base_url"),
                    api_key_env=self.base_config.get("api_key_env", "OPENAI_API_KEY"),
                    effort_style=self.base_config.get("effort_style", "openai"),
                )
            last: Exception | None = None
            for attempt in range(self._retries + 1):
                try:
                    text, usage = self._api.call(image_path, prompt, model, effort)
                    _usage_log().append(usage)
                    if text.strip():
                        return text
                except Exception as e:  # network / rate limit: retry, then transient
                    last = e
                time.sleep(5 * (attempt + 1))
            raise ProviderTransientError(f"openai call failed: {last}")
        last_err = ""
        for attempt in range(self._retries + 1):
            with tempfile.TemporaryDirectory(prefix="sx_codex_") as wd:
                out = os.path.join(wd, "out.md")
                cmd = [
                    self._codex,
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "-s",
                    "read-only",
                    "-C",
                    wd,
                    "-m",
                    model,
                    "-c",
                    f"model_reasoning_effort={effort}",
                    "-i",
                    image_path,
                    "-o",
                    out,
                    prompt,
                ]
                try:
                    p = subprocess.run(
                        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=self._timeout
                    )
                except subprocess.TimeoutExpired:
                    last_err = "timeout"
                    continue
                if p.returncode == 0 and os.path.exists(out):
                    txt = Path(out).read_text()
                    if txt.strip():
                        return txt
                last_err = (p.stderr or p.stdout)[-800:]
            time.sleep(5 * (attempt + 1))
        raise ProviderTransientError(f"codex exec failed: {last_err}")

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"unsupported product {request.product_type}")
        import pymupdf

        started_at = datetime.now()
        _USAGE.calls = []
        src = Path(request.source_file_path)
        doc = pymupdf.open(str(src))
        pages = []
        with tempfile.TemporaryDirectory(prefix="sx_pages_") as td:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=self._dpi)
                img = os.path.join(td, f"p{i}.png")
                pix.save(img)
                if self._stage == "baseline":
                    prompt = CODEX_PREAMBLE + SYSTEM_PROMPT_LAYOUT + "\n\n" + USER_PROMPT_LAYOUT
                    raw = self._codex_call(img, prompt)
                    pages.append(
                        {
                            "page_index": i,
                            "raw_content": raw,
                            "items": parse_layout_blocks(raw),
                            "width": pix.width,
                            "height": pix.height,
                        }
                    )
                    continue
                pages.append(self._full_page(page, i, img, pix.width, pix.height))
        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "pages": pages,
                "num_pages": len(pages),
                "model": self._model,
                "config": dict(self.base_config),
                "transport": self._transport,
                **_usage_summary(len(pages)),
            },
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        pages: list[PageIR] = []
        mds: list[str] = []
        layout_pages: list[ParseLayoutPageIR] = []
        doc = None
        stage = (raw_result.raw_output.get("config") or {}).get("stage", "baseline")
        if stage != "baseline":
            try:
                import pymupdf

                doc = pymupdf.open(str(raw_result.request.source_file_path))
            except Exception:
                doc = None
        floor_stats: list[dict[str, Any]] = []
        for pd in raw_result.raw_output.get("pages", []):
            idx = pd.get("page_index", 0)
            items = pd.get("items", [])
            ground = None
            if stage != "baseline":
                items = [dict(it, text=_prose_breaks(it.get("text", ""))) for it in items]
            if doc is not None and idx < len(doc):
                pd["proposer"] = "none" if (raw_result.raw_output.get("model") == "none") else "vlm"
                pd["_example_id"] = raw_result.request.example_id
                from .rules import rules as _rules

                R = _rules()
                if R is not None and os.environ.get("SX_O0", "1") == "1":
                    vf = floor.text_layer_validity_facts(doc[idx])
                    pd["authority"] = R.decide("text_layer_validity", **vf)
                    pd["validity_facts"] = vf
                items, st = apply_floor(doc[idx], pd, items)
                st["authority"] = pd.get("authority")
                geo_ok = (
                    (pd["authority"] in ("full_authority", "geometry_only"))
                    if "authority" in pd
                    else pd.get("text_layer") == "full"
                )
                if not geo_ok and R is not None and pd.get("authority") and items:
                    mask, inkband = floor.page_ink(doc[idx])
                    if R.decide("pixel_floor", authority=pd["authority"], ink=inkband) == "use":
                        ground = floor.raster_segments(doc[idx], items, mask, R)
                        st["pixel_floor"] = len(ground)
                if _SX_TOGGLES["segground"] and geo_ok:
                    segs = floor.pdf_segments(doc[idx])
                    if segs:
                        base = (
                            floor.reconcile_pictures(items, floor.pdf_images(doc[idx]))
                            if _SX_TOGGLES["images"]
                            else items
                        )
                        if _SX_TOGGLES["ink"]:
                            comps = floor.ink_components(doc[idx])
                            if os.environ.get("SX_INKMULTI", "1") == "1":  # admitted by ratchet R7
                                fine = floor.ink_components(doc[idx], gap=1.5)
                                comps = comps + [c for c in fine if all(floor._iou(c, d) < 0.9 for d in comps)]
                            base = floor.reconcile_ink(base, comps, segs)
                        if os.environ.get("SX_SEGUNION", "0") == "1":
                            segs = segs + floor.merged_segments(segs)
                        ground = floor.grounding_items(base, segs)
                        st["ground_items"] = len(ground)
                floor_stats.append(st)
            if stage == "baseline":
                md = items_to_markdown(items) if items else pd.get("raw_content", "")
            else:
                md = _sx_markdown(items) if items else pd.get("raw_content", "")
            le_file = os.environ.get("SX_LAYOUT_EVIDENCE", "")
            if (
                (pd.get("layout_dets") or (le_file and os.path.exists(le_file)))
                and doc is not None
                and idx < len(doc)
                and pd.get("authority")
            ):
                src = str(raw_result.request.source_file_path)
                key = "docs/" + src.split("/docs/")[-1] if "/docs/" in src else src
                dets = pd.get("layout_dets") or _load_claims(le_file).get(key, {}).get(str(idx))
                if dets and R is not None:
                    n_ok = sum(1 for d in dets if d["score"] >= 0.5)
                    src_ = R.decide(
                        "layout_source",
                        detector="none" if n_ok == 0 else "sparse" if n_ok < 3 else "normal",
                        authority=pd["authority"],
                    )
                    if src_ == "use_detector":
                        dg = floor.detector_items(doc[idx], dets, items, pd["authority"])
                        if dg:
                            ground = dg
            layout_pages.extend(
                build_layout_pages(
                    ground if ground else items,
                    pd.get("width", 0),
                    pd.get("height", 0),
                    md,
                    page_number=idx + 1,
                    bbox_scale=1000,
                )
            )
            pages.append(PageIR(page_index=idx, markdown=md))
            mds.append(md)
        # O11 residual emphasis: claims recorded at inference (live), else a sidecar for frozen-output replays
        from .rules import rules as _rules

        R = _rules()
        claim = (raw_result.raw_output.get("pages") or [{}])[0].get("emphasis")
        claims_file = os.environ.get("SX_EMPHASIS")
        if claim is None and claims_file and os.path.exists(claims_file):
            claim = _load_claims(claims_file).get(raw_result.request.example_id)
        no_vlm = raw_result.raw_output.get("model") == "none"  # claims come from the proposer: never on the no-VLM arm
        if R is not None and claim and "error" not in claim and mds and not no_vlm:
            mds[0], est = _apply_emphasis(mds[0], claim, R)
            pages = [
                PageIR(
                    page_index=p.page_index, markdown=(mds[0] if p.page_index == pages[0].page_index else p.markdown)
                )
                for p in pages
            ]
            raw_result.raw_output["emphasis_stats"] = est
        if os.environ.get("SX_BOLDSEAM", "1") == "1":  # admitted by ratchet v2 (F2)
            pages = [PageIR(page_index=p.page_index, markdown=_BOLD_SEAM_RE.sub(r"\1", p.markdown)) for p in pages]
            mds = [_BOLD_SEAM_RE.sub(r"\1", m) for m in mds]
        if floor_stats:
            raw_result.raw_output["floor_stats"] = floor_stats
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown="\n\n".join(mds),
            layout_pages=layout_pages,
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


_SX_TOGGLES = {
    k: os.environ.get(f"SX_{k.upper()}", "1") == "1" for k in ("snap", "markup", "fallback", "segground", "images")
}
_SX_TOGGLES["chartsnap"] = os.environ.get("SX_CHARTSNAP", "0") == "1"
_SX_TOGGLES["ink"] = os.environ.get("SX_INK", "1") == "1"  # admitted by ratchet R7
_SX_TOGGLES["charttitle"] = os.environ.get("SX_CHARTTITLE", "1") == "1"  # admitted by ratchet R4


def _bold_chart_titles(text: str) -> tuple[str, int]:
    """Make the caption line preceding each chart table a bold line, so the scorer binds it as title context."""
    parts = re.split(r"(<table[\s\S]*?</table>)", text, flags=re.IGNORECASE)
    n = 0
    for i in range(0, len(parts), 2):
        if i + 1 >= len(parts):
            break
        lines = parts[i].rstrip().split("\n")
        for j in range(len(lines) - 1, -1, -1):
            ln = lines[j].strip()
            if not ln:
                continue
            from .rules import rules as _rules

            R = _rules()
            if R is not None:
                ok = R.decide("chart_title", has_markup=ln.startswith(("**", "#", "<")), long=len(ln) > 200) == "bold"
            else:
                ok = not ln.startswith(("**", "#", "<")) and len(ln) <= 200
            if ok:
                lines[j] = f"**{ln}**"
                n += 1
            break
        parts[i] = "\n".join(lines) + "\n\n"
    return "".join(parts), n


def _textlayer_items(page: Any) -> list[dict[str, Any]]:
    """Blocks straight from the PDF text layer, on the 0-1000 grid (fallback proposer)."""
    W, H = page.rect.width, page.rect.height
    items = []
    for x0, y0, x1, y1, txt, _bn, btype in page.get_text("blocks", sort=True):
        if btype != 0 or not txt.strip():
            continue
        items.append(
            {
                "bbox": [x0 / W * 1000, y0 / H * 1000, x1 / W * 1000, y1 / H * 1000],
                "label": "Text",
                "text": " ".join(txt.split()),
            }
        )
    return items


def apply_floor(
    page: Any, pd: dict[str, Any], items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute the sealed oracle's route for one page using PDF-proven facts."""
    route = pd.get("route", "accept_snap")
    st: dict[str, Any] = {"route": route}
    if route in ("fallback_textlayer", "accept_markdown_textlayer_boxes") and _SX_TOGGLES["fallback"]:
        if route == "fallback_textlayer" or not items:
            items = _textlayer_items(page)
    if route in ("accept_snap", "accept_markdown_textlayer_boxes") and _SX_TOGGLES["snap"] and items:
        words = floor.page_words(page)
        items, n = floor.snap_items(items, words)
        st["snapped"] = n
    if _SX_TOGGLES["chartsnap"] and items and pd.get("text_layer", "none") != "none":
        from . import chart

        pics = [
            it
            for it in items
            if (it.get("label") or "").lower() in ("picture", "figure") and "<table" in it.get("text", "")
        ]
        if pics:
            axes = chart.calibrate_axes(page)
            cands = chart.candidate_values(page, axes)
            printed = chart.printed_numbers(page)
            k = 0
            for it in pics:
                it["text"], n = chart.refine_table_values(it["text"], cands, printed)
                k += n
            st["chartsnap"] = k
    from .rules import rules as _rules

    R = _rules()
    if R is not None and items and pd.get("authority"):
        owned, orphans = floor.word_ownership(page, items)
        n_pdf = 0
        for k, it in enumerate(items):
            lab = it.get("label") or ""
            if lab.lower() in floor._OWNER_LABELS_SELF:
                continue
            ptxt = floor.words_to_text(owned.get(k, []))
            src = R.decide(
                "content_source",
                authority=pd["authority"],
                label=floor.label_class(lab),
                agreement=floor.agreement_band(it.get("text", ""), ptxt),
            )
            if src == "pdf":
                it["text"] = ptxt
                n_pdf += 1
        st["content_pdf_blocks"] = n_pdf
        if os.environ.get("SX_ORPHANS", "0") == "1" and pd["authority"] == "full_authority" and orphans:
            n0 = len(items)
            items[:] = floor.orphan_items(orphans, items)
            st["orphan_blocks"] = len(items) - n0
    tq = pd.get("table_requery")
    tq_file = os.environ.get("SX_TABLE_REQ")
    if tq is None and tq_file and os.path.exists(tq_file):
        tq = _load_claims(tq_file).get(pd.get("_example_id", ""))
    if R is not None and tq and pd.get("proposer", "vlm") == "vlm":
        items[:], qst = _apply_table_requery(page, pd, items, tq, R)
        st["table_requery"] = qst
    if R is not None and pd.get("authority") and os.environ.get("SX_O12", "1") == "1":
        items[:], tst = _apply_tables(page, pd, items, R, proposer=pd.get("proposer", "vlm"))
        if tst:
            st["tables"] = tst
    if _SX_TOGGLES["charttitle"] and items:
        k = 0
        for it in items:
            if (it.get("label") or "").lower() in ("picture", "figure") and "<table" in it.get("text", ""):
                it["text"], n = _bold_chart_titles(it["text"])
                k += n
        st["charttitle"] = k
    style_ok = (
        (pd["authority"] in ("full_authority", "style_only"))
        if "authority" in pd
        else pd.get("text_layer", "none") != "none"
    )
    if _SX_TOGGLES["markup"] and items and style_ok:
        runs = floor.styled_runs(page)
        items, ms = floor.inject_markup(items, runs)
        st["markup"] = ms
    return items, st


_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_CONNECTOR_RE = r"(?i)\b(the|of|and|for|in|to|a|an|on|at|by|with|&|de|la|le|les|des|du|del|der|die|das|und|y|e|et)$"
_BOLD_SEAM_RE = re.compile(r"\*\*([ \t]+)\*\*")
_TABLE_CHUNK_RE = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)


_STRONG_RE = re.compile(r"<(strong|b)>\s*(.*?)\s*</\1>", re.IGNORECASE | re.DOTALL)
_EM_RE = re.compile(r"<(em|i)>\s*(.*?)\s*</\1>", re.IGNORECASE | re.DOTALL)
_DOUBLE_BOLD_RE = re.compile(r"\*\*\*\*(.*?)\*\*\*\*", re.DOTALL)


def _clean_prose_chunk(p: str) -> str:
    import html

    p = _BR_RE.sub("\n", p)
    p = _STRONG_RE.sub(lambda m: m.group(2) if m.group(2).startswith("**") else f"**{m.group(2)}**", p)
    p = _EM_RE.sub(lambda m: f"*{m.group(2)}*", p)
    p = _DOUBLE_BOLD_RE.sub(r"**\1**", p)
    return (
        html.unescape(p)
        if "&" in p
        and "<" not in p.replace("<sup>", "").replace("</sup>", "").replace("<sub>", "").replace("</sub>", "")
        else p
    )


def _prose_breaks(text: str) -> str:
    """Normalize prose outside tables: <br> -> newline, <strong>/<b> -> **, <em>/<i> -> *, entities."""
    parts = _TABLE_CHUNK_RE.split(text)
    return "".join(p if i % 2 else _clean_prose_chunk(p) for i, p in enumerate(parts))


def _sx_markdown(items: list[dict[str, Any]]) -> str:
    """Like items_to_markdown, but every line of a heading block keeps its heading marker."""
    out = []
    for it in items:
        lab = (it.get("label") or "").lower()
        txt = it.get("text", "")
        if not txt.strip():
            continue
        if lab in ("title", "section-header", "section_header"):
            mark = "#" if lab == "title" else "##"
            lines = [re.sub(r"^#{1,6}\s+", "", ln.strip()) for ln in txt.split("\n") if ln.strip()]
            from .rules import rules as _rules

            R = _rules()
            if R is not None and len(lines) > 1:
                joined = [lines[0]]
                for ln in lines[1:]:
                    prev = joined[-1]
                    pw = re.sub(r"[*_]", "", prev).rstrip()
                    nw = re.sub(r"^[*_]+", "", ln).lstrip()
                    act = R.decide(
                        "heading_join",
                        prev_connector=bool(re.search(_CONNECTOR_RE, pw)),
                        prev_punct=pw.endswith((".", ":", ";", "!", "?")),
                        next_lower=bool(nw[:1]) and nw[:1].islower(),
                        same_case=(pw.upper() == pw) == (nw.upper() == nw),
                    )
                    if act == "join":
                        joined[-1] = prev + " " + ln
                    else:
                        joined.append(ln)
                lines = joined
            out.append("\n".join(f"{mark} {ln}" for ln in lines))
        else:
            out.append(items_to_markdown([it]))
    return "\n\n".join(out)


_CLAIMS_CACHE: dict[str, dict] = {}


def _load_claims(path: str) -> dict:
    if path not in _CLAIMS_CACHE:
        _CLAIMS_CACHE[path] = json.load(open(path))
    return _CLAIMS_CACHE[path]


def _apply_emphasis(md: str, claim: dict, R: Any) -> tuple[str, dict]:
    """O11-gated application of Luna residual emphasis claims to the page markdown (outside tables)."""
    parts = _TABLE_CHUNK_RE.split(md)
    prose = "\n".join(p for i, p in enumerate(parts) if i % 2 == 0)
    page_chars = max(1, len(prose.replace("*", "").strip()))
    share = sum(len(b["text"]) for b in claim.get("bold", [])) / page_chars
    band = "low" if share < 0.10 else "mid" if share < 0.30 else "high"
    st = {"heading_admit": 0, "bold_admit": 0, "reject": 0, "page_share": round(share, 3)}
    for h in claim.get("headings", []):
        t = h["text"].strip()
        lvl = max(1, min(3, int(h.get("level", 2))))
        new_parts, done = [], False
        for i, p in enumerate(parts):
            if i % 2 == 1 or done:
                new_parts.append(p)
                continue
            lines = p.split("\n")
            for j, ln in enumerate(lines):
                bare = re.sub(r"^\s*(#{1,6}\s+|[-*•]\s+)", "", ln).replace("**", "").strip()
                if bare == t:
                    already = ln.lstrip().startswith("#")
                    v = R.decide(
                        "emphasis_residual",
                        kind="heading",
                        found="exact",
                        length="short" if len(t.split()) <= 12 else "long",
                        already=already,
                        page_share=band,
                    )
                    if v == "admit":
                        lines[j] = "#" * lvl + " " + bare
                        st["heading_admit"] += 1
                    else:
                        st["reject"] += 1
                    done = True
                    break
            new_parts.append("\n".join(lines))
        if not done:
            R.decide(
                "emphasis_residual", kind="heading", found="absent", length="short", already=False, page_share=band
            )
            st["reject"] += 1
        parts = new_parts
    for b in claim.get("bold", []):
        t = b["text"].strip()
        found = already = False
        for i, p in enumerate(parts):
            if i % 2 == 1:
                continue
            k = p.find(t)
            if k >= 0:
                found = True
                already = p[max(0, k - 2) : k] == "**" or bool(re.match(r"^\s*#", p[:k].split("\n")[-1] + " "))
                v = R.decide(
                    "emphasis_residual",
                    kind="bold",
                    found="exact",
                    length="short" if len(t.split()) <= 12 else "long",
                    already=already,
                    page_share=band,
                )
                if v == "admit":
                    parts[i] = p[:k] + "**" + t + "**" + p[k + len(t) :]
                    st["bold_admit"] += 1
                else:
                    st["reject"] += 1
                break
        if not found:
            R.decide("emphasis_residual", kind="bold", found="absent", length="short", already=False, page_share=band)
            st["reject"] += 1
    return "".join(parts), st


def _apply_tables(
    page: Any, pd: dict[str, Any], items: list[dict[str, Any]], R: Any, proposer: str = "vlm"
) -> tuple[list[dict[str, Any]], dict]:
    """O12: reconcile ruling-line PDF tables with the proposer's table items."""
    ptabs = floor.pdf_tables(page)
    if not ptabs:
        return items, {}
    out = [dict(it) for it in items]
    st = {"keep_vlm": 0, "use_pdf": 0, "add_pdf": 0, "skip": 0}
    auth = "full_authority" if pd.get("authority") == "full_authority" else "other"
    for t in ptabs:
        tb = t["bbox"]
        match = None
        for k, it in enumerate(out):
            b = it.get("bbox")
            if (
                (it.get("label") or "").lower() == "table"
                and isinstance(b, list)
                and (floor._iou(b, tb) >= 0.3 or floor._ioa(tb, b) >= 0.5 or floor._ioa(b, tb) >= 0.5)
            ):
                match = k
                break
        vtxt = out[match].get("text", "") if match is not None else ""
        if match is not None:
            vr, vc = floor.html_table_shape(vtxt)
            shape = "same" if (vr, vc) == (t["rows"], t["cols"]) else "different"
            agr = floor.agreement_band(vtxt, t["text"])
            agr = "low" if agr == "empty" else agr
        else:
            shape, agr = "na", "na"
        act = R.decide(
            "table_source",
            proposer=proposer,
            vlm="present" if match is not None else "absent",
            pdf="trivial" if t["rows"] < 2 or t["cols"] < 2 else "ok",
            authority=auth,
            agreement=agr,
            shape=shape,
        )
        st[act] += 1
        if act == "use_pdf":
            out[match]["text"] = t["html"]
        elif act == "add_pdf":
            # drop plain text blocks that only duplicate the table's cells, then insert the table in reading position
            out = [
                it
                for it in out
                if not (
                    isinstance(it.get("bbox"), list)
                    and (it.get("label") or "").lower() not in ("table", "picture", "figure")
                    and floor._ioa(it["bbox"], tb) >= 0.8
                )
            ]
            pos = len(out)
            for k, it in enumerate(out):
                b = it.get("bbox")
                if isinstance(b, list) and b[1] >= tb[1] and min(b[2], tb[2]) - max(b[0], tb[0]) > 0:
                    pos = k
                    break
            out.insert(pos, {"bbox": tb, "label": "Table", "text": t["html"]})
    return out, st


def _apply_table_requery(
    page: Any, pd: dict[str, Any], items: list[dict[str, Any]], claims: list[dict], R: Any
) -> tuple[list[dict[str, Any]], dict]:
    """O13: accept zoomed re-queries of the proposer's own tables (SX_TABLE_REQ_FORCE=1 accepts all: pilot only)."""
    out = [dict(it) for it in items]
    st = {"accept": 0, "keep": 0}
    grids = floor.pdf_tables(page) if pd.get("authority") == "full_authority" else []
    force = os.environ.get("SX_TABLE_REQ_FORCE", "0") == "1"

    def grid_fact(html: str, bb: list[float]) -> str:
        for t in grids:
            if floor._iou(bb, t["bbox"]) >= 0.3 or floor._ioa(t["bbox"], bb) >= 0.5 or floor._ioa(bb, t["bbox"]) >= 0.5:
                return "match" if floor.html_table_shape(html) == (t["rows"], t["cols"]) else "mismatch"
        return "none"

    for c in claims:
        k = c["item"]
        if k >= len(out) or (out[k].get("label") or "").lower() != "table":
            continue
        before, after, bb = out[k].get("text", ""), c["html"], out[k].get("bbox")
        agr = floor.agreement_band(before, after)
        agr = "low" if agr == "empty" else agr
        v = R.decide(
            "table_requery",
            before_consistent=floor.html_grid_consistent(before),
            after_consistent=floor.html_grid_consistent(after),
            agreement=agr,
            grid_before=grid_fact(before, bb) if isinstance(bb, list) else "none",
            grid_after=grid_fact(after, bb) if isinstance(bb, list) else "none",
        )
        if force or v == "accept":
            out[k]["text"] = after
            st["accept"] += 1
        else:
            st["keep"] += 1
    return out, st
