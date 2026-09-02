"""Provider for LiteParse PARSE."""

import json as _json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# Path to the `lit` CLI binary. Resolution order: ``LITEPARSE_BIN`` env var,
# then ``lit`` on ``PATH``. Falls back to a plain ``lit`` so the error message
# below still names the binary when neither is set.
_LIT_BIN = Path(os.environ.get("LITEPARSE_BIN") or shutil.which("lit") or "lit")

# Kwargs accepted in base_config that map directly to CLI flags.
_CLI_FLAG_MAP: dict[str, str] = {
    "ocr_server_url": "--ocr-server-url",
    "ocr_language": "--ocr-language",
    "tessdata_path": "--tessdata-path",
    "dpi": "--dpi",
    "num_workers": "--num-workers",
    "image_mode": "--image-mode",
}

# ``lit --extract-blocks`` block kinds -> LayoutItemIR.type. Kinds with no
# visual region of their own (``rule``) are skipped. Labels are the benchmark's
# canonical names so the layout label mapper is a straight lookup.
_BLOCK_KIND_TO_LABEL: dict[str, str] = {
    "heading": "Section-header",
    "paragraph": "Text",
    "list_item": "List-item",
    "table": "Table",
    "grid_fallback": "Table",
    "figure": "Picture",
    "code": "Code",
}


def _block_text(block: dict[str, Any]) -> str:
    """Best-effort text for a block: ``text`` for prose, cell text for tables."""
    text = block.get("text")
    if isinstance(text, str) and text:
        return text
    if block.get("kind") in {"table", "grid_fallback"}:
        rows = []
        header = block.get("header")
        if isinstance(header, list):
            rows.append(header)
        rows.extend(r for r in (block.get("rows") or []) if isinstance(r, list))
        lines = []
        for row in rows:
            cells = [str(c.get("text", "") if isinstance(c, dict) else c) for c in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)
    if block.get("kind") == "code":
        lines = block.get("lines")
        if isinstance(lines, list):
            return "\n".join(str(line) for line in lines)
    return ""


def _build_layout_pages(pages_raw: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Convert ``lit --extract-blocks`` output into normalized layout pages.

    Block bboxes are ``{x, y, width, height}`` in PDF points on the page's own
    coordinate system; they are normalized by the page size to the [0, 1]
    frame ``LayoutSegmentIR`` expects.
    """
    layout_pages: list[ParseLayoutPageIR] = []
    for page_data in pages_raw:
        blocks = page_data.get("blocks") or []
        width = page_data.get("width")
        height = page_data.get("height")
        if not blocks or not isinstance(width, int | float) or not isinstance(height, int | float):
            continue
        if width <= 0 or height <= 0:
            continue
        items: list[LayoutItemIR] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            label = _BLOCK_KIND_TO_LABEL.get(str(block.get("kind", "")))
            bbox = block.get("bbox")
            if label is None or not isinstance(bbox, dict):
                continue
            try:
                x = float(bbox["x"]) / width
                y = float(bbox["y"]) / height
                w = float(bbox["width"]) / width
                h = float(bbox["height"]) / height
            except (KeyError, TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            segment = LayoutSegmentIR(x=x, y=y, w=w, h=h, confidence=1.0, label=label)
            text = _block_text(block)
            items.append(
                LayoutItemIR(
                    type=label,
                    value=text,
                    md=text,
                    bbox=segment,
                    layout_segments=[segment],
                )
            )
        if not items:
            continue
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=int(page_data.get("page_index", 0)) + 1,
                width=float(width),
                height=float(height),
                md=page_data.get("text", "") or "",
                items=items,
            )
        )
    return layout_pages


@register_provider("liteparse")
class LiteParseProvider(Provider):
    """
    Provider for LiteParse PARSE.

    Uses the local in-process LiteParse Python bindings (no API key required).
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration. Recognized keys:
            - `output_format`: "markdown" | "text" | "json" (default: "markdown")
            - `ocr_enabled`: bool
            - `ocr_server_url`: str
            - `ocr_language`: str
            - `dpi`: float
        """
        super().__init__(provider_name, base_config)
        self._output_format = self.base_config.get("output_format", "markdown")
        self._ocr_enabled = self.base_config.get("ocr_enabled", True)
        self._preserve_small_text = self.base_config.get("preserve_very_small_text", False)
        self._extract_blocks = bool(self.base_config.get("extract_blocks", True))
        self._flag_kwargs = {k: v for k, v in self.base_config.items() if k in _CLI_FLAG_MAP and v is not None}

    def _build_cli_args(self, pdf_path: str, fmt: str) -> list[str]:
        args: list[str] = [str(_LIT_BIN), "parse", pdf_path, "--format", fmt, "--quiet"]
        if fmt == "json" and self._extract_blocks:
            # Layout blocks (kind + bbox in points) feed ParseOutput.layout_pages
            # so the Visual Grounding metrics can score LiteParse.
            args.append("--extract-blocks")
        # Ground truth uses plain text (no markdown link syntax), so disable
        # hyperlink extraction for benchmark parity.
        args.append("--no-links")
        if not self._ocr_enabled:
            args.append("--no-ocr")
        if self._preserve_small_text:
            args.append("--preserve-small-text")
        for key, flag in _CLI_FLAG_MAP.items():
            if key in self._flag_kwargs:
                args.extend([flag, str(self._flag_kwargs[key])])
        return args

    def _invoke_cli(self, pdf_path: str, fmt: str) -> str:
        try:
            proc = subprocess.run(
                self._build_cli_args(pdf_path, fmt),
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise ProviderConfigError(
                f"liteparse CLI binary not found at {_LIT_BIN}. "
                f"Install it with `pip install liteparse` (or `parse-bench[liteparse]`), "
                f"or point LITEPARSE_BIN at a `lit` binary."
            ) from e
        if proc.returncode != 0:
            raise ProviderPermanentError(
                f"lit parse --format {fmt} failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout

    def _parse(self, pdf_path: str) -> dict[str, Any]:
        json_stdout = self._invoke_cli(pdf_path, "json")
        try:
            parsed = _json.loads(json_stdout)
        except _json.JSONDecodeError as e:
            raise ProviderPermanentError(f"Failed to parse lit JSON output: {e}") from e

        pages_raw = parsed.get("pages", []) or []
        pages = [
            {
                "page_index": (p.get("page", 1) - 1) if isinstance(p.get("page"), int) and p["page"] > 0 else 0,
                "text": p.get("text", "") or "",
                "width": p.get("width"),
                "height": p.get("height"),
                "blocks": p.get("blocks") or [],
            }
            for p in pages_raw
        ]

        if self._output_format == "json":
            full_text = ""
        elif self._output_format == "markdown":
            full_text = self._invoke_cli(pdf_path, "markdown")
        else:
            full_text = "\n".join(p["text"] for p in pages)

        return {
            "pages": pages,
            "num_pages": len(pages),
            "text": full_text,
            "output_format": self._output_format,
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"LiteParseProvider only supports PARSE product type, got {request.product_type}"
            )

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        started_at = datetime.now()
        try:
            raw_output = self._parse(str(pdf_path))
            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )
        except (ProviderPermanentError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    @staticmethod
    def _convert_md_tables_to_html(content: str) -> str:
        """Convert markdown pipe tables to HTML so GriTS / TRM can score them.

        ParseBench's table metrics only see ``<table>`` blocks; pipe tables in
        markdown are invisible to them. We convert here at the boundary rather
        than changing liteparse's emitter.
        """
        import markdown2

        lines = content.split("\n")
        result_parts: list[str] = []
        table_lines: list[str] = []
        in_table = False

        def _flush() -> None:
            nonlocal table_lines
            if len(table_lines) >= 2:
                table_md = "\n".join(table_lines)
                html = markdown2.markdown(table_md, extras=["tables"]).strip()
                if "<table>" in html.lower():
                    result_parts.append(html)
                else:
                    result_parts.extend(table_lines)
            else:
                result_parts.extend(table_lines)
            table_lines = []

        for line in lines:
            is_table_line = "|" in line and line.strip().startswith("|")
            if is_table_line:
                in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    _flush()
                    in_table = False
                result_parts.append(line)

        if in_table:
            _flush()

        return "\n".join(result_parts)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"LiteParseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        convert_tables = self._output_format == "markdown"

        pages: list[PageIR] = []
        page_texts: list[str] = []
        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)
            text = page_data.get("text", "") or ""
            if convert_tables:
                text = self._convert_md_tables_to_html(text)
            pages.append(PageIR(page_index=page_index, markdown=text))
            page_texts.append(text)

        full_text = raw_result.raw_output.get("text") or "\n\n".join(page_texts)
        if convert_tables and full_text:
            full_text = self._convert_md_tables_to_html(full_text)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=_build_layout_pages(raw_result.raw_output.get("pages", [])),
            markdown=full_text,
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
