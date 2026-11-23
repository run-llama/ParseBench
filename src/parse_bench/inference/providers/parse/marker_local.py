"""Provider for Marker run locally (``marker-pdf``), as opposed to Datalab's hosted API (``datalab``).

Each file is converted once by ``_marker_runner/marker_local_runner.py`` under the Python of the environment
Marker is installed in: ``MARKER_PYTHON`` (default: ``python`` on ``PATH``). The runner prints
Marker's markdown and its JSON block tree, which has the same shape as Datalab's JSON output.
"""

import json as _json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.evaluation.metrics.attribution.text_utils import extract_text_from_html
from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.parse.liteparse import LiteParseProvider
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

# In a directory of its own: run as a script, its directory leads Marker's sys.path, and this
# package's modules (google.py) would shadow Marker's dependencies.
_RUNNER = Path(__file__).with_name("_marker_runner") / "marker_local_runner.py"

# Marker block type -> canonical label: datalab.DATALAB_LABEL_MAP (not imported: that module needs the
# Datalab SDK) plus the types local Marker also emits at the top level of a page.
MARKER_LABEL_MAP: dict[str, str] = {
    "Text": "Text",
    "SectionHeader": "Section-header",
    "Table": "Table",
    "Figure": "Picture",
    "Picture": "Picture",
    "ListGroup": "List-item",
    "PageHeader": "Page-header",
    "PageFooter": "Page-footer",
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Formula": "Formula",
    "Equation": "Formula",
    "Code": "Code",
    "Form": "Form",
    "Handwriting": "Text",
    "TableOfContents": "Document Index",
    "TextInlineMath": "Text",
    "Bibliography": "Text",
    "ComplexRegion": "Picture",
    "Diagram": "Picture",
    "ChemicalBlock": "Picture",
}
# Marker wraps a table / figure and its caption in a group; the members are the layout blocks.
_GROUPS = {"TableGroup", "FigureGroup", "PictureGroup"}


def _build_layout_pages(json_data: dict[str, Any]) -> list[ParseLayoutPageIR]:
    layout_pages: list[ParseLayoutPageIR] = []
    for page_idx, page in enumerate(json_data.get("children") or []):
        if page.get("block_type") != "Page":
            continue
        px0, py0, px1, py1 = (float(v) for v in page.get("bbox", [0, 0, 1, 1])[:4])
        page_w, page_h = max(px1 - px0, 1e-9), max(py1 - py0, 1e-9)
        blocks = []
        for block in page.get("children") or []:
            if block.get("block_type") in _GROUPS and block.get("children"):
                blocks.extend(block["children"])
            else:
                blocks.append(block)
        items: list[LayoutItemIR] = []
        for block in blocks:
            label = MARKER_LABEL_MAP.get(str(block.get("block_type", "")))
            bbox = block.get("bbox") or []
            if label is None or len(bbox) < 4:
                continue
            x0, y0, x1, y1 = (float(v) for v in bbox[:4])
            if x1 <= x0 or y1 <= y0:
                continue
            seg = LayoutSegmentIR(
                x=(x0 - px0) / page_w,
                y=(y0 - py0) / page_h,
                w=(x1 - x0) / page_w,
                h=(y1 - y0) / page_h,
                confidence=1.0,
                label=label,
            )
            # plain text: the layout metric tokenizes block text as is, so tags and attributes
            # (`<p block-type="Text">`) would count as words
            text = extract_text_from_html(block.get("html", "") or "")
            items.append(LayoutItemIR(type=label, value=text, md=text, bbox=seg, layout_segments=[seg]))
        layout_pages.append(ParseLayoutPageIR(page_number=page_idx + 1, width=page_w, height=page_h, items=items))
    return layout_pages


@register_provider("marker")
class MarkerLocalProvider(Provider):
    """Marker (``marker-pdf``), run locally."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        :param base_config: Recognized keys:
            - `mode`: Marker's ``mode`` ("fast" is its CPU mode, "no_ocr", ...); Marker's default if unset
        """
        super().__init__(provider_name, base_config)
        self._python = os.environ.get("MARKER_PYTHON") or "python"

    def _parse(self, path: Path) -> dict[str, Any]:
        args = [self._python, str(_RUNNER), str(path)]
        if self.base_config.get("mode"):
            args.extend(["--mode", str(self.base_config["mode"])])
        try:
            proc = subprocess.run(args, capture_output=True, text=True, check=False)
        except FileNotFoundError as e:
            raise ProviderConfigError(f"Python for Marker not found at {self._python}; set MARKER_PYTHON") from e
        if proc.returncode != 0:
            raise ProviderPermanentError(f"marker failed (exit {proc.returncode}): {proc.stderr.strip()[-2000:]}")
        try:
            out = _json.loads(proc.stdout)
        except _json.JSONDecodeError as e:
            raise ProviderPermanentError(f"Failed to parse marker runner output: {e}") from e
        return {"markdown": out.get("markdown", ""), "json": out.get("json", {}), "_config": self.base_config}

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"MarkerLocalProvider only supports PARSE, got {request.product_type}")
        path = Path(request.source_file_path)
        if not path.exists():
            raise ProviderPermanentError(f"File not found: {path}")
        started_at = datetime.now()
        raw_output = self._parse(path)
        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        # Marker's markdown writes tables as pipe tables; HTML lets the table metrics see them.
        markdown = LiteParseProvider._convert_md_tables_to_html(raw_result.raw_output.get("markdown", "") or "")
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[PageIR(page_index=0, markdown=markdown)],
            layout_pages=_build_layout_pages(raw_result.raw_output.get("json") or {}),
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
