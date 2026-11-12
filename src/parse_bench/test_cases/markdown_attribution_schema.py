"""Output-centric attribution references (not layout rules)."""

from __future__ import annotations

import math
import re
import warnings
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from parse_bench.test_cases.parse_rule_schemas import ParseRuleBase


def normalize_guideline_ids(value: Any) -> list[str]:
    """Validate applicable guide references without resolving a remote catalog."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+", item.strip()) is None
        for item in value
    ):
        raise ValueError("guideline_ids must be a list of lowercase kebab-case strings")
    return list(dict.fromkeys(item.strip() for item in value))


GuidelineIds = Annotated[list[str], BeforeValidator(normalize_guideline_ids)]


class AttributionRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1)
    bbox: list[float]

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, box: list[float]) -> list[float]:
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("bbox must contain four finite normalized xywh coordinates")
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError("bbox must be positive and inside the normalized page")
        return box


class AttributionReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str
    kind: str = "text"
    regions: list[AttributionRegion] = Field(min_length=1)
    order: int = Field(ge=0)
    exception: Literal["multi_column", "cross_page"] | None = None
    # How many predicted boxes this element may be claimed with before the
    # claim is penalised (quality 0, status ``multi_box_penalized``). 1 means
    # the ordinary one-box contract; set it to 2 for a paragraph that continues
    # in the next column. A claim with fewer boxes is scored normally: one box
    # spanning both columns also covers the gutter, which the IoU already charges.
    expected_bbox_count: int = Field(default=1, ge=1, strict=True)
    # Page furniture a Markdown output may leave out by design (running
    # headers and footers, page numbers; annotation guidelines, Page Furniture).
    # If nothing is assessed against it, it is ``optional`` and out of the
    # weights, like an unscorable reference, instead of ``missed``. If an
    # element does resolve to it, it is scored like any other reference.
    # Furniture (Page-header, Page-footer, Page-number) whose content is absent
    # from the Markdown is ``not_requested`` instead, whatever this flag.
    optional: bool = Field(default=False, strict=True)
    # Other transcriptions of the same source text, taken from a tier-neutral
    # source such as the PDF text layer clipped to the reference's box. An
    # element may resolve through the text or any alternative; two
    # alternatives of one reference never make it ambiguous.
    alt_texts: list[str] = Field(default_factory=list)
    # A Picture that carries no content the Markdown is expected to write:
    # purely decorative background art, an ornament, a scan speck (annotation
    # guidelines, "Decorative pictures in attribution references"). Logos,
    # label icons and photos are content, not decorative. There is no
    # decorative state: when nothing is assessed against it, it gets status
    # ``optional`` and is out of the weights; an image the output writes for
    # it still pairs with it by reading order and is scored.
    decorative: bool = Field(default=False, strict=True)
    # Span targets are optional *reference* evidence, never required parser output.
    spans: list[AttributionReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_regions(self) -> AttributionReference:
        if len(self.regions) > 1 and self.exception is None:
            raise ValueError("multiple reference regions require an explicit discontinuity exception")
        if self.expected_bbox_count > len(self.regions):
            warnings.warn(
                f"reference {self.id!r}: expected_bbox_count {self.expected_bbox_count} exceeds its "
                f"{len(self.regions)} ground-truth region(s); claims with that many boxes are not penalised",
                UserWarning,
                stacklevel=2,
            )
        if len(self.regions) > self.expected_bbox_count:
            warnings.warn(
                f"reference {self.id!r}: {len(self.regions)} ground-truth regions but expected_bbox_count "
                f"{self.expected_bbox_count}; a claim that matches the regions box for box is penalised",
                UserWarning,
                stacklevel=2,
            )
        return self


class MarkdownAttributionRule(ParseRuleBase):
    """Document-scoped source evidence, dispatched outside binary parse rules.

    No legacy ontology projection or implicit LLM judging. Reference recall
    enters only the sqrt-area coverage and the area-weighted secondary, where a
    reference nothing was assessed against counts as 0; the sqrt-area-weighted
    headline, like the count-weighted quality, leaves it out.
    Every rule is scored by scorer version 2 (docs/markdown_attribution_v1.md,
    "Scorer version 2"; metric names keep the historical ``md_attribution_v1_``
    prefix for continuity). ``version`` is optional; a rule that still says
    ``version: 1`` (older copies of the v0.3 sidecars) is read as version 2,
    since the version 1 scorer was removed. A later change of defaults needs a
    new scorer version.

    Every element is held to a box budget. A claim with more distinct boxes
    than its resolved source reference's ``expected_bbox_count`` (1 unless the
    ground truth says otherwise; a stack of merged references allows 1, a
    row of them one per reference) scores
    quality 0 with status ``multi_box_penalized`` and counts as assessed, so an
    engine that splits one element into several boxes is charged in both the
    sqrt-area-weighted headline and both secondary metrics. The penalty is
    always on.
    """

    model_config = ConfigDict(extra="forbid")
    # Pydantic discriminator: narrows the package's ``type: str``, which ty rejects on a mutable field.
    type: Literal["markdown_attribution"] = "markdown_attribution"  # type: ignore[assignment]
    guideline_ids: GuidelineIds = Field(
        default_factory=list,
        description="Applicable annotation guideline IDs, not compliance verdicts",
    )
    # Always 2 after validation. Accepted so older rule files still load:
    # absent, 1 (deprecated, the removed version 1 scorer) and 2 all mean 2.
    version: Literal[2] = 2
    references: list[AttributionReference]
    text_match_threshold: float = Field(default=0.92, ge=0.8, le=1)
    ambiguity_margin: float = Field(default=0.04, ge=0, le=0.2)
    # References one Markdown block may merge: a same-column stack or a same-row
    # line (evaluation/markdown_attribution.py:_merge). 6 covers a checklist or
    # a scorecard written as one paragraph with soft line breaks.
    max_merge_elements: int = Field(default=6, ge=1, le=6)
    # Tables resolve by cell-multiset F1 instead of whole-text similarity; the
    # best reference must reach this F1 (see evaluation/markdown_attribution.py).
    table_match_threshold: float = Field(default=0.6, ge=0.3, le=1)

    @field_validator("version", mode="before")
    @classmethod
    def version_1_reads_as_2(cls, value: object) -> object:
        return 2 if value == 1 and not isinstance(value, bool) else value

    @model_validator(mode="after")
    def document_scope(self) -> MarkdownAttributionRule:
        if self.page is not None:
            raise ValueError("markdown_attribution is document-scoped; regions carry page numbers")
        ids = [ref.id for ref in self.references]
        if len(ids) != len(set(ids)):
            raise ValueError("reference IDs must be unique")
        return self
