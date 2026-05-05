from __future__ import annotations

from typing import Any, Literal


from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class IndexRequest(BaseModel):
    root_path: str
    test_cases_path: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=5000, ge=1, le=10000)


class ArtifactFlags(BaseModel):
    has_v2_items_file: bool
    has_raw_file: bool
    has_result_file: bool
    has_v2_items_payload: bool


class VisualizableDocument(BaseModel):
    doc_id: str
    base_name: str
    relative_dir: str
    source_kind: Literal["pdf", "image"]
    source_ext: str
    last_modified_ms: int
    artifact_flags: ArtifactFlags
    evaluation_metrics: dict[str, float] = Field(default_factory=dict)


class FolderNode(BaseModel):
    name: str
    path: str
    document_count: int
    total_document_count: int
    children: list["FolderNode"] = Field(default_factory=list)


FolderNode.model_rebuild()


class IndexCounts(BaseModel):
    visualizable: int
    skipped: int
    warnings: int


class IndexResponse(BaseModel):
    session_id: str
    root_path: str
    resolved_root_path: str
    tree: FolderNode
    documents: list[VisualizableDocument]
    document_total: int
    page: int
    page_size: int
    has_more: bool
    counts: IndexCounts
    warnings: list[str]


class BrowseItem(BaseModel):
    name: str
    path: str
    last_modified_ms: int
    is_dir: bool = True


class BrowseResponse(BaseModel):
    current: str
    parent: str | None = None
    items: list[BrowseItem] = Field(default_factory=list)


class GroundingBbox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    label: str | None = None
    confidence: float | None = None
    start_index: int | None = None
    end_index: int | None = None


class GroundingGranularUnit(BaseModel):
    unit_id: str
    granularity: Literal["line", "word", "cell"]
    order_index: int
    text: str = ""
    bbox: GroundingBbox
    bboxes: list[GroundingBbox] = Field(default_factory=list)
    row_index: int | None = None
    column_index: int | None = None
    row_span: int | None = None
    column_span: int | None = None
    source_path: str | None = None
    provider: str | None = None


class GroundingGranularLayer(BaseModel):
    granularity: Literal["line", "word", "cell"]
    availability: Literal["available", "empty", "unavailable"]
    units: list[GroundingGranularUnit] = Field(default_factory=list)
    reason: str | None = None
    source: str | None = None


class GroundTruthRuleMatch(BaseModel):
    rule_id: str
    rule_type: Literal["layout", "extract_field"]
    page_number: int
    gt_bbox: GroundingBbox
    predicted_bbox: GroundingBbox | None = None
    predicted_bboxes: list[GroundingBbox] = Field(default_factory=list)
    iou: float | None = None
    bbox_recall: float | None = None

    field_path: str | None = None
    expected_value: str | int | float | bool | None = None
    evidence_index: int | None = None
    predicted_text: str | None = None
    predicted_granularity: Literal["line", "word", "extract_field"] | None = None
    matched_unit_ids: list[str] = Field(default_factory=list)
    text_score: float | None = None

    # extract_field rules carry additional evidence metadata:
    # a verification flag and free-form tags (notably "stray_evidence" for
    # evidence heuristically assigned to table wrap-extras / header clicks).
    # source_bbox_index preserves the position of this bbox in the original
    # multi-bbox rule so a multi-evidence field can round-trip.
    verified: bool | None = None
    tags: list[str] = Field(default_factory=list)
    source_bbox_index: int | None = None

    canonical_class: str | None = None
    normalized_attributes: dict[str, Any] = Field(default_factory=dict)
    gt_ro_index: int | None = None
    gt_text_norm: str | None = None
    predicted_class: str | None = None
    predicted_class_norm: str | None = None
    best_pred_index: int | None = None
    best_pred_ioa_gt: float | None = None
    localization_pass: bool | None = None
    localization_reason: str | None = None
    classification_pass: bool | None = None
    classification_reason: str | None = None
    attribution_applicable: bool | None = None
    attribution_pass: bool | None = None
    attribution_reason: str | None = None
    attribution_method: str | None = None
    attribution_threshold: float | None = None
    token_precision: float | None = None
    token_recall: float | None = None
    token_f1: float | None = None
    missing_tokens: list[str] = Field(default_factory=list)
    extra_tokens: list[str] = Field(default_factory=list)
    overall_pass: bool | None = None


class GroundingItem(BaseModel):
    item_id: str
    item_index: int
    page_number: int
    depth: int
    type: str
    md: str
    value: str | None = None
    source_path: str
    raw_payload: dict[str, Any] | None = None
    bboxes: list[GroundingBbox] = Field(default_factory=list)


class GroundingPage(BaseModel):
    page_number: int
    page_width: float
    page_height: float
    markdown: str | None = None
    items: list[GroundingItem] = Field(default_factory=list)
    granular_layers: list[GroundingGranularLayer] = Field(default_factory=list)
    gt_rules: list[GroundTruthRuleMatch] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    doc_id: str
    base_name: str
    relative_dir: str
    source_kind: Literal["pdf", "image"]
    source_ext: str
    source_file_url: str | None = None
    page_count: int
    pages: list[GroundingPage]
    selected_grounding_source: Literal["v2_items", "raw", "result"]
    selected_markdown_source: Literal["sidecar_md", "raw", "result"] | None = None
    document_markdown: str | None = None
    raw_json: str | None = None
    result_json: str | None = None
    artifact_flags: ArtifactFlags
