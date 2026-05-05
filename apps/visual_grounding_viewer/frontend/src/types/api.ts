export type SourceKind = 'pdf' | 'image'

export interface ArtifactFlags {
  has_v2_items_file: boolean
  has_raw_file: boolean
  has_result_file: boolean
  has_v2_items_payload: boolean
}

export interface VisualizableDocument {
  doc_id: string
  base_name: string
  relative_dir: string
  source_kind: SourceKind
  source_ext: string
  last_modified_ms: number
  artifact_flags: ArtifactFlags
  evaluation_metrics?: Record<string, number>
}

export interface FolderNode {
  name: string
  path: string
  document_count: number
  total_document_count: number
  children: FolderNode[]
}

export interface IndexCounts {
  visualizable: number
  skipped: number
  warnings: number
}

export interface IndexResponse {
  session_id: string
  root_path: string
  resolved_root_path: string
  tree: FolderNode
  documents: VisualizableDocument[]
  document_total: number
  page: number
  page_size: number
  has_more: boolean
  counts: IndexCounts
  warnings: string[]
}

export interface BrowseItem {
  name: string
  path: string
  last_modified_ms: number
  is_dir: boolean
}

export interface BrowseResponse {
  current: string
  parent: string | null
  items: BrowseItem[]
}

export interface GroundingBbox {
  x: number
  y: number
  w: number
  h: number
  label: string | null
  confidence: number | null
  start_index: number | null
  end_index: number | null
}

export interface GroundingItem {
  item_id: string
  item_index: number
  page_number: number
  depth: number
  type: string
  md: string
  value: string | null
  source_path: string
  raw_payload?: Record<string, unknown> | null
  bboxes: GroundingBbox[]
}

export type GroundingGranularity = 'line' | 'word' | 'cell'
export type GroundingLayerAvailability = 'available' | 'empty' | 'unavailable'

export interface GroundingGranularUnit {
  unit_id: string
  granularity: GroundingGranularity
  order_index: number
  text: string
  bbox: GroundingBbox
  bboxes: GroundingBbox[]
  row_index: number | null
  column_index: number | null
  row_span: number | null
  column_span: number | null
  source_path: string | null
  provider: string | null
}

export interface GroundingGranularLayer {
  granularity: GroundingGranularity
  availability: GroundingLayerAvailability
  units: GroundingGranularUnit[]
  reason: string | null
  source: string | null
}

export interface GroundTruthRuleMatch {
  rule_id: string
  rule_type: 'layout' | 'extract_field'
  page_number: number
  gt_bbox: GroundingBbox
  predicted_bbox: GroundingBbox | null
  predicted_bboxes: GroundingBbox[]
  iou: number | null
  bbox_recall: number | null
  field_path?: string | null
  expected_value?: string | number | boolean | null
  evidence_index?: number | null
  predicted_text?: string | null
  predicted_granularity?: 'line' | 'word' | 'extract_field' | null
  matched_unit_ids?: string[]
  text_score?: number | null
  // extract_field evidence metadata. Strays are evidence
  // heuristically assigned to wrap-extras / header-click noise; the frontend
  // styles them distinctly so they don't get confused with verified GT.
  verified?: boolean | null
  tags?: string[]
  source_bbox_index?: number | null
  canonical_class?: string | null
  normalized_attributes?: Record<string, unknown>
  gt_ro_index?: number | null
  gt_text_norm?: string | null
  predicted_class?: string | null
  predicted_class_norm?: string | null
  best_pred_index?: number | null
  best_pred_ioa_gt?: number | null
  localization_pass?: boolean | null
  localization_reason?: string | null
  classification_pass?: boolean | null
  classification_reason?: string | null
  attribution_applicable?: boolean | null
  attribution_pass?: boolean | null
  attribution_reason?: string | null
  attribution_method?: string | null
  attribution_threshold?: number | null
  token_precision?: number | null
  token_recall?: number | null
  token_f1?: number | null
  missing_tokens?: string[]
  extra_tokens?: string[]
  overall_pass?: boolean | null
}

export interface GroundingPage {
  page_number: number
  page_width: number
  page_height: number
  markdown: string | null
  items: GroundingItem[]
  granular_layers: GroundingGranularLayer[]
  gt_rules?: GroundTruthRuleMatch[]
}

export interface DocumentResponse {
  doc_id: string
  base_name: string
  relative_dir: string
  source_kind: SourceKind
  source_ext: string
  source_file_url: string | null
  page_count: number
  pages: GroundingPage[]
  selected_grounding_source: 'v2_items' | 'raw' | 'result'
  selected_markdown_source: 'sidecar_md' | 'raw' | 'result' | null
  document_markdown: string | null
  raw_json: string | null
  result_json: string | null
  artifact_flags: ArtifactFlags
}
