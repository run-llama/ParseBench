import { type CSSProperties, type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import { formatGranularUnitLabel, formatGranularUnitMetadata, type OverlayLayerVisibility } from '../lib/grounding'
import { computeGtOverlayMetrics } from '../lib/gtOverlay'
import type {
  DocumentResponse,
  GroundingGranularLayer,
  GroundingGranularUnit,
  GroundingGranularity,
  GroundingItem,
  GroundTruthRuleMatch,
} from '../types/api'
import { ItemMarkdownPane } from './ItemMarkdownPane'
import { TextDiff } from './TextDiff'

type RightTab = 'markdown' | 'elements' | 'granular' | 'gt' | 'raw' | 'result'
type ElementSortMode = 'default' | 'bbox_desc' | 'bbox_asc'
type GranularFilterMode = 'all' | GroundingGranularity
type GtRuleType = GroundTruthRuleMatch['rule_type']
type GtSortDirection = 'highest' | 'lowest'
type GtFieldSortMetric =
  | 'overall'
  | 'localization'
  | 'classification'
  | 'attribution'
  | 'iou'
  | 'text_score'
  | 'f1'
  | 'recall'
  | 'precision'
type GtLayoutSortMetric = 'overall' | 'localization' | 'classification' | 'attribution' | 'iou'
type GtSortMetric = GtFieldSortMetric | GtLayoutSortMetric

interface RightPanelProps {
  document: DocumentResponse
  pageItems: GroundingItem[]
  pageGranularLayers: GroundingGranularLayer[]
  pageGtRules: GroundTruthRuleMatch[]
  visibleLayers: OverlayLayerVisibility
  activeItemId: string | null
  hoveredItemId: string | null
  activeGranularUnit: GroundingGranularUnit | null
  hoveredGranularUnit: GroundingGranularUnit | null
  activeGranularPreview: GroundingGranularUnit | null
  hoveredGranularPreview: GroundingGranularUnit | null
  activeGtRule: GroundTruthRuleMatch | null
  hoveredGtRule: GroundTruthRuleMatch | null
  hoverSource: 'viewer' | 'sidebar' | null
  onHoverItem: (itemId: string | null) => void
  onSelectItem: (itemId: string) => void
  onHoverGranularUnit: (unitId: string | null, granularity: GroundingGranularity | null) => void
  onSelectGranularUnit: (unitId: string, granularity: GroundingGranularity) => void
  onHoverGranularPreview: (unit: GroundingGranularUnit | null) => void
  onSelectGranularPreview: (unit: GroundingGranularUnit | null) => void
  onHoverGtRule: (ruleId: string | null) => void
  onSelectGtRule: (ruleId: string) => void
  onHoverEvidence: (itemId: string | null, ruleIds: string[]) => void
  onSelectEvidence: (itemId: string | null, ruleIds: string[]) => void
  onCollapse: () => void
}

type JsonTreeValue = null | boolean | number | string | JsonTreeValue[] | { [key: string]: JsonTreeValue }
type ExtractViewMode = 'json' | 'rules'
type ExtractEvidenceFilterMode =
  | 'all'
  | 'overall_fail'
  | 'localization_fail'
  | 'attribution_fail'
  | 'no_prediction'
  | 'needs_review'
  | 'verified'
type ExtractEvidenceSortMode = 'document' | 'worst'

interface ExtractEvidenceAnchor {
  rules: GroundTruthRuleMatch[]
  items: GroundingItem[]
}

interface ExtractEvidenceNode {
  path: string
  label: string | null
  value: JsonTreeValue | undefined
  children: ExtractEvidenceNode[]
  anchors: ExtractEvidenceAnchor
  anchoredLeafCount: number
}

interface ExtractPathToken {
  label: string
  arrayIndex: boolean
}

interface MutableExtractEvidenceNode {
  path: string
  label: string | null
  value: JsonTreeValue | undefined
  children: Map<string, MutableExtractEvidenceNode>
  anchors: ExtractEvidenceAnchor
  order: number
}

interface ExtractEvidenceAggregate {
  ruleCount: number
  verifiedCount: number
  needsReviewCount: number
  overallFailCount: number
  localizationFailCount: number
  attributionFailCount: number
  noPredictionCount: number
  worstOverall: number | null
  worstLocalization: number | null
  worstAttribution: number | null
}

function summarizeBbox(unit: GroundingGranularUnit): string {
  const summary = `${Math.round(unit.bbox.x)}, ${Math.round(unit.bbox.y)} · ${Math.round(unit.bbox.w)}×${Math.round(unit.bbox.h)}`
  const regionCount = unit.bboxes.length
  return regionCount > 1 ? `${summary} · ${regionCount} regions` : summary
}

function previewText(value: string): string {
  const normalized = value.replace(/\s+/g, ' ').trim()
  if (!normalized) {
    return 'No text'
  }
  return normalized.length > 120 ? `${normalized.slice(0, 117)}...` : normalized
}

function layerDescription(layer: GroundingGranularLayer): string {
  if (layer.availability === 'unavailable') {
    return layer.reason ?? `${layer.granularity} overlays are unavailable on this page.`
  }
  if (layer.availability === 'empty') {
    return `No ${layer.granularity} overlays are present on this page.`
  }
  return `${layer.units.length} ${layer.granularity}${layer.units.length === 1 ? '' : 's'}`
}

function formatRuleValue(value: GroundTruthRuleMatch['expected_value']): string {
  if (value === null || value === undefined) {
    return 'null'
  }
  const normalized = String(value).replace(/\s+/g, ' ').trim()
  if (!normalized) {
    return '""'
  }
  return normalized.length > 140 ? `${normalized.slice(0, 137)}...` : normalized
}

function formatRulePercent(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return 'n/a'
  }
  return `${(value * 100).toFixed(1)}%`
}

function metricLabel(metric: GtSortMetric): string {
  if (metric === 'f1') {
    return 'F1'
  }
  if (metric === 'iou') {
    return 'IoU'
  }
  if (metric === 'overall') {
    return 'Overall'
  }
  if (metric === 'localization') {
    return 'Loc'
  }
  if (metric === 'classification') {
    return 'Class'
  }
  if (metric === 'attribution') {
    return 'Attr'
  }
  if (metric === 'text_score') {
    return 'Text'
  }
  if (metric === 'recall') {
    return 'R'
  }
  return 'P'
}

function gtScoreTone(value: number | null): 'bad' | 'warn' | 'good' | 'great' | 'na' {
  if (value === null || Number.isNaN(value)) {
    return 'na'
  }
  if (value < 0.5) {
    return 'bad'
  }
  if (value < 0.8) {
    return 'warn'
  }
  if (value < 0.9) {
    return 'good'
  }
  return 'great'
}

function gtRuleTypeLabel(ruleType: GtRuleType): string {
  if (ruleType === 'layout') {
    return 'layout elements'
  }
  if (ruleType === 'extract_field') {
    return 'extract field evidence'
  }
  return 'field evidence'
}

function ruleIsStray(rule: GroundTruthRuleMatch): boolean {
  return (rule.tags ?? []).some((tag) => tag === 'stray_evidence')
}

function rulePreviewLabel(rule: GroundTruthRuleMatch): string {
  if (rule.rule_type === 'layout') {
    return rule.gt_ro_index !== null ? `${rule.canonical_class ?? 'layout'} · ro:${rule.gt_ro_index}` : (rule.canonical_class ?? 'layout')
  }
  const fieldPath = rule.field_path ?? 'field'
  return rule.evidence_index !== null ? `${fieldPath} · #${rule.evidence_index}` : fieldPath
}

function rulePreviewSubmeta(rule: GroundTruthRuleMatch): string {
  if (rule.rule_type === 'layout') {
    return rule.predicted_class ?? 'no match'
  }
  if (rule.predicted_granularity === 'extract_field') {
    return 'extract citation'
  }
  return rule.predicted_granularity ?? 'no match'
}

function gtRuleMetricValue(rule: GroundTruthRuleMatch, metric: GtSortMetric): number | null {
  if (rule.rule_type === 'layout') {
    if (metric === 'iou') {
      return rule.iou
    }
    if (metric === 'overall') {
      return rule.overall_pass === null ? null : rule.overall_pass ? 1 : 0
    }
    if (metric === 'localization') {
      return rule.localization_pass === null ? null : rule.localization_pass ? 1 : 0
    }
    if (metric === 'classification') {
      return rule.classification_pass === null ? null : rule.classification_pass ? 1 : 0
    }
    if (metric === 'attribution') {
      if (rule.attribution_applicable === false) {
        return null
      }
      return rule.attribution_pass === null ? null : rule.attribution_pass ? 1 : 0
    }
    return null
  }

  // extract_field: prefer the Wave-1 / Phase-1 attribution verdicts (which
  // come from the metric's rule_results). Fall back to the geometry-only
  // metrics from computeGtOverlayMetrics when the dimension is missing.
  if (metric === 'overall') {
    return rule.overall_pass == null ? null : rule.overall_pass ? 1 : 0
  }
  if (metric === 'localization') {
    return rule.localization_pass == null ? null : rule.localization_pass ? 1 : 0
  }
  if (metric === 'classification') {
    return rule.classification_pass == null ? null : rule.classification_pass ? 1 : 0
  }
  if (metric === 'attribution') {
    return rule.attribution_pass == null ? null : rule.attribution_pass ? 1 : 0
  }
  if (metric === 'text_score') {
    return rule.text_score == null ? null : rule.text_score
  }
  if (metric === 'iou') {
    // Prefer the metric's iou (field-evidence-spec field IoU) when present;
    // fall back to the viz's geometric IoU.
    return rule.iou ?? computeGtOverlayMetrics(rule).iou
  }
  const metrics = computeGtOverlayMetrics(rule)
  if (metric === 'f1') {
    return metrics.f1
  }
  if (metric === 'recall') {
    return metrics.recall
  }
  if (metric === 'precision') {
    return metrics.precision
  }
  return null
}

function gtStatusCopy(value: boolean | null, unavailableCopy = 'n/a'): string {
  if (value === null) {
    return unavailableCopy
  }
  return value ? 'pass' : 'fail'
}

function summarizeJsonValue(value: JsonTreeValue): string {
  if (Array.isArray(value)) {
    return `[${value.length}]`
  }
  if (value === null) {
    return 'null'
  }
  if (typeof value === 'object') {
    return `{${Object.keys(value).length}}`
  }
  if (typeof value === 'string') {
    return `"${value.length > 36 ? `${value.slice(0, 33)}...` : value}"`
  }
  return String(value)
}

function isJsonTreeValue(value: unknown): value is JsonTreeValue {
  if (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return true
  }
  if (Array.isArray(value)) {
    return value.every(isJsonTreeValue)
  }
  if (typeof value === 'object') {
    return Object.values(value as Record<string, unknown>).every(isJsonTreeValue)
  }
  return false
}

function parseExtractedData(resultJson: string | null): JsonTreeValue | null {
  if (!resultJson) {
    return null
  }
  try {
    const payload = JSON.parse(resultJson) as Record<string, unknown>
    const output =
      payload.output && typeof payload.output === 'object' && !Array.isArray(payload.output)
        ? (payload.output as Record<string, unknown>)
        : null
    const extractedData = output?.extracted_data ?? payload.extracted_data
    return isJsonTreeValue(extractedData) ? extractedData : null
  } catch {
    return null
  }
}

function fieldPathFromItem(item: GroundingItem): string | null {
  const rawPayload = item.raw_payload
  const fieldPath = rawPayload?.field_path
  return typeof fieldPath === 'string' && fieldPath.length > 0 ? fieldPath : null
}

function buildExtractEvidenceAnchors(
  rules: GroundTruthRuleMatch[],
  items: GroundingItem[],
): Map<string, ExtractEvidenceAnchor> {
  const anchors = new Map<string, ExtractEvidenceAnchor>()

  const ensureAnchor = (path: string): ExtractEvidenceAnchor => {
    const existing = anchors.get(path)
    if (existing) {
      return existing
    }
    const next = { rules: [], items: [] }
    anchors.set(path, next)
    return next
  }

  rules.forEach((rule) => {
    if (rule.rule_type !== 'extract_field' || !rule.field_path) {
      return
    }
    ensureAnchor(rule.field_path).rules.push(rule)
  })

  items.forEach((item) => {
    const fieldPath = fieldPathFromItem(item)
    if (!fieldPath) {
      return
    }
    ensureAnchor(fieldPath).items.push(item)
  })

  return anchors
}

function childExtractPath(parentPath: string, childLabel: string, parentIsArray: boolean): string {
  if (parentIsArray) {
    return parentPath ? `${parentPath}[${childLabel}]` : `[${childLabel}]`
  }
  return parentPath ? `${parentPath}.${childLabel}` : childLabel
}

function parseExtractFieldPath(path: string): ExtractPathToken[] {
  const tokens: ExtractPathToken[] = []
  let cursor = 0
  let buffer = ''

  const flushBuffer = () => {
    if (buffer.length > 0) {
      tokens.push({ label: buffer, arrayIndex: false })
      buffer = ''
    }
  }

  while (cursor < path.length) {
    const char = path[cursor]
    if (char === '.') {
      flushBuffer()
      cursor += 1
      continue
    }
    if (char === '[') {
      flushBuffer()
      const closeIndex = path.indexOf(']', cursor)
      if (closeIndex === -1) {
        buffer += char
        cursor += 1
        continue
      }
      tokens.push({ label: path.slice(cursor + 1, closeIndex), arrayIndex: true })
      cursor = closeIndex + 1
      continue
    }
    buffer += char
    cursor += 1
  }

  flushBuffer()
  return tokens
}

function getExtractValueAtTokens(value: JsonTreeValue | undefined, tokens: ExtractPathToken[]): JsonTreeValue | undefined {
  let current: JsonTreeValue | undefined = value
  for (const token of tokens) {
    if (current === undefined || current === null) {
      return undefined
    }
    if (token.arrayIndex) {
      if (!Array.isArray(current)) {
        return undefined
      }
      const index = Number.parseInt(token.label, 10)
      if (!Number.isInteger(index) || index < 0 || index >= current.length) {
        return undefined
      }
      current = current[index]
    } else {
      if (Array.isArray(current) || typeof current !== 'object') {
        return undefined
      }
      current = current[token.label]
    }
  }
  return current
}

function makeMutableExtractNode(
  path: string,
  label: string | null,
  value: JsonTreeValue | undefined,
  anchors: ExtractEvidenceAnchor,
  order: number,
): MutableExtractEvidenceNode {
  return {
    path,
    label,
    value,
    children: new Map(),
    anchors,
    order,
  }
}

function extractArrayIndexSortValue(label: string | null): number | null {
  if (!label) {
    return null
  }
  const match = label.match(/^\[(\d+)\]$/)
  if (!match) {
    return null
  }
  return Number.parseInt(match[1], 10)
}

function buildExtractEvidenceNodeFromAnchors(
  extractedData: JsonTreeValue | null,
  anchorsByPath: Map<string, ExtractEvidenceAnchor>,
): ExtractEvidenceNode | null {
  const root = makeMutableExtractNode('', null, extractedData ?? undefined, { rules: [], items: [] }, 0)
  let order = 1

  anchorsByPath.forEach((anchors, fieldPath) => {
    const tokens = parseExtractFieldPath(fieldPath)
    if (tokens.length === 0) {
      root.anchors = anchors
      return
    }

    let current = root
    tokens.forEach((token, tokenIndex) => {
      const nextPath = token.arrayIndex ? `${current.path}[${token.label}]` : childExtractPath(current.path, token.label, false)
      const childKey = token.arrayIndex ? `[${token.label}]` : token.label
      const existing = current.children.get(childKey)
      const childValue = getExtractValueAtTokens(extractedData ?? undefined, tokens.slice(0, tokenIndex + 1))
      if (existing) {
        if (existing.value === undefined && childValue !== undefined) {
          existing.value = childValue
        }
        current = existing
        return
      }
      const child = makeMutableExtractNode(
        nextPath,
        childKey,
        childValue,
        tokenIndex === tokens.length - 1 ? anchors : { rules: [], items: [] },
        order,
      )
      order += 1
      current.children.set(childKey, child)
      current = child
    })

    if (current.path === fieldPath) {
      current.anchors = anchors
    }
  })

  const finalize = (node: MutableExtractEvidenceNode): ExtractEvidenceNode => {
    const children = Array.from(node.children.values())
      .sort((left, right) => {
        const leftIndex = extractArrayIndexSortValue(left.label)
        const rightIndex = extractArrayIndexSortValue(right.label)
        if (leftIndex !== null && rightIndex !== null && leftIndex !== rightIndex) {
          return leftIndex - rightIndex
        }
        return left.order - right.order
      })
      .map(finalize)
    const hasAnchor = node.anchors.rules.length > 0 || node.anchors.items.length > 0
    const anchoredLeafCount =
      children.length > 0 ? children.reduce((sum, child) => sum + child.anchoredLeafCount, 0) : hasAnchor ? 1 : 0
    return {
      path: node.path,
      label: node.label,
      value: node.value,
      children,
      anchors: node.anchors,
      anchoredLeafCount,
    }
  }

  const finalized = finalize(root)
  return finalized.anchoredLeafCount > 0 ? finalized : null
}

function formatExtractJsonValue(value: JsonTreeValue | undefined): string {
  if (value === undefined) {
    return 'missing'
  }
  if (value === null) {
    return 'null'
  }
  if (Array.isArray(value) || typeof value === 'object') {
    return summarizeJsonValue(value)
  }
  return previewText(String(value))
}

function firstAnchorItem(anchors: ExtractEvidenceAnchor): GroundingItem | null {
  return anchors.items[0] ?? null
}

function firstAnchorRule(anchors: ExtractEvidenceAnchor): GroundTruthRuleMatch | null {
  return anchors.rules[0] ?? null
}

function extractNodeIsBranch(node: ExtractEvidenceNode): boolean {
  return node.children.length > 0 || Array.isArray(node.value) || (node.value !== null && typeof node.value === 'object')
}

function extractNodeIsArrayRecord(node: ExtractEvidenceNode): boolean {
  return extractArrayIndexSortValue(node.label) !== null && node.value !== null && typeof node.value === 'object' && !Array.isArray(node.value)
}

function emptyExtractEvidenceAggregate(): ExtractEvidenceAggregate {
  return {
    ruleCount: 0,
    verifiedCount: 0,
    needsReviewCount: 0,
    overallFailCount: 0,
    localizationFailCount: 0,
    attributionFailCount: 0,
    noPredictionCount: 0,
    worstOverall: null,
    worstLocalization: null,
    worstAttribution: null,
  }
}

function minNullableMetric(left: number | null, right: number | null): number | null {
  if (left === null) {
    return right
  }
  if (right === null) {
    return left
  }
  return Math.min(left, right)
}

function mergeExtractEvidenceAggregate(
  left: ExtractEvidenceAggregate,
  right: ExtractEvidenceAggregate,
): ExtractEvidenceAggregate {
  return {
    ruleCount: left.ruleCount + right.ruleCount,
    verifiedCount: left.verifiedCount + right.verifiedCount,
    needsReviewCount: left.needsReviewCount + right.needsReviewCount,
    overallFailCount: left.overallFailCount + right.overallFailCount,
    localizationFailCount: left.localizationFailCount + right.localizationFailCount,
    attributionFailCount: left.attributionFailCount + right.attributionFailCount,
    noPredictionCount: left.noPredictionCount + right.noPredictionCount,
    worstOverall: minNullableMetric(left.worstOverall, right.worstOverall),
    worstLocalization: minNullableMetric(left.worstLocalization, right.worstLocalization),
    worstAttribution: minNullableMetric(left.worstAttribution, right.worstAttribution),
  }
}

function ruleMetricPasses(rule: GroundTruthRuleMatch, metric: GtSortMetric): boolean {
  if (metric === 'overall' && rule.overall_pass !== null && rule.overall_pass !== undefined) {
    return rule.overall_pass
  }
  if (metric === 'localization' && rule.localization_pass !== null && rule.localization_pass !== undefined) {
    return rule.localization_pass
  }
  if (metric === 'attribution' && rule.attribution_pass !== null && rule.attribution_pass !== undefined) {
    return rule.attribution_pass
  }
  const value = gtRuleMetricValue(rule, metric)
  return value !== null && !Number.isNaN(value) && value >= 1
}

function ruleMetricFails(rule: GroundTruthRuleMatch, metric: GtSortMetric): boolean {
  if (metric === 'overall' && rule.overall_pass !== null && rule.overall_pass !== undefined) {
    return !rule.overall_pass
  }
  if (metric === 'localization' && rule.localization_pass !== null && rule.localization_pass !== undefined) {
    return !rule.localization_pass
  }
  if (metric === 'attribution' && rule.attribution_pass !== null && rule.attribution_pass !== undefined) {
    return !rule.attribution_pass
  }
  const value = gtRuleMetricValue(rule, metric)
  return value !== null && !Number.isNaN(value) && value < 1
}

function ruleHasNoPrediction(rule: GroundTruthRuleMatch): boolean {
  return !rule.predicted_bbox && rule.predicted_bboxes.length === 0
}

function ruleNeedsReview(rule: GroundTruthRuleMatch): boolean {
  if (rule.verified === false) {
    return true
  }
  if (rule.verified === true && ruleMetricPasses(rule, 'overall')) {
    return false
  }
  return !ruleMetricPasses(rule, 'overall')
}

function extractRuleAggregate(rule: GroundTruthRuleMatch): ExtractEvidenceAggregate {
  const overall = gtRuleMetricValue(rule, 'overall')
  const localization = gtRuleMetricValue(rule, 'localization')
  const attribution = gtRuleMetricValue(rule, 'attribution')
  const needsReview = ruleNeedsReview(rule)

  return {
    ruleCount: 1,
    verifiedCount: needsReview ? 0 : 1,
    needsReviewCount: needsReview ? 1 : 0,
    overallFailCount: ruleMetricFails(rule, 'overall') ? 1 : 0,
    localizationFailCount: ruleMetricFails(rule, 'localization') ? 1 : 0,
    attributionFailCount: ruleMetricFails(rule, 'attribution') ? 1 : 0,
    noPredictionCount: ruleHasNoPrediction(rule) ? 1 : 0,
    worstOverall: overall,
    worstLocalization: localization,
    worstAttribution: attribution,
  }
}

function extractEvidenceAggregate(node: ExtractEvidenceNode): ExtractEvidenceAggregate {
  const ownAggregate = node.anchors.rules.reduce(
    (aggregate, rule) => mergeExtractEvidenceAggregate(aggregate, extractRuleAggregate(rule)),
    emptyExtractEvidenceAggregate(),
  )
  return node.children.reduce(
    (aggregate, child) => mergeExtractEvidenceAggregate(aggregate, extractEvidenceAggregate(child)),
    ownAggregate,
  )
}

function extractEvidenceFilterMatches(
  aggregate: ExtractEvidenceAggregate,
  filterMode: ExtractEvidenceFilterMode,
): boolean {
  if (filterMode === 'all') {
    return true
  }
  if (filterMode === 'overall_fail') {
    return aggregate.overallFailCount > 0
  }
  if (filterMode === 'localization_fail') {
    return aggregate.localizationFailCount > 0
  }
  if (filterMode === 'attribution_fail') {
    return aggregate.attributionFailCount > 0
  }
  if (filterMode === 'no_prediction') {
    return aggregate.noPredictionCount > 0
  }
  if (filterMode === 'needs_review') {
    return aggregate.needsReviewCount > 0
  }
  return aggregate.ruleCount > 0 && aggregate.needsReviewCount === 0
}

function compareExtractEvidenceDocumentOrder(left: ExtractEvidenceNode, right: ExtractEvidenceNode): number {
  const leftIndex = extractArrayIndexSortValue(left.label)
  const rightIndex = extractArrayIndexSortValue(right.label)
  if (leftIndex !== null && rightIndex !== null && leftIndex !== rightIndex) {
    return leftIndex - rightIndex
  }
  return 0
}

function sortExtractEvidenceChildren(
  children: ExtractEvidenceNode[],
  sortMode: ExtractEvidenceSortMode,
): ExtractEvidenceNode[] {
  const decorated = children.map((node, index) => ({
    node,
    index,
    aggregate: extractEvidenceAggregate(node),
  }))

  decorated.sort((left, right) => {
    if (sortMode === 'worst' && (extractNodeIsArrayRecord(left.node) || extractNodeIsArrayRecord(right.node))) {
      const leftWorst = left.aggregate.worstOverall ?? Number.POSITIVE_INFINITY
      const rightWorst = right.aggregate.worstOverall ?? Number.POSITIVE_INFINITY
      if (leftWorst !== rightWorst) {
        return leftWorst - rightWorst
      }
      if (left.aggregate.overallFailCount !== right.aggregate.overallFailCount) {
        return right.aggregate.overallFailCount - left.aggregate.overallFailCount
      }
      if (left.aggregate.needsReviewCount !== right.aggregate.needsReviewCount) {
        return right.aggregate.needsReviewCount - left.aggregate.needsReviewCount
      }
      if (left.aggregate.noPredictionCount !== right.aggregate.noPredictionCount) {
        return right.aggregate.noPredictionCount - left.aggregate.noPredictionCount
      }
    }

    const documentOrder = compareExtractEvidenceDocumentOrder(left.node, right.node)
    return documentOrder !== 0 ? documentOrder : left.index - right.index
  })

  return decorated.map(({ node }) => node)
}

function cloneExtractEvidenceNodeWithChildren(
  node: ExtractEvidenceNode,
  children: ExtractEvidenceNode[],
): ExtractEvidenceNode {
  const hasAnchor = node.anchors.rules.length > 0 || node.anchors.items.length > 0
  const anchoredLeafCount =
    children.length > 0 ? children.reduce((sum, child) => sum + child.anchoredLeafCount, 0) : hasAnchor ? 1 : 0
  return {
    ...node,
    children,
    anchoredLeafCount,
  }
}

function prepareExtractEvidenceNode(
  node: ExtractEvidenceNode,
  filterMode: ExtractEvidenceFilterMode,
  sortMode: ExtractEvidenceSortMode,
): ExtractEvidenceNode | null {
  const aggregate = extractEvidenceAggregate(node)
  const nodeMatchesFilter = extractEvidenceFilterMatches(aggregate, filterMode)
  const sortedChildren = sortExtractEvidenceChildren(node.children, sortMode)

  if (filterMode === 'all' || (extractNodeIsArrayRecord(node) && nodeMatchesFilter)) {
    return cloneExtractEvidenceNodeWithChildren(
      node,
      sortedChildren
        .map((child) => prepareExtractEvidenceNode(child, 'all', sortMode))
        .filter((child): child is ExtractEvidenceNode => child !== null),
    )
  }

  const filteredChildren = sortedChildren
    .map((child) => prepareExtractEvidenceNode(child, filterMode, sortMode))
    .filter((child): child is ExtractEvidenceNode => child !== null)

  if (!nodeMatchesFilter && filteredChildren.length === 0) {
    return null
  }

  return cloneExtractEvidenceNodeWithChildren(node, filteredChildren)
}

function extractNodeTypeLabel(node: ExtractEvidenceNode): string {
  if (Array.isArray(node.value)) {
    return 'array'
  }
  if (node.value === undefined) {
    return node.children.length > 0 ? 'object' : 'missing'
  }
  if (node.value === null) {
    return 'null'
  }
  return typeof node.value
}

function JsonLeaf({ value }: { value: JsonTreeValue }) {
  if (value === null) {
    return <span className="json-token json-null">null</span>
  }
  if (typeof value === 'string') {
    return <span className="json-token json-string">"{value}"</span>
  }
  if (typeof value === 'number') {
    return <span className="json-token json-number">{value}</span>
  }
  if (typeof value === 'boolean') {
    return <span className="json-token json-boolean">{String(value)}</span>
  }
  return <span className="json-token json-unknown">{String(value)}</span>
}

function JsonNode({
  label,
  value,
  path,
  depth,
  expandedPaths,
  onToggle,
}: {
  label: string | null
  value: JsonTreeValue
  path: string
  depth: number
  expandedPaths: Set<string>
  onToggle: (path: string) => void
}) {
  const isArray = Array.isArray(value)
  const isObject = value !== null && typeof value === 'object' && !isArray
  const isBranch = isArray || isObject
  const expanded = expandedPaths.has(path)
  const entries = isArray
    ? value.map((entry, index) => [String(index), entry] as const)
    : isObject
      ? Object.entries(value)
      : []

  return (
    <div className="json-node" style={{ '--json-depth': depth } as CSSProperties}>
      <div className="json-row">
        {isBranch ? (
          <button className="json-toggle" onClick={() => onToggle(path)} aria-label={expanded ? 'Collapse' : 'Expand'}>
            {expanded ? '▾' : '▸'}
          </button>
        ) : (
          <span className="json-toggle-spacer" />
        )}
        {label !== null ? (
          <>
            <span className="json-key">"{label}"</span>
            <span className="json-colon">:</span>
          </>
        ) : null}
        {isBranch ? (
          <button className="json-branch" onClick={() => onToggle(path)}>
            <span className="json-bracket">{isArray ? '[' : '{'}</span>
            <span className="json-summary">{summarizeJsonValue(value)}</span>
            <span className="json-bracket">{isArray ? ']' : '}'}</span>
          </button>
        ) : (
          <JsonLeaf value={value} />
        )}
      </div>
      {isBranch && expanded ? (
        <div className="json-children">
          {entries.map(([childLabel, childValue]) => (
            <JsonNode
              key={`${path}.${childLabel}`}
              label={childLabel}
              value={childValue}
              path={`${path}.${childLabel}`}
              depth={depth + 1}
              expandedPaths={expandedPaths}
              onToggle={onToggle}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}

function JsonPane({ rawJson }: { rawJson: string | null }) {
  const parsed = useMemo(() => {
    if (!rawJson) {
      return { ok: false, value: null as JsonTreeValue | null }
    }
    try {
      return { ok: true, value: JSON.parse(rawJson) as JsonTreeValue }
    } catch {
      return { ok: false, value: null as JsonTreeValue | null }
    }
  }, [rawJson])
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(() => new Set(['root']))

  if (!rawJson) {
    return <div className="json-pane-empty">No JSON payload available.</div>
  }

  if (!parsed.ok) {
    return <pre className="json-view">{rawJson}</pre>
  }

  const togglePath = (path: string) => {
    setExpandedPaths((current) => {
      const next = new Set(current)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  return (
    <div className="json-pane">
      <JsonNode label={null} value={parsed.value} path="root" depth={0} expandedPaths={expandedPaths} onToggle={togglePath} />
    </div>
  )
}

function ElementsList({
  items,
  activeItemId,
  hoveredItemId,
  hoverSource,
  onHoverItem,
  onSelectItem,
  listRef,
}: {
  items: GroundingItem[]
  activeItemId: string | null
  hoveredItemId: string | null
  hoverSource: 'viewer' | 'sidebar' | null
  onHoverItem: (itemId: string | null) => void
  onSelectItem: (itemId: string) => void
  listRef: RefObject<HTMLUListElement | null>
}) {
  const [manualExpandedItems, setManualExpandedItems] = useState<Record<string, boolean>>({})
  const [copiedItemId, setCopiedItemId] = useState<string | null>(null)
  const [sortMode, setSortMode] = useState<ElementSortMode>('default')
  const autoExpandedItemId = hoveredItemId ?? null

  const sortedItems = useMemo(() => {
    const withArea = items.map((item) => ({
      item,
      bboxArea: item.bboxes.reduce((sum, bbox) => sum + bbox.w * bbox.h, 0),
    }))

    if (sortMode === 'bbox_desc') {
      withArea.sort((a, b) => {
        if (b.bboxArea !== a.bboxArea) {
          return b.bboxArea - a.bboxArea
        }
        return a.item.item_index - b.item.item_index
      })
    } else if (sortMode === 'bbox_asc') {
      withArea.sort((a, b) => {
        if (a.bboxArea !== b.bboxArea) {
          return a.bboxArea - b.bboxArea
        }
        return a.item.item_index - b.item.item_index
      })
    } else {
      withArea.sort((a, b) => a.item.item_index - b.item.item_index)
    }

    return withArea
  }, [items, sortMode])

  useEffect(() => {
    if (!copiedItemId) {
      return
    }

    const timeoutId = window.setTimeout(() => setCopiedItemId(null), 1200)
    return () => window.clearTimeout(timeoutId)
  }, [copiedItemId])

  const toggleExpanded = (itemId: string) => {
    setManualExpandedItems((prev) => ({
      ...prev,
      [itemId]: !prev[itemId],
    }))
  }

  const copyItemJson = async (item: GroundingItem) => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(item.raw_payload ?? null, null, 2))
      setCopiedItemId(item.item_id)
    } catch {
      setCopiedItemId(null)
    }
  }

  return (
    <>
      <div className="elements-toolbar">
        <label htmlFor="elements-sort">Sort</label>
        <select
          id="elements-sort"
          value={sortMode}
          onChange={(event) => setSortMode(event.target.value as ElementSortMode)}
        >
          <option value="default">Default</option>
          <option value="bbox_desc">BBox size: largest first</option>
          <option value="bbox_asc">BBox size: smallest first</option>
        </select>
      </div>
      <ul className="elements-list" ref={listRef}>
        {sortedItems.map(({ item, bboxArea }) => {
          const active = item.item_id === activeItemId || item.item_id === hoveredItemId
          const viewerFocused = hoverSource === 'viewer' && item.item_id === hoveredItemId
          const expanded = Boolean(manualExpandedItems[item.item_id]) || autoExpandedItemId === item.item_id
          const className = [active ? 'element-row active' : 'element-row', viewerFocused ? 'viewer-focus' : '']
            .filter(Boolean)
            .join(' ')
          return (
            <li key={item.item_id}>
              <article className="element-card">
                <button
                  className={className}
                  data-item-id={item.item_id}
                  onMouseEnter={() => onHoverItem(item.item_id)}
                  onMouseLeave={() => onHoverItem(null)}
                  onClick={() => {
                    onSelectItem(item.item_id)
                    toggleExpanded(item.item_id)
                  }}
                >
                  <span className="element-type">{item.type}</span>
                  <span className="element-label">
                    ro:{item.item_index} · bbox:{item.bboxes.length} · area:{Math.round(bboxArea)}{' '}
                    {expanded ? '▾' : '▸'}
                  </span>
                </button>
                {expanded ? (
                  <div className="element-json-panel">
                    <div className="element-json-header">
                      <span>raw_payload</span>
                      <button
                        className="element-copy-button"
                        onClick={(event) => {
                          event.stopPropagation()
                          void copyItemJson(item)
                        }}
                      >
                        {copiedItemId === item.item_id ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                    <pre className="element-json">{JSON.stringify(item.raw_payload ?? null, null, 2)}</pre>
                  </div>
                ) : null}
              </article>
            </li>
          )
        })}
      </ul>
    </>
  )
}

function GranularPane({
  layers,
  activeUnit,
  hoveredUnit,
  hoverSource,
  onHoverGranularUnit,
  onSelectGranularUnit,
  listRef,
}: {
  layers: GroundingGranularLayer[]
  activeUnit: GroundingGranularUnit | null
  hoveredUnit: GroundingGranularUnit | null
  hoverSource: 'viewer' | 'sidebar' | null
  onHoverGranularUnit: (unitId: string | null, granularity: GroundingGranularity | null) => void
  onSelectGranularUnit: (unitId: string, granularity: GroundingGranularity) => void
  listRef: RefObject<HTMLDivElement | null>
}) {
  const [filterMode, setFilterMode] = useState<GranularFilterMode>('all')

  const filteredLayers = useMemo(() => {
    if (filterMode === 'all') {
      return layers
    }
    const match = layers.find((layer) => layer.granularity === filterMode)
    return match ? [match] : []
  }, [filterMode, layers])

  const focusedUnit = hoveredUnit ?? activeUnit

  return (
    <div className="granular-pane">
      <div className="granular-toolbar">
        <label htmlFor="granular-filter">Show</label>
        <select
          id="granular-filter"
          value={filterMode}
          onChange={(event) => setFilterMode(event.target.value as GranularFilterMode)}
        >
          <option value="all">All layers</option>
          <option value="line">Lines</option>
          <option value="word">Words</option>
          <option value="cell">Cells</option>
        </select>
      </div>

      <div className="granular-summary-grid">
        {(['line', 'word', 'cell'] as const).map((granularity) => {
          const layer =
            layers.find((candidate) => candidate.granularity === granularity) ??
            ({
              granularity,
              availability: 'unavailable',
              units: [],
              reason: `No ${granularity} overlays were returned for this page.`,
              source: null,
            } satisfies GroundingGranularLayer)
          const className = [
            'granular-summary-card',
            `layer-${granularity}`,
            filterMode === granularity ? 'active' : '',
            layer.availability === 'unavailable' ? 'disabled' : '',
          ]
            .filter(Boolean)
            .join(' ')
          return (
            <button
              key={granularity}
              className={className}
              onClick={() => setFilterMode((current) => (current === granularity ? 'all' : granularity))}
              disabled={layer.availability === 'unavailable'}
              title={layerDescription(layer)}
            >
              <span className="granular-summary-label">{granularity}</span>
              <strong>{layer.availability === 'unavailable' ? 'n/a' : layer.units.length}</strong>
              <span className="granular-summary-meta">
                {layer.availability === 'available' ? layer.source ?? 'normalized' : layer.availability}
              </span>
            </button>
          )
        })}
      </div>

      <div className="granular-selection-card">
        {focusedUnit ? (
          <>
            <header>
              <span>{formatGranularUnitLabel(focusedUnit)}</span>
              <strong>#{focusedUnit.order_index}</strong>
            </header>
            <p>{previewText(focusedUnit.text)}</p>
            <dl className="granular-detail-grid">
              <div>
                <dt>bbox</dt>
                <dd>{summarizeBbox(focusedUnit)}</dd>
              </div>
              <div>
                <dt>provider</dt>
                <dd>{focusedUnit.provider ?? 'normalized'}</dd>
              </div>
              <div>
                <dt>source</dt>
                <dd>{focusedUnit.source_path ?? 'n/a'}</dd>
              </div>
              <div>
                <dt>meta</dt>
                <dd>{formatGranularUnitMetadata(focusedUnit) ?? 'n/a'}</dd>
              </div>
            </dl>
          </>
        ) : (
          <p className="granular-empty-state">Hover or click a line, word, or cell overlay to inspect it here.</p>
        )}
      </div>

      <div className="granular-layer-list" ref={listRef}>
        {filteredLayers.map((layer) => {
          const viewerFocused = hoverSource === 'viewer' && hoveredUnit?.granularity === layer.granularity
          return (
            <section key={layer.granularity} className="granular-layer-section">
              <header className="granular-layer-header">
                <div>
                  <h4>{layer.granularity}</h4>
                  <span>{layerDescription(layer)}</span>
                </div>
                <span className={viewerFocused ? 'granular-layer-badge viewer-focus' : 'granular-layer-badge'}>
                  {layer.source ?? layer.availability}
                </span>
              </header>

              {layer.availability === 'unavailable' ? (
                <div className="granular-layer-note">{layer.reason ?? 'Unavailable on this page.'}</div>
              ) : null}
              {layer.availability === 'empty' ? (
                <div className="granular-layer-note">No units for this page.</div>
              ) : null}

              {layer.availability === 'available' ? (
                <ul className="granular-unit-list">
                  {layer.units.map((unit) => {
                    const active = unit.unit_id === activeUnit?.unit_id || unit.unit_id === hoveredUnit?.unit_id
                    const className = [
                      'granular-unit-row',
                      `layer-${unit.granularity}`,
                      active ? 'active' : '',
                      hoverSource === 'viewer' && hoveredUnit?.unit_id === unit.unit_id ? 'viewer-focus' : '',
                    ]
                      .filter(Boolean)
                      .join(' ')

                    return (
                      <li key={unit.unit_id}>
                        <button
                          className={className}
                          data-granular-id={unit.unit_id}
                          onMouseEnter={() => onHoverGranularUnit(unit.unit_id, unit.granularity)}
                          onMouseLeave={() => onHoverGranularUnit(null, null)}
                          onClick={() => onSelectGranularUnit(unit.unit_id, unit.granularity)}
                        >
                          <div className="granular-unit-main">
                            <span className="granular-unit-label">
                              {formatGranularUnitLabel(unit)} · #{unit.order_index}
                            </span>
                            <span className="granular-unit-preview">{previewText(unit.text)}</span>
                          </div>
                          <span className="granular-unit-meta">
                            {formatGranularUnitMetadata(unit) ?? summarizeBbox(unit)}
                          </span>
                        </button>
                      </li>
                    )
                  })}
                </ul>
              ) : null}
            </section>
          )
        })}
      </div>
    </div>
  )
}

function collectExtractBranchPaths(node: ExtractEvidenceNode, target: Set<string>) {
  if (!extractNodeIsBranch(node)) {
    return
  }
  target.add(node.path)
  node.children.forEach((child) => collectExtractBranchPaths(child, target))
}

function extractDisplayLabel(node: ExtractEvidenceNode): string {
  if (node.label === null) {
    return 'extracted_data'
  }
  return node.path.endsWith(`[${node.label}]`) ? `[${node.label}]` : node.label
}

function extractEvidenceMetricRows(rule: GroundTruthRuleMatch): Array<[string, string]> {
  const rows: Array<[string, string]> = [
    ['Overall', gtStatusCopy(rule.overall_pass ?? null)],
    ['Loc', gtStatusCopy(rule.localization_pass ?? null)],
    ['Class', gtStatusCopy(rule.classification_pass ?? null)],
    ['Attr', gtStatusCopy(rule.attribution_pass ?? null)],
    ['IoU', formatRulePercent(rule.iou ?? null)],
  ]

  if (rule.text_score !== null && rule.text_score !== undefined) {
    rows.push(['Text', formatRulePercent(rule.text_score)])
  }
  if (rule.bbox_recall !== null && rule.bbox_recall !== undefined) {
    rows.push(['BBox recall', formatRulePercent(rule.bbox_recall)])
  }
  if (rule.predicted_granularity) {
    rows.push(['Granularity', rule.predicted_granularity])
  }
  if (rule.predicted_bboxes.length > 0) {
    rows.push(['Pred bboxes', String(rule.predicted_bboxes.length)])
  }
  if ((rule.matched_unit_ids ?? []).length > 0) {
    rows.push(['Matched units', String((rule.matched_unit_ids ?? []).length)])
  }
  if (rule.localization_reason) {
    rows.push(['Loc reason', rule.localization_reason])
  }
  if (rule.attribution_reason) {
    rows.push(['Attr reason', rule.attribution_reason])
  }

  return rows
}

function ExtractEvidenceTreeNode({
  node,
  depth,
  expandedPaths,
  expandedDetailPaths,
  onToggle,
  onToggleDetails,
  activeItemId,
  hoveredItemId,
  activeRule,
  hoveredRule,
  onHoverEvidence,
  onSelectEvidence,
}: {
  node: ExtractEvidenceNode
  depth: number
  expandedPaths: Set<string>
  expandedDetailPaths: Set<string>
  onToggle: (path: string) => void
  onToggleDetails: (path: string) => void
  activeItemId: string | null
  hoveredItemId: string | null
  activeRule: GroundTruthRuleMatch | null
  hoveredRule: GroundTruthRuleMatch | null
  onHoverEvidence: (itemId: string | null, ruleIds: string[]) => void
  onSelectEvidence: (itemId: string | null, ruleIds: string[]) => void
}) {
  const branch = extractNodeIsBranch(node)
  const aggregate = extractEvidenceAggregate(node)
  const expanded = expandedPaths.has(node.path)
  const detailExpanded = expandedDetailPaths.has(node.path)
  const item = firstAnchorItem(node.anchors)
  const rule = firstAnchorRule(node.anchors)
  const ruleIds = node.anchors.rules.map((candidate) => candidate.rule_id)
  const hasEvidence = Boolean(item || rule)
  const active =
    item?.item_id === activeItemId ||
    item?.item_id === hoveredItemId ||
    ruleIds.some((ruleId) => ruleId === activeRule?.rule_id) ||
    ruleIds.some((ruleId) => ruleId === hoveredRule?.rule_id)
  const className = [
    'extract-evidence-row',
    branch ? 'branch' : 'leaf',
    active ? 'active' : '',
    node.anchors.items.length === 0 && node.anchors.rules.length > 0 ? 'missing-prediction' : '',
    aggregate.needsReviewCount > 0 ? 'needs-review' : '',
    aggregate.overallFailCount > 0 ? 'has-fails' : '',
  ]
    .filter(Boolean)
    .join(' ')
  const ruleMetric = rule ? gtRuleMetricValue(rule, 'overall') : null
  const expectedValue = rule ? formatRuleValue(rule.expected_value) : 'n/a'
  const predictedValue = formatExtractJsonValue(node.value)

  const handleHover = (entering: boolean) => {
    if (!hasEvidence) {
      return
    }
    onHoverEvidence(entering ? item?.item_id ?? null : null, entering ? ruleIds : [])
  }

  return (
    <div className="extract-evidence-node" style={{ '--extract-depth': depth } as CSSProperties}>
      <button
        className={className}
        data-item-id={item?.item_id}
        data-gt-rule-ids={ruleIds.join(' ')}
        title={node.path || 'extracted_data'}
        onMouseEnter={() => handleHover(true)}
        onMouseLeave={() => handleHover(false)}
        onClick={() => {
          if (branch) {
            onToggle(node.path)
          } else if (hasEvidence) {
            onToggleDetails(node.path)
          }
          if (hasEvidence) {
            onSelectEvidence(item?.item_id ?? null, ruleIds)
          }
        }}
      >
        <span className="extract-evidence-toggle">
          {branch ? (expanded ? '▾' : '▸') : hasEvidence ? detailExpanded ? '▾' : '▸' : ''}
        </span>
        <span className="extract-evidence-key">{extractDisplayLabel(node)}</span>
        <span className="extract-evidence-type">{extractNodeTypeLabel(node)}</span>
        <span className="extract-evidence-value">{branch ? `${node.anchoredLeafCount} fields` : predictedValue}</span>
        <span className="extract-evidence-chips">
          {aggregate.overallFailCount > 0 ? (
            <span className="extract-evidence-chip bad">{aggregate.overallFailCount} fail</span>
          ) : null}
          {aggregate.needsReviewCount > 0 ? (
            <span className="extract-evidence-chip warn">
              <span className="extract-evidence-status-dot" />
              {aggregate.needsReviewCount} review
            </span>
          ) : null}
          {branch && aggregate.ruleCount > 0 && aggregate.needsReviewCount === 0 ? (
            <span className="extract-evidence-chip good">verified</span>
          ) : null}
          {branch && aggregate.worstOverall !== null ? (
            <span className={`extract-evidence-chip score-inline-${gtScoreTone(aggregate.worstOverall)}`}>
              Worst {formatRulePercent(aggregate.worstOverall)}
            </span>
          ) : null}
          {node.anchors.items.length > 0 ? (
            <span className="extract-evidence-chip good">
              {node.anchors.items.length} pred bbox{node.anchors.items.length === 1 ? '' : 'es'}
            </span>
          ) : null}
          {node.anchors.rules.length > 0 ? (
            <span className="extract-evidence-chip">{node.anchors.rules.length} GT</span>
          ) : null}
          {rule ? (
            <span className={`extract-evidence-chip score-inline-${gtScoreTone(ruleMetric)}`}>
              Overall {formatRulePercent(ruleMetric)}
            </span>
          ) : null}
        </span>
      </button>
      {!branch && (rule || item) ? (
        <div className="extract-evidence-detail">
          <div>
            <span>Expected</span>
            <strong>{expectedValue}</strong>
          </div>
          <div>
            <span>Pred</span>
            <strong className={node.value === undefined ? 'missing' : ''}>{predictedValue}</strong>
          </div>
          {rule && detailExpanded ? (
            <div className="extract-evidence-metrics">
              {extractEvidenceMetricRows(rule).map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
      {branch && expanded ? (
        <div className="extract-evidence-children">
          {node.children.map((child) => (
            <ExtractEvidenceTreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              expandedPaths={expandedPaths}
              expandedDetailPaths={expandedDetailPaths}
              onToggle={onToggle}
              onToggleDetails={onToggleDetails}
              activeItemId={activeItemId}
              hoveredItemId={hoveredItemId}
              activeRule={activeRule}
              hoveredRule={hoveredRule}
              onHoverEvidence={onHoverEvidence}
              onSelectEvidence={onSelectEvidence}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}

function ExtractEvidenceTree({
  rootNode,
  activeItemId,
  hoveredItemId,
  activeRule,
  hoveredRule,
  onHoverEvidence,
  onSelectEvidence,
  listRef,
}: {
  rootNode: ExtractEvidenceNode
  activeItemId: string | null
  hoveredItemId: string | null
  activeRule: GroundTruthRuleMatch | null
  hoveredRule: GroundTruthRuleMatch | null
  onHoverEvidence: (itemId: string | null, ruleIds: string[]) => void
  onSelectEvidence: (itemId: string | null, ruleIds: string[]) => void
  listRef: RefObject<HTMLDivElement | null>
}) {
  const [collapsedPaths, setCollapsedPaths] = useState<Set<string>>(() => new Set())
  const [expandedDetailPaths, setExpandedDetailPaths] = useState<Set<string>>(() => new Set())
  const [filterMode, setFilterMode] = useState<ExtractEvidenceFilterMode>('all')
  const [sortMode, setSortMode] = useState<ExtractEvidenceSortMode>('document')
  const rootAggregate = useMemo(() => extractEvidenceAggregate(rootNode), [rootNode])
  const visibleRootNode = useMemo(
    () => prepareExtractEvidenceNode(rootNode, filterMode, sortMode),
    [filterMode, rootNode, sortMode],
  )
  const branchPaths = useMemo(() => {
    const next = new Set<string>()
    if (visibleRootNode) {
      collectExtractBranchPaths(visibleRootNode, next)
    }
    return next
  }, [visibleRootNode])
  const expandedPaths = useMemo(
    () => new Set(Array.from(branchPaths).filter((path) => !collapsedPaths.has(path))),
    [branchPaths, collapsedPaths],
  )

  const togglePath = (path: string) => {
    setCollapsedPaths((current) => {
      const next = new Set(current)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  const toggleDetails = (path: string) => {
    setExpandedDetailPaths((current) => {
      const next = new Set(current)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  return (
    <div className="extract-evidence-pane" ref={listRef}>
      <div className="extract-evidence-summary">
        <strong>Extracted JSON</strong>
        <span>
          {rootNode.anchoredLeafCount} anchored fields · {rootAggregate.overallFailCount} failing ·{' '}
          {rootAggregate.needsReviewCount} review
        </span>
      </div>
      <div className="extract-evidence-controls">
        <label htmlFor="extract-evidence-filter">Show</label>
        <select
          id="extract-evidence-filter"
          value={filterMode}
          onChange={(event) => setFilterMode(event.target.value as ExtractEvidenceFilterMode)}
        >
          <option value="all">All anchored fields</option>
          <option value="overall_fail">Overall fails</option>
          <option value="localization_fail">Loc fails</option>
          <option value="attribution_fail">Attr fails</option>
          <option value="no_prediction">No prediction</option>
          <option value="needs_review">Needs review</option>
          <option value="verified">Verified only</option>
        </select>
        <label htmlFor="extract-evidence-sort">Sort</label>
        <select
          id="extract-evidence-sort"
          value={sortMode}
          onChange={(event) => setSortMode(event.target.value as ExtractEvidenceSortMode)}
        >
          <option value="document">Document order</option>
          <option value="worst">Worst records first</option>
        </select>
      </div>
      {visibleRootNode ? (
        <ExtractEvidenceTreeNode
          node={visibleRootNode}
          depth={0}
          expandedPaths={expandedPaths}
          expandedDetailPaths={expandedDetailPaths}
          onToggle={togglePath}
          onToggleDetails={toggleDetails}
          activeItemId={activeItemId}
          hoveredItemId={hoveredItemId}
          activeRule={activeRule}
          hoveredRule={hoveredRule}
          onHoverEvidence={onHoverEvidence}
          onSelectEvidence={onSelectEvidence}
        />
      ) : (
        <div className="extract-evidence-empty">No extract fields match the current filter.</div>
      )}
    </div>
  )
}

function GtPane({
  document,
  rules,
  pageItems,
  activeItemId,
  hoveredItemId,
  activeRule,
  hoveredRule,
  onHoverGtRule,
  onSelectGtRule,
  onHoverEvidence,
  onSelectEvidence,
  listRef,
}: {
  document: DocumentResponse
  rules: GroundTruthRuleMatch[]
  pageItems: GroundingItem[]
  activeItemId: string | null
  hoveredItemId: string | null
  activeRule: GroundTruthRuleMatch | null
  hoveredRule: GroundTruthRuleMatch | null
  onHoverGtRule: (ruleId: string | null) => void
  onSelectGtRule: (ruleId: string) => void
  onHoverEvidence: (itemId: string | null, ruleIds: string[]) => void
  onSelectEvidence: (itemId: string | null, ruleIds: string[]) => void
  listRef: RefObject<HTMLDivElement | null>
}) {
  const [sortDirection, setSortDirection] = useState<GtSortDirection>('lowest')
  const availableRuleTypes = useMemo(() => Array.from(new Set(rules.map((rule) => rule.rule_type))), [rules])
  const [manualSelectedRuleType, setManualSelectedRuleType] = useState<GtRuleType>('extract_field')
  const [fieldSortMetric, setFieldSortMetric] = useState<GtFieldSortMetric>('overall')
  const [layoutSortMetric, setLayoutSortMetric] = useState<GtLayoutSortMetric>('overall')
  const [extractViewMode, setExtractViewMode] = useState<ExtractViewMode>('json')
  const [expandedRuleIds, setExpandedRuleIds] = useState<Set<string>>(() => new Set())

  const selectedRuleType = useMemo(() => {
    const focusedType = hoveredRule?.rule_type ?? activeRule?.rule_type ?? null
    if (focusedType && availableRuleTypes.includes(focusedType)) {
      return focusedType
    }
    if (availableRuleTypes.includes(manualSelectedRuleType)) {
      return manualSelectedRuleType
    }
    return (availableRuleTypes[0] ?? manualSelectedRuleType) as GtRuleType
  }, [activeRule, availableRuleTypes, hoveredRule, manualSelectedRuleType])

  const filteredRules = useMemo(
    () => rules.filter((rule) => rule.rule_type === selectedRuleType),
    [rules, selectedRuleType],
  )
  const sortMetric: GtSortMetric = selectedRuleType === 'layout' ? layoutSortMetric : fieldSortMetric
  const extractEvidenceRoot = useMemo(() => {
    if (selectedRuleType !== 'extract_field') {
      return null
    }
    const extractedData = parseExtractedData(document.result_json)
    if (!extractedData) {
      return null
    }
    const anchors = buildExtractEvidenceAnchors(filteredRules, pageItems)
    if (anchors.size === 0) {
      return null
    }
    return buildExtractEvidenceNodeFromAnchors(extractedData, anchors)
  }, [document.result_json, filteredRules, pageItems, selectedRuleType])
  const effectiveExtractViewMode: ExtractViewMode = extractEvidenceRoot ? extractViewMode : 'rules'

  const sortedRules = useMemo(() => {
    const decorated = filteredRules.map((rule, index) => ({
      rule,
      metricValue: gtRuleMetricValue(rule, sortMetric),
      index,
    }))

    decorated.sort((left, right) => {
      const leftMissing = left.metricValue === null || Number.isNaN(left.metricValue)
      const rightMissing = right.metricValue === null || Number.isNaN(right.metricValue)
      if (leftMissing !== rightMissing) {
        return leftMissing ? 1 : -1
      }
      const leftValue = left.metricValue ?? Number.NEGATIVE_INFINITY
      const rightValue = right.metricValue ?? Number.NEGATIVE_INFINITY
      if (leftValue !== rightValue) {
        return sortDirection === 'lowest' ? leftValue - rightValue : rightValue - leftValue
      }
      return left.index - right.index
    })

    return decorated
  }, [filteredRules, sortDirection, sortMetric])

  const toggleRuleExpanded = (ruleId: string) => {
    setExpandedRuleIds((current) => {
      const next = new Set(current)
      if (next.has(ruleId)) {
        next.delete(ruleId)
      } else {
        next.add(ruleId)
      }
      return next
    })
  }

  return (
    <div className="gt-pane">
      <div className="gt-toolbar">
        {availableRuleTypes.length > 1 ? (
          <>
            <label htmlFor="gt-type">Type</label>
            <select
              id="gt-type"
              value={selectedRuleType}
              onChange={(event) => setManualSelectedRuleType(event.target.value as GtRuleType)}
            >
              {availableRuleTypes.map((ruleType) => (
                <option key={ruleType} value={ruleType}>
                  {gtRuleTypeLabel(ruleType)}
                </option>
              ))}
            </select>
          </>
        ) : null}
        <label htmlFor="gt-sort-direction">Sort</label>
        <select
          id="gt-sort-direction"
          value={sortDirection}
          onChange={(event) => setSortDirection(event.target.value as GtSortDirection)}
        >
          <option value="lowest">Lowest</option>
          <option value="highest">Highest</option>
        </select>
        <select
          id="gt-sort-metric"
          value={sortMetric}
          disabled={effectiveExtractViewMode === 'json'}
          onChange={(event) => {
            if (selectedRuleType === 'layout') {
              setLayoutSortMetric(event.target.value as GtLayoutSortMetric)
            } else {
              setFieldSortMetric(event.target.value as GtFieldSortMetric)
            }
          }}
        >
          {selectedRuleType === 'layout' ? (
            <>
              <option value="overall">Overall</option>
              <option value="localization">Localization</option>
              <option value="classification">Classification</option>
              <option value="attribution">Attribution</option>
              <option value="iou">IoU</option>
            </>
          ) : (
            <>
              <option value="overall">Overall</option>
              <option value="localization">Localization</option>
              <option value="attribution">Attribution</option>
              <option value="iou">IoU</option>
              <option value="text_score">Text score</option>
              <option value="f1">Geometry F1</option>
              <option value="recall">Geometry recall</option>
              <option value="precision">Geometry precision</option>
            </>
          )}
        </select>
        {selectedRuleType === 'extract_field' && extractEvidenceRoot ? (
          <div className="gt-view-toggle" aria-label="Extract field view">
            <button
              className={effectiveExtractViewMode === 'json' ? 'active' : ''}
              onClick={() => setExtractViewMode('json')}
            >
              Extract JSON
            </button>
            <button
              className={effectiveExtractViewMode === 'rules' ? 'active' : ''}
              onClick={() => setExtractViewMode('rules')}
            >
              Rule rows
            </button>
          </div>
        ) : null}
      </div>

      {effectiveExtractViewMode === 'json' && extractEvidenceRoot ? (
        <ExtractEvidenceTree
          rootNode={extractEvidenceRoot}
          activeItemId={activeItemId}
          hoveredItemId={hoveredItemId}
          activeRule={activeRule}
          hoveredRule={hoveredRule}
          onHoverEvidence={onHoverEvidence}
          onSelectEvidence={onSelectEvidence}
          listRef={listRef}
        />
      ) : (
        <div className="gt-rule-list" ref={listRef}>
        {sortedRules.map(({ rule, metricValue }) => {
          const active = rule.rule_id === activeRule?.rule_id || rule.rule_id === hoveredRule?.rule_id
          const expanded = expandedRuleIds.has(rule.rule_id)
          const stray = ruleIsStray(rule)
          const className = [
            'gt-rule-row',
            active ? 'active' : '',
            rule.predicted_bbox ? 'matched' : 'unmatched',
            stray ? 'stray' : '',
            rule.verified === false ? 'unverified' : '',
          ]
            .filter(Boolean)
            .join(' ')
          return (
            <article
              key={rule.rule_id}
              className={className}
              data-gt-rule-id={rule.rule_id}
              onMouseEnter={() => onHoverGtRule(rule.rule_id)}
              onMouseLeave={() => onHoverGtRule(null)}
            >
              <button
                className="gt-rule-summary"
                onClick={() => {
                  onSelectGtRule(rule.rule_id)
                  toggleRuleExpanded(rule.rule_id)
                }}
              >
                <div className="gt-rule-main">
                  <span className="gt-rule-label">{rulePreviewLabel(rule)}</span>
                  <span className="gt-rule-submeta">{rulePreviewSubmeta(rule)}</span>
                </div>
                <div className="gt-rule-meta">
                  <span className={`score-inline score-inline-${gtScoreTone(metricValue)}`}>
                    {metricLabel(sortMetric)} {formatRulePercent(metricValue)}
                  </span>
                  <span className="gt-rule-chevron">{expanded ? '▾' : '▸'}</span>
                </div>
              </button>
              {expanded ? (
                <div className="gt-rule-details">
                  {rule.rule_type === 'layout' ? (
                    <>
                      <div className="gt-detail-chip-row">
                        <span className={`score-inline score-inline-${gtScoreTone(rule.overall_pass == null ? null : rule.overall_pass ? 1 : 0)}`}>
                          Overall {gtStatusCopy(rule.overall_pass ?? null)}
                        </span>
                        <span className={`score-inline score-inline-${gtScoreTone(rule.localization_pass == null ? null : rule.localization_pass ? 1 : 0)}`}>
                          Loc {gtStatusCopy(rule.localization_pass ?? null)}
                        </span>
                        <span className={`score-inline score-inline-${gtScoreTone(rule.classification_pass == null ? null : rule.classification_pass ? 1 : 0)}`}>
                          Class {gtStatusCopy(rule.classification_pass ?? null)}
                        </span>
                        <span
                          className={`score-inline score-inline-${gtScoreTone(
                            rule.attribution_applicable === false ? null : rule.attribution_pass == null ? null : rule.attribution_pass ? 1 : 0,
                          )}`}
                        >
                          Attr {rule.attribution_applicable === false ? 'n/a' : gtStatusCopy(rule.attribution_pass ?? null)}
                        </span>
                      </div>
                      <div className="gt-selection-copy-row">
                        <span className="gt-selection-copy-label">GT</span>
                        <span>{rule.canonical_class ?? 'n/a'}</span>
                      </div>
                      <div className="gt-selection-copy-row">
                        <span className="gt-selection-copy-label">Pred</span>
                        <span>{rule.predicted_class ?? 'n/a'}</span>
                      </div>
                      {rule.gt_text_norm ? (
                        <div className="gt-selection-copy-row">
                          <span className="gt-selection-copy-label">GT text</span>
                          <span>{previewText(rule.gt_text_norm)}</span>
                        </div>
                      ) : null}
                      {rule.predicted_text ? (
                        <div className="gt-selection-copy-row">
                          <span className="gt-selection-copy-label">Pred text</span>
                          <span>{previewText(rule.predicted_text)}</span>
                        </div>
                      ) : null}
                      {(rule.token_precision != null || rule.token_recall != null || rule.token_f1 != null) ? (
                        <div className="gt-selection-copy-row">
                          <span className="gt-selection-copy-label">Tokens</span>
                          <span>
                            P {formatRulePercent(rule.token_precision ?? null)} · R {formatRulePercent(rule.token_recall ?? null)} · F1{' '}
                            {formatRulePercent(rule.token_f1 ?? null)}
                          </span>
                        </div>
                      ) : null}
                      {(rule.missing_tokens ?? []).length > 0 ? (
                        <div className="gt-selection-copy-row">
                          <span className="gt-selection-copy-label">Missing</span>
                          <span>{previewText((rule.missing_tokens ?? []).join(', '))}</span>
                        </div>
                      ) : null}
                      {(rule.extra_tokens ?? []).length > 0 ? (
                        <div className="gt-selection-copy-row">
                          <span className="gt-selection-copy-label">Extra</span>
                          <span>{previewText((rule.extra_tokens ?? []).join(', '))}</span>
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <>
                      <div className="gt-detail-chip-row">
                        <span
                          className={`score-inline score-inline-${gtScoreTone(
                            rule.overall_pass == null ? null : rule.overall_pass ? 1 : 0,
                          )}`}
                        >
                          Overall {gtStatusCopy(rule.overall_pass ?? null)}
                        </span>
                        <span
                          className={`score-inline score-inline-${gtScoreTone(
                            rule.localization_pass == null ? null : rule.localization_pass ? 1 : 0,
                          )}`}
                        >
                          Loc {gtStatusCopy(rule.localization_pass ?? null)}
                        </span>
                        <span
                          className={`score-inline score-inline-${gtScoreTone(
                            rule.classification_pass == null ? null : rule.classification_pass ? 1 : 0,
                          )}`}
                        >
                          Class {gtStatusCopy(rule.classification_pass ?? null)}
                        </span>
                        <span
                          className={`score-inline score-inline-${gtScoreTone(
                            rule.attribution_pass == null ? null : rule.attribution_pass ? 1 : 0,
                          )}`}
                        >
                          Attr {gtStatusCopy(rule.attribution_pass ?? null)}
                        </span>
                        {stray ? <span className="score-inline score-inline-bad">stray</span> : null}
                        {rule.verified === false ? (
                          <span className="score-inline score-inline-warn">unverified</span>
                        ) : null}
                      </div>
                      {(rule.predicted_granularity ||
                        rule.iou != null ||
                        rule.text_score != null ||
                        rule.attribution_method ||
                        rule.attribution_reason ||
                        rule.localization_reason) ? (
                        <div className="gt-selection-copy-row">
                          <span className="gt-selection-copy-label">Metric</span>
                          <span>
                            {[
                              rule.predicted_granularity ? `granularity=${rule.predicted_granularity}` : null,
                              rule.iou != null ? `iou=${(rule.iou * 100).toFixed(1)}%` : null,
                              rule.text_score != null ? `text=${(rule.text_score * 100).toFixed(1)}%` : null,
                              rule.attribution_method ? `mode=${rule.attribution_method}` : null,
                              rule.attribution_reason ? `attr_reason=${rule.attribution_reason}` : null,
                              rule.localization_reason ? `loc_reason=${rule.localization_reason}` : null,
                            ]
                              .filter((part): part is string => part !== null)
                              .join(' · ')}
                          </span>
                        </div>
                      ) : null}
                      <div className="gt-selection-copy-row">
                        <span className="gt-selection-copy-label">Expected</span>
                        <span>{formatRuleValue(rule.expected_value)}</span>
                      </div>
                      <div className="gt-selection-copy-row">
                        <span className="gt-selection-copy-label">Pred</span>
                        <span>{rule.predicted_text ? previewText(rule.predicted_text) : 'n/a'}</span>
                      </div>
                      {typeof rule.expected_value === 'string' && rule.predicted_text ? (
                        <TextDiff expected={rule.expected_value} actual={rule.predicted_text} />
                      ) : null}
                    </>
                  )}
                </div>
              ) : null}
            </article>
          )
        })}
        </div>
      )}
    </div>
  )
}

export function RightPanel({
  document,
  pageItems,
  pageGranularLayers,
  pageGtRules,
  visibleLayers,
  activeItemId,
  hoveredItemId,
  activeGranularUnit,
  hoveredGranularUnit,
  activeGranularPreview,
  hoveredGranularPreview,
  activeGtRule,
  hoveredGtRule,
  hoverSource,
  onHoverItem,
  onSelectItem,
  onHoverGranularUnit,
  onSelectGranularUnit,
  onHoverGranularPreview,
  onSelectGranularPreview,
  onHoverGtRule,
  onSelectGtRule,
  onHoverEvidence,
  onSelectEvidence,
  onCollapse,
}: RightPanelProps) {
  const [manualTab, setManualTab] = useState<RightTab>('markdown')
  const elementsListRef = useRef<HTMLUListElement | null>(null)
  const granularListRef = useRef<HTMLDivElement | null>(null)
  const gtListRef = useRef<HTMLDivElement | null>(null)
  const previousActiveGtRuleIdRef = useRef<string | null>(null)

  const totalGtRules = useMemo(
    () => document.pages.reduce((sum, page) => sum + (page.gt_rules?.length ?? 0), 0),
    [document.pages],
  )
  const tabs = useMemo(() => {
    const nextTabs: RightTab[] = ['markdown', 'elements', 'granular']
    if (totalGtRules > 0) {
      nextTabs.push('gt')
    }
    if (document.raw_json) {
      nextTabs.push('raw')
    }
    if (document.result_json) {
      nextTabs.push('result')
    }
    return nextTabs
  }, [document.raw_json, document.result_json, totalGtRules])
  const tab: RightTab = tabs.includes(manualTab) ? manualTab : 'markdown'

  useEffect(() => {
    const nextRuleId = activeGtRule?.rule_id ?? null
    const previousRuleId = previousActiveGtRuleIdRef.current
    previousActiveGtRuleIdRef.current = nextRuleId

    if (!nextRuleId || nextRuleId === previousRuleId || !tabs.includes('gt')) {
      return
    }

    const timeoutId = window.setTimeout(() => setManualTab('gt'), 0)
    return () => window.clearTimeout(timeoutId)
  }, [activeGtRule, tabs])

  const totalGranularUnits = useMemo(
    () =>
      pageGranularLayers.reduce((sum, layer) => {
        return layer.availability === 'available' ? sum + layer.units.length : sum
      }, 0),
    [pageGranularLayers],
  )

  useEffect(() => {
    if (tab !== 'elements') {
      return
    }

    const targetId = hoveredItemId ?? activeItemId
    if (!targetId) {
      return
    }

    const button = elementsListRef.current?.querySelector(
      `button[data-item-id="${targetId}"]`,
    ) as HTMLButtonElement | null
    if (!button) {
      return
    }

    button.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [activeItemId, hoveredItemId, hoverSource, tab])

  useEffect(() => {
    if (tab !== 'granular') {
      return
    }

    const targetId = hoveredGranularUnit?.unit_id ?? activeGranularUnit?.unit_id
    if (!targetId) {
      return
    }

    const button = granularListRef.current?.querySelector(
      `button[data-granular-id="${targetId}"]`,
    ) as HTMLButtonElement | null
    if (!button) {
      return
    }

    button.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [activeGranularUnit, hoveredGranularUnit, hoverSource, tab])

  useEffect(() => {
    if (tab !== 'gt') {
      return
    }

    const targetId = activeGtRule?.rule_id
    if (!targetId) {
      return
    }

    const escapedTargetId = CSS.escape(targetId)
    const element = gtListRef.current?.querySelector(
      `[data-gt-rule-id="${escapedTargetId}"], [data-gt-rule-ids~="${escapedTargetId}"]`,
    ) as HTMLElement | null
    if (!element) {
      return
    }

    element.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [activeGtRule, tab])

  return (
    <div className="right-panel">
      <header className="panel-header">
        <div>
          <h3>Page Data</h3>
          <span>
            {pageItems.length} items · {totalGranularUnits} granular overlays · {pageGtRules.length} GT rules
          </span>
        </div>
        <button className="panel-collapse-button" onClick={onCollapse} aria-label="Collapse right sidebar">
          →
        </button>
      </header>

      <div className="tab-row">
        {tabs.map((tabName) => (
          <button
            key={tabName}
            className={tab === tabName ? 'tab active' : 'tab'}
            onClick={() => setManualTab(tabName)}
          >
            {tabName === 'gt' ? 'tests' : tabName}
          </button>
        ))}
      </div>

      {tab === 'markdown' ? (
        <ItemMarkdownPane
          items={pageItems}
          visibleLayers={visibleLayers}
          activeItemId={activeItemId}
          hoveredItemId={hoveredItemId}
          activeGranularPreview={activeGranularPreview}
          hoveredGranularPreview={hoveredGranularPreview}
          hoverSource={hoverSource}
          onHoverItem={onHoverItem}
          onSelectItem={onSelectItem}
          onHoverGranularPreview={onHoverGranularPreview}
          onSelectGranularPreview={onSelectGranularPreview}
        />
      ) : null}

      {tab === 'elements' ? (
        <ElementsList
          items={pageItems}
          activeItemId={activeItemId}
          hoveredItemId={hoveredItemId}
          hoverSource={hoverSource}
          onHoverItem={onHoverItem}
          onSelectItem={onSelectItem}
          listRef={elementsListRef}
        />
      ) : null}

      {tab === 'granular' ? (
        <GranularPane
          layers={pageGranularLayers}
          activeUnit={activeGranularUnit}
          hoveredUnit={hoveredGranularUnit}
          hoverSource={hoverSource}
          onHoverGranularUnit={onHoverGranularUnit}
          onSelectGranularUnit={onSelectGranularUnit}
          listRef={granularListRef}
        />
      ) : null}

      {tab === 'gt' ? (
        <GtPane
          document={document}
          rules={pageGtRules}
          pageItems={pageItems}
          activeItemId={activeItemId}
          hoveredItemId={hoveredItemId}
          activeRule={activeGtRule}
          hoveredRule={hoveredGtRule}
          onHoverGtRule={onHoverGtRule}
          onSelectGtRule={onSelectGtRule}
          onHoverEvidence={onHoverEvidence}
          onSelectEvidence={onSelectEvidence}
          listRef={gtListRef}
        />
      ) : null}

      {tab === 'raw' ? <JsonPane key={`raw:${document.doc_id}`} rawJson={document.raw_json} /> : null}
      {tab === 'result' ? <JsonPane key={`result:${document.doc_id}`} rawJson={document.result_json} /> : null}
    </div>
  )
}
