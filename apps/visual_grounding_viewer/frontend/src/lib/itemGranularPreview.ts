import type { OverlayLayerVisibility } from './grounding'
import type { GroundingBbox, GroundingGranularUnit, GroundingItem } from '../types/api'

export type ItemInteractionMode = 'cell' | 'line' | 'word' | null

export interface ItemInteractionData {
  mode: ItemInteractionMode
  cellUnits: GroundingGranularUnit[]
  lineUnits: GroundingGranularUnit[]
  wordUnits: GroundingGranularUnit[]
}

export interface MatchedTextUnit {
  unit: GroundingGranularUnit
  start: number
  end: number
}

interface LineContext {
  lineText: string
  lineBBox: GroundingBbox
  lineSpan: [number, number]
  sourceText: string
  rawWords: Array<Record<string, unknown>>
  key: string
}

const HTML_ENTITY_REPLACEMENTS: Record<string, string> = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'",
  '&nbsp;': ' ',
}

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }
  return value as Record<string, unknown>
}

function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function escapeForRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function decodeHtmlEntities(value: string): string {
  return value.replace(
    /&(amp|lt|gt|quot|#39|nbsp);/g,
    (entity) => HTML_ENTITY_REPLACEMENTS[entity] ?? entity,
  )
}

function extractTextFromHtml(value: string): string {
  return decodeHtmlEntities(
    value
      .replace(/<\s*br\s*\/?\s*>/gi, '\n')
      .replace(/<[^>]+>/g, ''),
  )
}

function normalizeGroundedText(value: string): string {
  const withBreaks = value.replace(/<\s*br\s*\/?\s*>/gi, '\n')
  if (/[<>]/.test(withBreaks)) {
    return extractTextFromHtml(withBreaks).trim()
  }
  return decodeHtmlEntities(withBreaks).trim()
}

function normalizeBboxPayload(value: unknown): GroundingBbox | null {
  const payload = asObject(value)
  if (!payload) {
    return null
  }
  const x = asNumber(payload.x)
  const y = asNumber(payload.y)
  const w = asNumber(payload.w)
  const h = asNumber(payload.h)
  if (x === null || y === null || w === null || h === null) {
    return null
  }
  return {
    x,
    y,
    w,
    h,
    label: null,
    confidence: null,
    start_index: null,
    end_index: null,
  }
}

function normalizeBboxPayloads(value: unknown): GroundingBbox[] {
  if (Array.isArray(value)) {
    return value
      .map((entry) => normalizeBboxPayload(entry))
      .filter((entry): entry is GroundingBbox => entry !== null)
  }
  const single = normalizeBboxPayload(value)
  return single ? [single] : []
}

function mergeBboxes(bboxes: GroundingBbox[]): GroundingBbox | null {
  if (bboxes.length === 0) {
    return null
  }
  const left = Math.min(...bboxes.map((bbox) => bbox.x))
  const top = Math.min(...bboxes.map((bbox) => bbox.y))
  const right = Math.max(...bboxes.map((bbox) => bbox.x + bbox.w))
  const bottom = Math.max(...bboxes.map((bbox) => bbox.y + bbox.h))
  return {
    x: left,
    y: top,
    w: Math.max(0, right - left),
    h: Math.max(0, bottom - top),
    label: null,
    confidence: null,
    start_index: null,
    end_index: null,
  }
}

function coerceSpan(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length !== 2) {
    return null
  }
  const start = asNumber(value[0])
  const end = asNumber(value[1])
  if (start === null || end === null) {
    return null
  }
  const normalizedStart = Math.trunc(start)
  const normalizedEnd = Math.trunc(end)
  if (normalizedEnd <= normalizedStart) {
    return null
  }
  return [normalizedStart, normalizedEnd]
}

function sliceSpanText(sourceText: string, span: [number, number]): string {
  const start = Math.max(0, span[0])
  const end = Math.min(sourceText.length, span[1])
  if (end <= start) {
    return ''
  }
  return sourceText.slice(start, end)
}

function resolveGroundingSourceText(rawNode: Record<string, unknown>, grounding: Record<string, unknown>): string {
  const sourceName = asString(grounding.source)
  if (sourceName === 'caption') {
    return asString(rawNode.caption)
  }
  if (sourceName === 'value') {
    return asString(rawNode.value)
  }
  return asString(rawNode.md) || asString(rawNode.value) || asString(rawNode.caption) || asString(rawNode.html)
}

function coerceCellText(cell: unknown): string {
  if (typeof cell === 'string') {
    return normalizeGroundedText(cell)
  }
  if (typeof cell === 'number' || typeof cell === 'boolean') {
    return String(cell)
  }
  const payload = asObject(cell)
  if (!payload) {
    return ''
  }
  return normalizeGroundedText(
    asString(payload.text) || asString(payload.md) || asString(payload.value) || asString(payload.html),
  )
}

function buildLineContexts(rawNode: Record<string, unknown>, itemId: string): LineContext[] {
  const contexts: LineContext[] = []
  const grounding = asObject(rawNode.grounding)
  if (!grounding) {
    return contexts
  }

  const sourceText = resolveGroundingSourceText(rawNode, grounding)
  for (const [lineIndex, rawLineEntry] of asList(grounding.lines).entries()) {
    const rawLine = asObject(rawLineEntry)
    if (!rawLine) {
      continue
    }
    const lineSpan = coerceSpan(rawLine.span)
    const lineBBox = normalizeBboxPayload(rawLine.bbox)
    if (!lineSpan || !lineBBox) {
      continue
    }
    const lineText = normalizeGroundedText(sliceSpanText(sourceText, lineSpan))
    if (!lineText) {
      continue
    }
    contexts.push({
      lineText,
      lineBBox,
      lineSpan,
      sourceText,
      rawWords: asList(rawLine.words).map((entry) => asObject(entry)).filter((entry): entry is Record<string, unknown> => entry !== null),
      key: `${itemId}:line:${lineIndex}`,
    })
  }

  const sourceRows = asList(rawNode.rows)
  const groundedRows = asList(grounding.rows)
  for (const [rowIndex, groundedRowEntry] of groundedRows.entries()) {
    const groundedRow = asList(groundedRowEntry)
    const sourceRow = asList(sourceRows[rowIndex])
    if (groundedRow.length === 0 || sourceRow.length === 0) {
      continue
    }
    for (const [columnIndex, groundedCellEntry] of groundedRow.entries()) {
      const groundedCell = asObject(groundedCellEntry)
      if (!groundedCell) {
        continue
      }
      const cellText = coerceCellText(sourceRow[columnIndex])
      if (!cellText) {
        continue
      }
      for (const [lineIndex, rawLineEntry] of asList(groundedCell.lines).entries()) {
        const rawLine = asObject(rawLineEntry)
        if (!rawLine) {
          continue
        }
        const lineSpan = coerceSpan(rawLine.span)
        const lineBBox = normalizeBboxPayload(rawLine.bbox)
        if (!lineSpan || !lineBBox) {
          continue
        }
        const lineText = normalizeGroundedText(sliceSpanText(cellText, lineSpan))
        if (!lineText) {
          continue
        }
        contexts.push({
          lineText,
          lineBBox,
          lineSpan,
          sourceText: cellText,
          rawWords: asList(rawLine.words).map((entry) => asObject(entry)).filter((entry): entry is Record<string, unknown> => entry !== null),
          key: `${itemId}:table-line:${rowIndex}:${columnIndex}:${lineIndex}`,
        })
      }
    }
  }

  return contexts
}

function buildLineUnits(lineContexts: LineContext[]): GroundingGranularUnit[] {
  return lineContexts.map((context, index) => ({
    unit_id: `${context.key}:${index}`,
    granularity: 'line',
    order_index: index,
    text: context.lineText,
    bbox: context.lineBBox,
    bboxes: [context.lineBBox],
    row_index: null,
    column_index: null,
    row_span: null,
    column_span: null,
    source_path: context.key,
    provider: 'llamaparse-item',
  }))
}

function iterateTokenSpans(sourceText: string, lineSpan: [number, number]): Array<[number, number]> {
  const lineText = sliceSpanText(sourceText, lineSpan)
  const matches = lineText.matchAll(/\S+/gu)
  return Array.from(matches, (match) => [lineSpan[0] + match.index!, lineSpan[0] + match.index! + match[0].length])
}

function buildWordUnits(lineContexts: LineContext[]): GroundingGranularUnit[] {
  const units: GroundingGranularUnit[] = []
  let orderIndex = 0

  for (const context of lineContexts) {
    for (const [tokenIndex, tokenSpan] of iterateTokenSpans(context.sourceText, context.lineSpan).entries()) {
      const matchingWordBboxes = context.rawWords
        .map((rawWord) => {
          const wordSpan = coerceSpan(rawWord.span)
          const wordBBox = normalizeBboxPayload(rawWord.bbox)
          if (!wordSpan || !wordBBox) {
            return null
          }
          if (wordSpan[1] <= tokenSpan[0] || wordSpan[0] >= tokenSpan[1]) {
            return null
          }
          return wordBBox
        })
        .filter((bbox): bbox is GroundingBbox => bbox !== null)

      if (matchingWordBboxes.length === 0) {
        continue
      }

      const bbox = mergeBboxes(matchingWordBboxes)
      if (!bbox) {
        continue
      }

      const tokenText = normalizeGroundedText(context.sourceText.slice(tokenSpan[0], tokenSpan[1]))
      if (!tokenText) {
        continue
      }

      units.push({
        unit_id: `${context.key}:word:${tokenIndex}`,
        granularity: 'word',
        order_index: orderIndex,
        text: tokenText,
        bbox,
        bboxes: matchingWordBboxes,
        row_index: null,
        column_index: null,
        row_span: null,
        column_span: null,
        source_path: context.key,
        provider: 'llamaparse-item',
      })
      orderIndex += 1
    }
  }

  return units
}

function buildCellUnits(item: GroundingItem): GroundingGranularUnit[] {
  const rawNode = asObject(item.raw_payload)
  if (!rawNode) {
    return []
  }
  const grounding = asObject(rawNode.grounding)
  if (!grounding) {
    return []
  }

  const sourceRows = asList(rawNode.rows)
  const groundedRows = asList(grounding.rows)
  const units: GroundingGranularUnit[] = []

  for (const [rowIndex, groundedRowEntry] of groundedRows.entries()) {
    const groundedRow = asList(groundedRowEntry)
    const sourceRow = asList(sourceRows[rowIndex])
    if (groundedRow.length === 0 || sourceRow.length === 0) {
      continue
    }

    for (const [columnIndex, groundedCellEntry] of groundedRow.entries()) {
      const groundedCell = asObject(groundedCellEntry)
      if (!groundedCell) {
        continue
      }

      let bboxes = normalizeBboxPayloads(groundedCell.bbox)
      if (bboxes.length === 0) {
        bboxes = asList(groundedCell.lines)
          .map((lineEntry) => asObject(lineEntry))
          .filter((lineEntry): lineEntry is Record<string, unknown> => lineEntry !== null)
          .map((lineEntry) => normalizeBboxPayload(lineEntry.bbox))
          .filter((bbox): bbox is GroundingBbox => bbox !== null)
      }
      if (bboxes.length === 0) {
        continue
      }

      const bbox = mergeBboxes(bboxes)
      if (!bbox) {
        continue
      }

      units.push({
        unit_id: `${item.item_id}:cell:${rowIndex}:${columnIndex}`,
        granularity: 'cell',
        order_index: units.length,
        text: coerceCellText(sourceRow[columnIndex]),
        bbox,
        bboxes,
        row_index: rowIndex,
        column_index: columnIndex,
        row_span: Math.trunc(asNumber(groundedCell.row_span) ?? 1),
        column_span: Math.trunc(asNumber(groundedCell.column_span) ?? 1),
        source_path: `${item.source_path}.grounding.rows[${rowIndex}][${columnIndex}]`,
        provider: 'llamaparse-item',
      })
    }
  }

  return units
}

export function buildItemInteractionData(
  item: GroundingItem,
  visibleLayers: OverlayLayerVisibility,
): ItemInteractionData {
  const cellUnits = buildCellUnits(item)
  const lineContexts = buildLineContexts(asObject(item.raw_payload) ?? {}, item.item_id)
  const lineUnits = buildLineUnits(lineContexts)
  const wordUnits = buildWordUnits(lineContexts)

  let mode: ItemInteractionMode = null
  if (visibleLayers.cell && cellUnits.length > 0) {
    mode = 'cell'
  } else if (visibleLayers.line && lineUnits.length > 0) {
    mode = 'line'
  } else if (visibleLayers.word && wordUnits.length > 0) {
    mode = 'word'
  }

  return {
    mode,
    cellUnits,
    lineUnits,
    wordUnits,
  }
}

export function unitsForMode(interaction: ItemInteractionData): GroundingGranularUnit[] {
  if (interaction.mode === 'cell') {
    return interaction.cellUnits
  }
  if (interaction.mode === 'line') {
    return interaction.lineUnits
  }
  if (interaction.mode === 'word') {
    return interaction.wordUnits
  }
  return []
}

export function matchUnitsToTextContent(textContent: string, units: GroundingGranularUnit[]): MatchedTextUnit[] {
  const matches: MatchedTextUnit[] = []
  let cursor = 0

  for (const unit of units) {
    const text = unit.text.trim()
    if (!text) {
      continue
    }

    const pattern = new RegExp(escapeForRegex(text).replace(/\s+/g, '\\s+'), 'u')
    const haystack = textContent.slice(cursor)
    const match = haystack.match(pattern)
    if (!match || match.index === undefined) {
      continue
    }

    const start = cursor + match.index
    const end = start + match[0].length
    matches.push({ unit, start, end })
    cursor = end
  }

  return matches
}

export function caretTextOffsetFromPoint(root: HTMLElement, x: number, y: number): number | null {
  const doc = root.ownerDocument
  if (!doc) {
    return null
  }

  let container: Node | null = null
  let offset = 0

  if (typeof doc.caretPositionFromPoint === 'function') {
    const position = doc.caretPositionFromPoint(x, y)
    if (position) {
      container = position.offsetNode
      offset = position.offset
    }
  } else if (typeof doc.caretRangeFromPoint === 'function') {
    const range = doc.caretRangeFromPoint(x, y)
    if (range) {
      container = range.startContainer
      offset = range.startOffset
    }
  }

  if (!container) {
    return null
  }

  const parent = container.nodeType === Node.TEXT_NODE ? container.parentNode : container
  if (parent && !root.contains(parent)) {
    return null
  }

  const walker = doc.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  let total = 0
  let current = walker.nextNode()
  while (current) {
    if (current === container) {
      return total + Math.min(offset, current.textContent?.length ?? 0)
    }
    total += current.textContent?.length ?? 0
    current = walker.nextNode()
  }

  if (container === root) {
    return Math.min(offset, root.textContent?.length ?? 0)
  }

  return null
}
