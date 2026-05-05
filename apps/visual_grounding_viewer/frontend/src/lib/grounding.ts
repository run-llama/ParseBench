import type {
  GroundingGranularLayer,
  GroundingGranularUnit,
  GroundingItem,
  GroundingPage,
} from '../types/api'

export type OverlayLayerName = 'layout' | 'container' | 'line' | 'word' | 'cell' | 'field'
export type OverlayItemLayerName = 'layout' | 'container' | 'field'

export interface OverlayLayerVisibility {
  layout: boolean
  container: boolean
  line: boolean
  word: boolean
  cell: boolean
  field: boolean
}

export interface OverlayBox {
  key: string
  itemId: string | null
  unitId: string | null
  layer: OverlayLayerName
  granularity: OverlayLayerName
  label: string
  colorKey: string
  readingOrder: number | null
  showReadingOrder: boolean
  isExtractEvidence: boolean
  x: number
  y: number
  w: number
  h: number
  text: string
  metadataLabel: string | null
}

function normalizeOverlayClass(value: string | null | undefined): string | null {
  if (!value) {
    return null
  }

  const normalized = value.trim().toLowerCase().replace(/[\s_]+/g, '-')
  return normalized || null
}

function overlayColorKey(itemType: string, bboxLabel: string | null): string {
  const normalizedType = normalizeOverlayClass(itemType)
  const normalizedLabel = normalizeOverlayClass(bboxLabel)

  if (normalizedType === 'text' && normalizedLabel && normalizedLabel !== normalizedType) {
    return normalizedLabel
  }

  if (normalizedType && normalizedType !== 'unknown') {
    return normalizedType
  }

  if (normalizedLabel) {
    return normalizedLabel
  }

  return 'unknown'
}

function layoutColorKey(itemType: string, bboxLabel: string | null): string {
  return `layout-${overlayColorKey(itemType, bboxLabel)}`
}

function containerColorKey(itemType: string, bboxLabel: string | null): string {
  return `container-${overlayColorKey(itemType, bboxLabel)}`
}

function fieldColorKey(): string {
  return 'field-unmatched'
}

function granularColorKey(granularity: OverlayLayerName): string {
  return `granular-${granularity}`
}

function trimText(value: string | null | undefined): string {
  return typeof value === 'string' ? value.trim() : ''
}

export function formatGranularUnitLabel(unit: GroundingGranularUnit): string {
  if (unit.granularity === 'cell') {
    const row = unit.row_index === null ? '?' : unit.row_index + 1
    const column = unit.column_index === null ? '?' : unit.column_index + 1
    return `cell r${row} c${column}`
  }
  return unit.granularity
}

export function formatGranularUnitMetadata(unit: GroundingGranularUnit): string | null {
  if (unit.granularity !== 'cell') {
    return null
  }

  const parts: string[] = []
  if (unit.row_index !== null) {
    parts.push(`row ${unit.row_index + 1}`)
  }
  if (unit.column_index !== null) {
    parts.push(`col ${unit.column_index + 1}`)
  }
  if (unit.row_span !== null && unit.row_span > 1) {
    parts.push(`rowspan ${unit.row_span}`)
  }
  if (unit.column_span !== null && unit.column_span > 1) {
    parts.push(`colspan ${unit.column_span}`)
  }

  return parts.length > 0 ? parts.join(' · ') : null
}

function normalizeItemType(item: GroundingItem): string | null {
  return normalizeOverlayClass(item.type)
}

function normalizeItemLabel(item: GroundingItem): string | null {
  return normalizeOverlayClass(item.bboxes[0]?.label)
}

export function isContainerItem(item: GroundingItem): boolean {
  const normalizedType = normalizeItemType(item)
  const normalizedLabel = normalizeItemLabel(item)
  const containerClasses = new Set([
    'list',
    'list-item',
    'list-group',
    'header',
    'footer',
    'page-header',
    'page-footer',
  ])

  return (
    (normalizedType !== null && containerClasses.has(normalizedType)) ||
    (normalizedLabel !== null && containerClasses.has(normalizedLabel))
  )
}

export function isTableItem(item: GroundingItem): boolean {
  const normalizedType = normalizeItemType(item)
  const normalizedLabel = normalizeItemLabel(item)
  return normalizedType === 'table' || normalizedLabel === 'table'
}

export function isExtractEvidenceItem(item: GroundingItem): boolean {
  return normalizeItemType(item) === 'extract-field' || item.source_path.startsWith('field_citations.')
}

export function itemCountForLayer(page: GroundingPage, layer: OverlayItemLayerName): number {
  if (layer === 'layout') {
    return page.items.filter((item) => !isContainerItem(item) && !isExtractEvidenceItem(item)).length
  }
  if (layer === 'container') {
    return page.items.filter((item) => isContainerItem(item)).length
  }
  return page.items.filter((item) => isExtractEvidenceItem(item)).length
}

function layoutBoxesForPage(page: GroundingPage, layer: OverlayItemLayerName): OverlayBox[] {
  const boxes: OverlayBox[] = []

  for (const item of page.items) {
    const container = isContainerItem(item)
    const isExtractEvidence = isExtractEvidenceItem(item)
    const includeItem =
      (layer === 'layout' && !container && !isExtractEvidence) ||
      (layer === 'container' && container) ||
      (layer === 'field' && isExtractEvidence)
    if (!includeItem) {
      continue
    }
    for (let idx = 0; idx < item.bboxes.length; idx += 1) {
      const bbox = item.bboxes[idx]
      boxes.push({
        key: `${item.item_id}:${idx}`,
        itemId: item.item_id,
        unitId: null,
        layer,
        granularity: layer,
        label: layer === 'field' ? 'field' : (bbox.label ?? item.type),
        colorKey:
          layer === 'layout'
            ? layoutColorKey(item.type, bbox.label)
            : layer === 'container'
              ? containerColorKey(item.type, bbox.label)
              : fieldColorKey(),
        readingOrder: item.item_index,
        showReadingOrder: layer === 'layout',
        isExtractEvidence,
        x: bbox.x,
        y: bbox.y,
        w: bbox.w,
        h: bbox.h,
        text: trimText(item.md || item.value || ''),
        metadataLabel: null,
      })
    }
  }

  return boxes
}

function granularBoxesForLayer(layer: GroundingGranularLayer): OverlayBox[] {
  return layer.units.flatMap((unit) => {
    const bboxes = unit.bboxes.length > 0 ? unit.bboxes : [unit.bbox]
    return bboxes.map((bbox, regionIndex) => ({
      key: `${layer.granularity}:${unit.unit_id}:${regionIndex}`,
      itemId: null,
      unitId: unit.unit_id,
      layer: layer.granularity,
      granularity: layer.granularity,
      label: formatGranularUnitLabel(unit),
      colorKey: granularColorKey(layer.granularity),
      readingOrder: unit.order_index,
      showReadingOrder: false,
      isExtractEvidence: false,
      x: bbox.x,
      y: bbox.y,
      w: bbox.w,
      h: bbox.h,
      text: trimText(unit.text),
      metadataLabel: formatGranularUnitMetadata(unit),
    }))
  })
}

export function boxesForPage(page: GroundingPage, visibleLayers?: OverlayLayerVisibility): OverlayBox[] {
  const resolvedVisibleLayers: OverlayLayerVisibility = visibleLayers ?? {
    layout: true,
    container: false,
    line: true,
    word: true,
    cell: true,
    field: true,
  }

  const boxes: OverlayBox[] = []

  if (resolvedVisibleLayers.layout) {
    boxes.push(...layoutBoxesForPage(page, 'layout'))
  }

  if (resolvedVisibleLayers.container) {
    boxes.push(...layoutBoxesForPage(page, 'container'))
  }

  if (resolvedVisibleLayers.field) {
    boxes.push(...layoutBoxesForPage(page, 'field'))
  }

  for (const layer of page.granular_layers) {
    if (!resolvedVisibleLayers[layer.granularity] || layer.availability !== 'available') {
      continue
    }
    boxes.push(...granularBoxesForLayer(layer))
  }

  return boxes.sort((left, right) => {
    const layerRank = {
      layout: 0,
      container: 1,
      cell: 2,
      field: 3,
      line: 4,
      word: 5,
    } satisfies Record<OverlayLayerName, number>

    if (layerRank[left.layer] !== layerRank[right.layer]) {
      return layerRank[left.layer] - layerRank[right.layer]
    }

    const leftOrder = left.readingOrder ?? Number.MAX_SAFE_INTEGER
    const rightOrder = right.readingOrder ?? Number.MAX_SAFE_INTEGER
    if (leftOrder !== rightOrder) {
      return leftOrder - rightOrder
    }

    return left.key.localeCompare(right.key)
  })
}

export function findItemById(items: GroundingItem[], itemId: string | null): GroundingItem | null {
  if (!itemId) {
    return null
  }

  return items.find((item) => item.item_id === itemId) ?? null
}

export function findGranularLayer(page: GroundingPage, granularity: GroundingGranularUnit['granularity']): GroundingGranularLayer | null {
  return page.granular_layers.find((layer) => layer.granularity === granularity) ?? null
}

export function findGranularUnitById(page: GroundingPage, unitId: string | null): GroundingGranularUnit | null {
  if (!unitId) {
    return null
  }

  for (const layer of page.granular_layers) {
    const match = layer.units.find((unit) => unit.unit_id === unitId)
    if (match) {
      return match
    }
  }

  return null
}
