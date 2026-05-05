import { useEffect, useMemo, useRef, useState } from 'react'

import { getDocument, GlobalWorkerOptions, Util } from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

import { boxesForPage, itemCountForLayer, type OverlayBox, type OverlayLayerName, type OverlayLayerVisibility } from '../lib/grounding'
import { gtOverlayPredRects, partitionGtOverlayRegions } from '../lib/gtOverlay'
import type {
  GroundingBbox,
  GroundingGranularUnit,
  GroundingGranularity,
  GroundingLayerAvailability,
  GroundingPage,
  GroundTruthRuleMatch,
  SourceKind,
} from '../types/api'

GlobalWorkerOptions.workerSrc = pdfWorkerUrl

interface ViewerPaneProps {
  page: GroundingPage
  sourceKind: SourceKind
  sourceUrl: string | null
  assetUrl: string
  hoverSource: 'viewer' | 'sidebar' | null
  visibleLayers: OverlayLayerVisibility
  activeItemId: string | null
  hoveredItemId: string | null
  activeGranularUnitId: string | null
  hoveredGranularUnitId: string | null
  activeGranularPreview: GroundingGranularUnit | null
  hoveredGranularPreview: GroundingGranularUnit | null
  activeGtRules: GroundTruthRuleMatch[]
  hoveredGtRules: GroundTruthRuleMatch[]
  onToggleLayer: (layer: OverlayLayerName) => void
  onShowAllLayers: () => void
  onShowLayoutOnly: () => void
  onHoverItem: (itemId: string | null) => void
  onSelectItem: (itemId: string) => void
  onSelectEvidence: (itemId: string | null, ruleIds: string[]) => void
  onHoverGranularUnit: (unitId: string | null, granularity: GroundingGranularity | null) => void
  onSelectGranularUnit: (unitId: string, granularity: GroundingGranularity) => void
}

type ViewerRenderStatus = 'idle' | 'loading' | 'loaded' | 'error'

const COLORS = ['#d7263d', '#3f88c5', '#f49d37', '#140f2d', '#2e8b57', '#8f2d56', '#4f5d75']
const PREVIEW_HIGHLIGHT_COLOR = '#ffd54a'
const FIELD_MATCH_IOU_THRESHOLD = 0.95

function colorForLabel(label: string): string {
  const explicitColors: Record<string, string> = {
    'granular-line': '#3f88c5',
    'granular-word': '#f49d37',
    'granular-cell': '#2e8b57',
    'layout-text': '#d7263d',
    'layout-heading': '#c855bc',
    'layout-title': '#9f6ad8',
    'layout-table': '#5a78ff',
    'layout-list': '#7b61ff',
    'layout-header': '#cc4b6f',
    'layout-image': '#8f2d56',
    'layout-picture': '#8f2d56',
    'layout-unknown': '#4f5d75',
    'field-unmatched': '#d96b6b',
    'container-list': '#f2c14e',
    'container-list-item': '#f2c14e',
    'container-list-group': '#f2c14e',
    'container-header': '#58a4b0',
    'container-page-header': '#58a4b0',
    'container-footer': '#7d8597',
    'container-page-footer': '#7d8597',
  }
  if (label in explicitColors) {
    return explicitColors[label]
  }
  let hash = 0
  for (let i = 0; i < label.length; i += 1) {
    hash = (hash << 5) - hash + label.charCodeAt(i)
    hash |= 0
  }
  return COLORS[Math.abs(hash) % COLORS.length]
}

function layerAvailability(
  page: GroundingPage,
  layer: OverlayLayerName,
): {
  availability: GroundingLayerAvailability
  count: number
  reason: string | null
} {
  if (layer === 'layout' || layer === 'container' || layer === 'field') {
    const count = itemCountForLayer(page, layer)
    return {
      availability: count > 0 ? 'available' : 'empty',
      count,
      reason: null,
    }
  }

  const granularLayer = page.granular_layers.find((candidate) => candidate.granularity === layer)
  if (!granularLayer) {
    return {
      availability: 'unavailable',
      count: 0,
      reason: 'No normalized overlay data was returned for this layer.',
    }
  }

  return {
    availability: granularLayer.availability,
    count: granularLayer.units.length,
    reason: granularLayer.reason,
  }
}

function layerCountLabel(count: number, layer: OverlayLayerName): string {
  if (layer === 'layout') {
    return `${count} items`
  }
  if (layer === 'container') {
    return `${count} containers`
  }
  if (layer === 'field') {
    return `${count} fields`
  }
  return `${count} ${layer}${count === 1 ? '' : 's'}`
}

type BboxGeometry = Pick<GroundingBbox, 'x' | 'y' | 'w' | 'h'>

function bboxArea(bbox: BboxGeometry): number {
  return Math.max(bbox.w, 0) * Math.max(bbox.h, 0)
}

function bboxIou(left: BboxGeometry, right: BboxGeometry): number {
  const x1 = Math.max(left.x, right.x)
  const y1 = Math.max(left.y, right.y)
  const x2 = Math.min(left.x + left.w, right.x + right.w)
  const y2 = Math.min(left.y + left.h, right.y + right.h)
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const union = bboxArea(left) + bboxArea(right) - intersection
  return union > 0 ? intersection / union : 0
}

function extractRuleStatus(rule: GroundTruthRuleMatch): 'pass' | 'loc-only' | 'fail' {
  if (rule.localization_pass && rule.attribution_pass) {
    return 'pass'
  }
  if (rule.localization_pass) {
    return 'loc-only'
  }
  return 'fail'
}

function boxAsBbox(box: OverlayBox): BboxGeometry {
  return {
    x: box.x,
    y: box.y,
    w: box.w,
    h: box.h,
  }
}

function matchedGranularBboxesForRule(rule: GroundTruthRuleMatch | null, page: GroundingPage): GroundingBbox[] {
  if (!rule || rule.rule_type !== 'extract_field' || !rule.matched_unit_ids || rule.matched_unit_ids.length === 0) {
    return []
  }

  const matchedIds = new Set(rule.matched_unit_ids)
  const bboxes: GroundingBbox[] = []
  for (const item of page.items) {
    if (!matchedIds.has(item.item_id)) {
      continue
    }
    bboxes.push(...item.bboxes)
  }
  for (const layer of page.granular_layers) {
    for (const unit of layer.units) {
      if (!matchedIds.has(unit.unit_id)) {
        continue
      }
      bboxes.push(...(unit.bboxes.length > 0 ? unit.bboxes : [unit.bbox]))
    }
  }
  return bboxes
}

function renderTextLayer(
  container: HTMLDivElement,
  textContent: { items: Array<Record<string, unknown>> },
  viewport: { width: number; height: number; scale: number; transform: number[] },
) {
  container.innerHTML = ''
  container.style.width = `${viewport.width}px`
  container.style.height = `${viewport.height}px`

  for (const item of textContent.items) {
    if (typeof item.str !== 'string' || item.str.trim() === '' || !Array.isArray(item.transform)) {
      continue
    }

    const tx = Util.transform(viewport.transform, item.transform as number[])
    const fontHeight = Math.sqrt(tx[2] * tx[2] + tx[3] * tx[3])
    const angle = Math.atan2(tx[1], tx[0])

    const span = document.createElement('span')
    span.textContent = item.str
    span.style.fontSize = `${fontHeight}px`
    span.style.fontFamily = 'sans-serif'
    span.style.left = `${tx[4]}px`
    span.style.top = `${tx[5] - fontHeight}px`

    const transforms: string[] = []
    if (typeof item.width === 'number' && fontHeight > 0) {
      const measuredWidth = item.str.length * fontHeight * 0.5
      const targetWidth = item.width * viewport.scale
      if (measuredWidth > 0) {
        const scaleX = targetWidth / measuredWidth
        if (scaleX > 0.5 && scaleX < 2) {
          transforms.push(`scaleX(${scaleX})`)
        }
      }
    }
    if (Math.abs(angle) > 0.01) {
      transforms.push(`rotate(${angle}rad)`)
    }
    if (transforms.length > 0) {
      span.style.transform = transforms.join(' ')
      span.style.transformOrigin = 'left bottom'
    }

    container.appendChild(span)
  }
}

export function ViewerPane({
  page,
  sourceKind,
  sourceUrl,
  assetUrl,
  hoverSource,
  visibleLayers,
  activeItemId,
  hoveredItemId,
  activeGranularUnitId,
  hoveredGranularUnitId,
  activeGranularPreview,
  hoveredGranularPreview,
  activeGtRules,
  hoveredGtRules,
  onToggleLayer,
  onShowAllLayers,
  onShowLayoutOnly,
  onHoverItem,
  onSelectItem,
  onSelectEvidence,
  onHoverGranularUnit,
  onSelectGranularUnit,
}: ViewerPaneProps) {
  const paneRef = useRef<HTMLDivElement | null>(null)
  const imageWrapRef = useRef<HTMLDivElement | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const textLayerRef = useRef<HTMLDivElement | null>(null)
  const pdfDocumentRef = useRef<{ url: string; document: Awaited<ReturnType<typeof getDocument>['promise']> } | null>(
    null,
  )
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  const [pdfBaseSize, setPdfBaseSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  const [renderedSize, setRenderedSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  const [paneWidth, setPaneWidth] = useState(0)
  const [zoomFactor, setZoomFactor] = useState(1)
  const [renderStatus, setRenderStatus] = useState<ViewerRenderStatus>(sourceKind === 'image' ? 'loading' : 'idle')
  const [renderError, setRenderError] = useState<string | null>(null)

  const boxes = useMemo(() => boxesForPage(page, visibleLayers), [page, visibleLayers])
  const pageExtractGtRules = useMemo(
    () =>
      (page.gt_rules ?? []).filter(
        (rule) =>
          rule.rule_type === 'extract_field' &&
          rule.page_number === page.page_number &&
          rule.gt_bbox &&
          !(rule.tags ?? []).includes('stray_evidence'),
      ),
    [page],
  )
  const localizedExtractRules = useMemo(
    () => pageExtractGtRules.filter((rule) => rule.localization_pass),
    [pageExtractGtRules],
  )
  const matchedFieldBoxKeys = useMemo(() => {
    if (!visibleLayers.field || localizedExtractRules.length === 0) {
      return new Set<string>()
    }

    const matchedUnitIds = new Set<string>()
    const matchedBboxes: GroundingBbox[] = []
    for (const rule of localizedExtractRules) {
      for (const unitId of rule.matched_unit_ids ?? []) {
        matchedUnitIds.add(unitId)
      }
      matchedBboxes.push(...(rule.predicted_bboxes ?? []))
    }

    const matchedKeys = new Set<string>()
    for (const box of boxes) {
      if (box.layer !== 'field') {
        continue
      }
      if (box.itemId !== null && matchedUnitIds.has(box.itemId)) {
        matchedKeys.add(box.key)
        continue
      }
      const bbox = boxAsBbox(box)
      if (matchedBboxes.some((matchedBbox) => bboxIou(bbox, matchedBbox) >= FIELD_MATCH_IOU_THRESHOLD)) {
        matchedKeys.add(box.key)
      }
    }
    return matchedKeys
  }, [boxes, localizedExtractRules, visibleLayers.field])
  const extractGtOverlays = useMemo(
    () =>
      visibleLayers.field
        ? pageExtractGtRules.map((rule) => ({
            rule,
            bbox: rule.gt_bbox as GroundingBbox,
            status: extractRuleStatus(rule),
          }))
        : [],
    [pageExtractGtRules, visibleLayers.field],
  )
  const ruleIdsByFieldBoxKey = useMemo(() => {
    const ruleIdsByKey = new Map<string, string[]>()
    if (!visibleLayers.field || pageExtractGtRules.length === 0) {
      return ruleIdsByKey
    }

    for (const box of boxes) {
      if (box.layer !== 'field') {
        continue
      }

      const bbox = boxAsBbox(box)
      const ruleIds = pageExtractGtRules
        .filter((rule) => {
          if (box.itemId !== null && (rule.matched_unit_ids ?? []).includes(box.itemId)) {
            return true
          }
          return (rule.predicted_bboxes ?? []).some(
            (predictedBbox) => bboxIou(bbox, predictedBbox) >= FIELD_MATCH_IOU_THRESHOLD,
          )
        })
        .map((rule) => rule.rule_id)

      if (ruleIds.length > 0) {
        ruleIdsByKey.set(box.key, ruleIds)
      }
    }

    return ruleIdsByKey
  }, [boxes, pageExtractGtRules, visibleLayers.field])
  const focusedItemId = hoveredItemId ?? activeItemId
  const focusedGranularUnitId = hoveredGranularUnitId ?? activeGranularUnitId
  const focusedGranularPreview = hoveredGranularPreview ?? activeGranularPreview
  const focusedGtRules = hoveredGtRules.length > 0 ? hoveredGtRules : activeGtRules
  const focusedGtPartitions = useMemo(
    () =>
      focusedGtRules.map((rule) => {
        const predBboxes = matchedGranularBboxesForRule(rule, page)
        return {
          rule,
          partition: partitionGtOverlayRegions(rule, predBboxes),
        }
      }),
    [focusedGtRules, page],
  )
  const hasFocusedSelection = Boolean(focusedItemId || focusedGranularUnitId || focusedGtRules.length > 0)
  const intrinsicSize = sourceKind === 'pdf' ? pdfBaseSize : imageSize

  const layerControls = useMemo(
    () =>
      (['layout', 'container', 'line', 'word', 'cell', 'field'] as const).map((layer) => ({
        layer,
        ...layerAvailability(page, layer),
      })),
    [page],
  )

  useEffect(() => {
    const pane = paneRef.current
    if (!pane) {
      return
    }

    const updatePaneWidth = () => {
      setPaneWidth(Math.max(0, pane.clientWidth - 20))
    }

    updatePaneWidth()

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', updatePaneWidth)
      return () => {
        window.removeEventListener('resize', updatePaneWidth)
      }
    }

    const observer = new ResizeObserver(updatePaneWidth)
    observer.observe(pane)
    return () => {
      observer.disconnect()
    }
  }, [])

  useEffect(() => {
    return () => {
      const current = pdfDocumentRef.current
      if (current) {
        void current.document.destroy()
      }
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let renderTask: { cancel: () => void; promise: Promise<void> } | null = null

    async function ensurePdfDocument() {
      if (!sourceUrl) {
        throw new Error('Missing PDF source URL.')
      }
      const cached = pdfDocumentRef.current
      if (cached && cached.url === sourceUrl) {
        return cached.document
      }
      if (cached) {
        await cached.document.destroy()
        pdfDocumentRef.current = null
      }

      const loadingTask = getDocument(sourceUrl)
      const document = await loadingTask.promise
      pdfDocumentRef.current = { url: sourceUrl, document }
      return document
    }

    async function renderPdfPage() {
      if (sourceKind !== 'pdf') {
        return
      }
      setRenderStatus('loading')
      setRenderError(null)

      const document = await ensurePdfDocument()
      if (cancelled) {
        return
      }

      const pdfPage = await document.getPage(page.page_number)
      const baseViewport = pdfPage.getViewport({ scale: 1 })
      if (cancelled) {
        return
      }

      setPdfBaseSize({ width: baseViewport.width, height: baseViewport.height })

      const fitScale = baseViewport.width > 0 && paneWidth > 0 ? Math.min(1, paneWidth / baseViewport.width) : 1
      const viewport = pdfPage.getViewport({ scale: fitScale * zoomFactor })
      const canvas = canvasRef.current
      const textLayer = textLayerRef.current
      if (!canvas || !textLayer) {
        return
      }

      const context = canvas.getContext('2d')
      if (!context) {
        throw new Error('Failed to get PDF canvas context.')
      }

      const pixelRatio = window.devicePixelRatio || 1
      canvas.width = Math.ceil(viewport.width * pixelRatio)
      canvas.height = Math.ceil(viewport.height * pixelRatio)
      canvas.style.width = `${viewport.width}px`
      canvas.style.height = `${viewport.height}px`

      const nextRenderTask = pdfPage.render({
        canvas,
        canvasContext: context,
        viewport,
        transform: pixelRatio === 1 ? undefined : [pixelRatio, 0, 0, pixelRatio, 0, 0],
      })
      renderTask = nextRenderTask

      await nextRenderTask.promise
      const textContent = await pdfPage.getTextContent()
      if (cancelled) {
        return
      }

      renderTextLayer(textLayer, textContent as { items: Array<Record<string, unknown>> }, {
        width: viewport.width,
        height: viewport.height,
        scale: viewport.scale,
        transform: viewport.transform,
      })
      setRenderedSize({ width: viewport.width, height: viewport.height })
      setRenderStatus('loaded')
    }

    if (sourceKind === 'pdf') {
      void renderPdfPage().catch((error) => {
        if (cancelled) {
          return
        }
        setRenderError(error instanceof Error ? error.message : String(error))
        setRenderStatus('error')
      })
    }

    return () => {
      cancelled = true
      renderTask?.cancel()
    }
  }, [page.page_number, paneWidth, sourceKind, sourceUrl, zoomFactor])

  const fitScale = useMemo(() => {
    if (intrinsicSize.width <= 0 || paneWidth <= 0) {
      return 1
    }
    return Math.min(1, paneWidth / intrinsicSize.width)
  }, [intrinsicSize.width, paneWidth])

  const zoomScale = fitScale * zoomFactor
  const renderedWidth =
    sourceKind === 'pdf'
      ? renderedSize.width || (intrinsicSize.width > 0 ? Math.max(1, Math.round(intrinsicSize.width * zoomScale)) : undefined)
      : intrinsicSize.width > 0
        ? Math.max(1, Math.round(intrinsicSize.width * zoomScale))
        : undefined
  const renderedHeight =
    sourceKind === 'pdf'
      ? renderedSize.height ||
        (intrinsicSize.height > 0 ? Math.max(1, Math.round(intrinsicSize.height * zoomScale)) : undefined)
      : intrinsicSize.height > 0
        ? Math.max(1, Math.round(intrinsicSize.height * zoomScale))
        : undefined

  const baseWidth = page.page_width > 0 ? page.page_width : intrinsicSize.width || 1
  const baseHeight = page.page_height > 0 ? page.page_height : intrinsicSize.height || 1
  const overlayWidth = renderedWidth ?? 0
  const overlayHeight = renderedHeight ?? 0

  const canZoomOut = zoomFactor > 0.25
  const canZoomIn = zoomFactor < 6
  const zoomPercentage = Math.round(zoomFactor * 100)

  useEffect(() => {
    if (hoverSource !== 'sidebar' || !hoveredGranularPreview) {
      return
    }

    const pane = paneRef.current
    const imageWrap = imageWrapRef.current
    if (!pane || !imageWrap || overlayWidth <= 0 || overlayHeight <= 0 || baseWidth <= 0 || baseHeight <= 0) {
      return
    }

    const previewBoxes =
      hoveredGranularPreview.bboxes.length > 0 ? hoveredGranularPreview.bboxes : [hoveredGranularPreview.bbox]
    if (previewBoxes.length === 0) {
      return
    }

    const left = Math.min(...previewBoxes.map((bbox) => bbox.x))
    const top = Math.min(...previewBoxes.map((bbox) => bbox.y))
    const right = Math.max(...previewBoxes.map((bbox) => bbox.x + bbox.w))
    const bottom = Math.max(...previewBoxes.map((bbox) => bbox.y + bbox.h))

    const centerX = ((left + right) / 2 / baseWidth) * overlayWidth
    const centerY = ((top + bottom) / 2 / baseHeight) * overlayHeight

    const nextScrollLeft = Math.max(0, imageWrap.offsetLeft + centerX - pane.clientWidth / 2)
    const nextScrollTop = Math.max(0, imageWrap.offsetTop + centerY - pane.clientHeight / 2)

    pane.scrollTo({
      left: nextScrollLeft,
      top: nextScrollTop,
      behavior: 'smooth',
    })
  }, [baseHeight, baseWidth, hoverSource, hoveredGranularPreview, overlayHeight, overlayWidth])

  useEffect(() => {
    if (hoverSource !== 'sidebar' || hoveredGtRules.length === 0) {
      return
    }

    const pane = paneRef.current
    const imageWrap = imageWrapRef.current
    if (!pane || !imageWrap || overlayWidth <= 0 || overlayHeight <= 0 || baseWidth <= 0 || baseHeight <= 0) {
      return
    }

    const previewBoxes = hoveredGtRules.flatMap((rule) => [
      rule.gt_bbox,
      ...gtOverlayPredRects(rule, matchedGranularBboxesForRule(rule, page)),
    ])

    const left = Math.min(...previewBoxes.map((bbox) => bbox.x))
    const top = Math.min(...previewBoxes.map((bbox) => bbox.y))
    const right = Math.max(...previewBoxes.map((bbox) => bbox.x + bbox.w))
    const bottom = Math.max(...previewBoxes.map((bbox) => bbox.y + bbox.h))

    const centerX = ((left + right) / 2 / baseWidth) * overlayWidth
    const centerY = ((top + bottom) / 2 / baseHeight) * overlayHeight

    const nextScrollLeft = Math.max(0, imageWrap.offsetLeft + centerX - pane.clientWidth / 2)
    const nextScrollTop = Math.max(0, imageWrap.offsetTop + centerY - pane.clientHeight / 2)

    pane.scrollTo({
      left: nextScrollLeft,
      top: nextScrollTop,
      behavior: 'smooth',
    })
  }, [baseHeight, baseWidth, hoverSource, hoveredGtRules, overlayHeight, overlayWidth, page])

  return (
    <div className="viewer-pane" ref={paneRef}>
      <div className="viewer-layer-toolbar" aria-label="Overlay layers">
        <div className="viewer-toolbar-group viewer-page-controls">
          <span>Page {page.page_number}</span>
          <div className="viewer-zoom-controls" aria-label="Zoom controls">
            <button
              onClick={() => setZoomFactor((value) => Math.max(0.25, Math.round(value * 0.9 * 1000) / 1000))}
              disabled={!canZoomOut}
              title="Zoom out"
            >
              -
            </button>
            <span>{zoomPercentage}%</span>
            <button
              onClick={() => setZoomFactor((value) => Math.min(6, Math.round(value * 1.1 * 1000) / 1000))}
              disabled={!canZoomIn}
              title="Zoom in"
            >
              +
            </button>
            <button onClick={() => setZoomFactor(1)} title="Fit to page">
              Fit
            </button>
          </div>
          <span>
            {Math.round(page.page_width)} × {Math.round(page.page_height)}
          </span>
        </div>

        <div className="viewer-toolbar-group viewer-layer-actions">
          <button
            className="viewer-layer-action"
            onClick={onShowAllLayers}
          >
            All layers
          </button>
          <button className="viewer-layer-action" onClick={onShowLayoutOnly}>
            Layout only
          </button>
        </div>

        {layerControls.map(({ layer, availability, count, reason }) => {
          const active = visibleLayers[layer] && availability !== 'unavailable'
          const disabled = availability === 'unavailable'
          const className = [
            'viewer-layer-chip',
            `layer-${layer}`,
            active ? 'active' : '',
            disabled ? 'disabled' : '',
          ]
            .filter(Boolean)
            .join(' ')
          const title =
            availability === 'unavailable'
              ? reason ?? `${layer} overlays are unavailable for this page.`
              : availability === 'empty'
                ? `No ${layer} overlays are present on this page.`
                : layerCountLabel(count, layer)

          return (
            <button
              key={layer}
              className={className}
              onClick={() => onToggleLayer(layer)}
              disabled={disabled}
              title={title}
            >
              <span className="viewer-layer-chip-label">{layer}</span>
              <span className="viewer-layer-chip-count">
                {availability === 'unavailable' ? 'n/a' : count}
              </span>
            </button>
          )
        })}
      </div>

      {renderStatus === 'error' ? (
        <div className="viewer-error-card">
          <strong>Unable to render this page.</strong>
          <span>{renderError ?? 'Unknown render failure.'}</span>
        </div>
      ) : null}

      <div className="viewer-image-wrap" ref={imageWrapRef}>
        {sourceKind === 'pdf' ? (
          <div
            className="viewer-pdf-stack"
            style={{
              width: renderedWidth ? `${renderedWidth}px` : undefined,
              height: renderedHeight ? `${renderedHeight}px` : undefined,
            }}
          >
            <canvas ref={canvasRef} className="viewer-pdf-canvas" />
            <div ref={textLayerRef} className="viewer-text-layer" />
          </div>
        ) : (
          <img
            ref={imageRef}
            src={assetUrl}
            alt={`page-${page.page_number}`}
            className="viewer-image"
            style={{
              width: renderedWidth ? `${renderedWidth}px` : undefined,
              height: renderedHeight ? `${renderedHeight}px` : undefined,
            }}
            onLoad={(event) => {
              const image = event.currentTarget
              setImageSize({
                width: image.naturalWidth || image.clientWidth,
                height: image.naturalHeight || image.clientHeight,
              })
              setRenderedSize({
                width: image.clientWidth || image.naturalWidth,
                height: image.clientHeight || image.naturalHeight,
              })
              setRenderStatus('loaded')
            }}
            onError={() => {
              setRenderError('Failed to load the page asset.')
              setRenderStatus('error')
            }}
          />
        )}

        <svg
          className="viewer-overlay"
          width={overlayWidth}
          height={overlayHeight}
          viewBox={`0 0 ${baseWidth} ${baseHeight}`}
          preserveAspectRatio="none"
        >
          {focusedGranularPreview ? (
            <g className="overlay-preview-group">
              {(focusedGranularPreview.bboxes.length > 0 ? focusedGranularPreview.bboxes : [focusedGranularPreview.bbox]).map(
                (bbox, index) => {
                  return (
                    <rect
                      key={`${focusedGranularPreview.unit_id}:preview:${index}`}
                      x={bbox.x}
                      y={bbox.y}
                      width={Math.max(bbox.w, 1)}
                      height={Math.max(bbox.h, 1)}
                      className="overlay-box preview"
                      stroke={PREVIEW_HIGHLIGHT_COLOR}
                      fill={PREVIEW_HIGHLIGHT_COLOR}
                    />
                  )
                },
              )}
            </g>
          ) : null}
          {extractGtOverlays.length > 0 ? (
            <g className="overlay-field-gt-group">
              {extractGtOverlays.map(({ rule, bbox, status }) => (
                <rect
                  key={`${rule.rule_id}:field-gt`}
                  x={bbox.x}
                  y={bbox.y}
                  width={Math.max(bbox.w, 1)}
                  height={Math.max(bbox.h, 1)}
                  className={`overlay-field-gt ${status}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    onSelectEvidence(null, [rule.rule_id])
                  }}
                />
              ))}
            </g>
          ) : null}
          {boxes.map((box) => {
            if (box.layer === 'field' && matchedFieldBoxKeys.has(box.key)) {
              return null
            }
            if (focusedGtRules.length > 0 && box.isExtractEvidence) {
              return null
            }
            const x = box.x
            const y = box.y
            const width = box.w
            const height = box.h
            const highlighted =
              (box.itemId !== null && focusedItemId === box.itemId) ||
              (box.unitId !== null && focusedGranularUnitId === box.unitId)
            const muted = hasFocusedSelection && !highlighted
            const fill = colorForLabel(box.colorKey)
            const className = [
              'overlay-box',
              `layer-${box.layer}`,
              highlighted ? 'active' : '',
              muted ? 'muted' : '',
            ]
              .filter(Boolean)
              .join(' ')
            const labelX = x + 4
            const labelY = Math.max(y - 6, 12)
            const roRadius = 10
            const roCx = Math.min(Math.max(x + width - roRadius / 2, roRadius), baseWidth - roRadius)
            const roCy = Math.min(Math.max(y - roRadius / 2, roRadius), baseHeight - roRadius)
            const roClassName = [
              'overlay-reading-order',
              highlighted ? 'active' : '',
              muted ? 'muted' : '',
            ]
              .filter(Boolean)
              .join(' ')
            const overlayLabel =
              box.metadataLabel && box.text
                ? `${box.label} · ${box.metadataLabel} · ${box.text}`
                : box.metadataLabel
                  ? `${box.label} · ${box.metadataLabel}`
                  : box.text
                    ? `${box.label} · ${box.text}`
                    : box.label
            const granularLayer =
              box.layer === 'line' || box.layer === 'word' || box.layer === 'cell' ? box.layer : null
            const showOverlayLabel = false

            return (
              <g key={box.key}>
                {box.layer === 'word' ? (
                  <>
                    {highlighted ? (
                      <rect
                        x={x}
                        y={y}
                        width={Math.max(width, 1)}
                        height={Math.max(height, 1)}
                        className="overlay-word-highlight"
                        fill={PREVIEW_HIGHLIGHT_COLOR}
                      />
                    ) : null}
                    <rect
                      x={x}
                      y={y}
                      width={Math.max(width, 1)}
                      height={Math.max(height, 1)}
                      className="overlay-word-hitbox"
                      onMouseEnter={() => {
                        if (box.itemId !== null) {
                          onHoverItem(box.itemId)
                          return
                        }
                        if (box.unitId !== null && granularLayer !== null) {
                          onHoverGranularUnit(box.unitId, granularLayer)
                        }
                      }}
                      onMouseLeave={() => {
                        if (box.itemId !== null) {
                          onHoverItem(null)
                          return
                        }
                        if (box.unitId !== null && granularLayer !== null) {
                          onHoverGranularUnit(null, null)
                        }
                      }}
                      onClick={() => {
                        if (box.itemId !== null) {
                          onSelectItem(box.itemId)
                          return
                        }
                        if (box.unitId !== null && granularLayer !== null) {
                          onSelectGranularUnit(box.unitId, granularLayer)
                        }
                      }}
                    />
                    <line
                      x1={x}
                      y1={y}
                      x2={x}
                      y2={y + Math.max(height, 1)}
                      className={[
                        'overlay-word-boundary',
                        highlighted ? 'active' : '',
                        muted ? 'muted' : '',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                      stroke={highlighted ? PREVIEW_HIGHLIGHT_COLOR : fill}
                    />
                    {highlighted ? (
                      <line
                        x1={x + Math.max(width, 1)}
                        y1={y}
                        x2={x + Math.max(width, 1)}
                        y2={y + Math.max(height, 1)}
                        className={[
                          'overlay-word-boundary',
                          'active',
                          muted ? 'muted' : '',
                        ]
                          .filter(Boolean)
                          .join(' ')}
                        stroke={PREVIEW_HIGHLIGHT_COLOR}
                      />
                    ) : null}
                  </>
                ) : (
                  <rect
                    x={x}
                    y={y}
                    width={Math.max(width, 1)}
                    height={Math.max(height, 1)}
                    className={className}
                    stroke={fill}
                    fill={fill}
                    onMouseEnter={() => {
                      if (box.itemId !== null) {
                        onHoverItem(box.itemId)
                        return
                      }
                      if (box.unitId !== null && granularLayer !== null) {
                        onHoverGranularUnit(box.unitId, granularLayer)
                      }
                    }}
                    onMouseLeave={() => {
                      if (box.itemId !== null) {
                        onHoverItem(null)
                        return
                      }
                      if (box.unitId !== null && granularLayer !== null) {
                        onHoverGranularUnit(null, null)
                      }
                    }}
                    onClick={() => {
                      if (box.layer === 'field') {
                        onSelectEvidence(box.itemId, ruleIdsByFieldBoxKey.get(box.key) ?? [])
                        return
                      }
                      if (box.itemId !== null) {
                        onSelectItem(box.itemId)
                        return
                      }
                      if (box.unitId !== null && granularLayer !== null) {
                        onSelectGranularUnit(box.unitId, granularLayer)
                      }
                    }}
                  />
                )}
                {showOverlayLabel ? (
                  <text x={labelX} y={labelY} className="overlay-label">
                    {overlayLabel}
                  </text>
                ) : null}
                {box.showReadingOrder && box.readingOrder !== null ? (
                  <>
                    <circle cx={roCx} cy={roCy} r={roRadius} className={roClassName} />
                    <text x={roCx} y={roCy + 4} className={roClassName + ' text'}>
                      {box.readingOrder}
                    </text>
                  </>
                ) : null}
              </g>
            )
          })}
          {focusedGtPartitions.map(({ rule, partition }) => {
            const focusedStray = (rule.tags ?? []).includes('stray_evidence')
            const groupClassName = ['overlay-gt-group', focusedStray ? 'stray' : ''].filter(Boolean).join(' ')
            const gtOnlyClassName = ['overlay-gt-gt-only', focusedStray ? 'stray' : ''].filter(Boolean).join(' ')
            const predOnlyClassName = ['overlay-gt-pred-only', focusedStray ? 'stray' : ''].filter(Boolean).join(' ')
            const overlapClassName = ['overlay-gt-overlap', focusedStray ? 'stray' : ''].filter(Boolean).join(' ')
            return (
              <g key={rule.rule_id} className={groupClassName}>
                {partition.gtOnly.map((bbox, index) => (
                  <rect
                    key={`${rule.rule_id}:gt-only:${index}`}
                    x={bbox.x}
                    y={bbox.y}
                    width={Math.max(bbox.w, 1)}
                    height={Math.max(bbox.h, 1)}
                    className={gtOnlyClassName}
                  />
                ))}
                {partition.predOnly.map((bbox, index) => (
                  <rect
                    key={`${rule.rule_id}:pred-only:${index}`}
                    x={bbox.x}
                    y={bbox.y}
                    width={Math.max(bbox.w, 1)}
                    height={Math.max(bbox.h, 1)}
                    className={predOnlyClassName}
                  />
                ))}
                {partition.overlap.map((bbox, index) => (
                  <rect
                    key={`${rule.rule_id}:overlap:${index}`}
                    x={bbox.x}
                    y={bbox.y}
                    width={Math.max(bbox.w, 1)}
                    height={Math.max(bbox.h, 1)}
                    className={overlapClassName}
                  />
                ))}
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
