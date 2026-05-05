import type { GroundingBbox, GroundTruthRuleMatch } from '../types/api'

export interface GtOverlayMetrics {
  precision: number | null
  recall: number | null
  f1: number | null
  iou: number | null
  gtArea: number
  predArea: number
  overlapArea: number
}

export interface GtOverlayPartition {
  overlap: GroundingBbox[]
  gtOnly: GroundingBbox[]
  predOnly: GroundingBbox[]
}

interface RectEdges {
  left: number
  top: number
  right: number
  bottom: number
}

function toRectEdges(bbox: GroundingBbox): RectEdges | null {
  const width = Math.max(0, bbox.w)
  const height = Math.max(0, bbox.h)
  if (width <= 0 || height <= 0) {
    return null
  }
  return {
    left: bbox.x,
    top: bbox.y,
    right: bbox.x + width,
    bottom: bbox.y + height,
  }
}

function fromRectEdges(rect: RectEdges, label: string): GroundingBbox {
  return {
    x: rect.left,
    y: rect.top,
    w: rect.right - rect.left,
    h: rect.bottom - rect.top,
    label,
    confidence: null,
    start_index: null,
    end_index: null,
  }
}

function rectArea(rect: RectEdges): number {
  return Math.max(0, rect.right - rect.left) * Math.max(0, rect.bottom - rect.top)
}

function inRect(rect: RectEdges, x: number, y: number): boolean {
  return rect.left <= x && x <= rect.right && rect.top <= y && y <= rect.bottom
}

function unionArea(rectangles: RectEdges[]): number {
  if (rectangles.length === 0) {
    return 0
  }

  const xs = [...new Set(rectangles.flatMap((rect) => [rect.left, rect.right]))].sort((a, b) => a - b)
  const ys = [...new Set(rectangles.flatMap((rect) => [rect.top, rect.bottom]))].sort((a, b) => a - b)
  let total = 0

  for (let xIndex = 0; xIndex < xs.length - 1; xIndex += 1) {
    const left = xs[xIndex]
    const right = xs[xIndex + 1]
    if (right <= left) {
      continue
    }
    for (let yIndex = 0; yIndex < ys.length - 1; yIndex += 1) {
      const top = ys[yIndex]
      const bottom = ys[yIndex + 1]
      if (bottom <= top) {
        continue
      }
      if (rectangles.some((rect) => rect.left <= left && rect.right >= right && rect.top <= top && rect.bottom >= bottom)) {
        total += (right - left) * (bottom - top)
      }
    }
  }

  return total
}

function classifiedRects(gtRect: RectEdges, predRects: RectEdges[]): GtOverlayPartition {
  const xs = [...new Set([gtRect.left, gtRect.right, ...predRects.flatMap((rect) => [rect.left, rect.right])])].sort(
    (a, b) => a - b,
  )
  const ys = [...new Set([gtRect.top, gtRect.bottom, ...predRects.flatMap((rect) => [rect.top, rect.bottom])])].sort(
    (a, b) => a - b,
  )

  const overlap: GroundingBbox[] = []
  const gtOnly: GroundingBbox[] = []
  const predOnly: GroundingBbox[] = []

  for (let xIndex = 0; xIndex < xs.length - 1; xIndex += 1) {
    const left = xs[xIndex]
    const right = xs[xIndex + 1]
    if (right <= left) {
      continue
    }
    for (let yIndex = 0; yIndex < ys.length - 1; yIndex += 1) {
      const top = ys[yIndex]
      const bottom = ys[yIndex + 1]
      if (bottom <= top) {
        continue
      }
      const sampleX = (left + right) / 2
      const sampleY = (top + bottom) / 2
      const inGt = inRect(gtRect, sampleX, sampleY)
      const inPred = predRects.some((rect) => inRect(rect, sampleX, sampleY))
      if (!inGt && !inPred) {
        continue
      }
      const bbox = fromRectEdges({ left, top, right, bottom }, 'gt-overlay')
      if (inGt && inPred) {
        overlap.push(bbox)
      } else if (inGt) {
        gtOnly.push(bbox)
      } else {
        predOnly.push(bbox)
      }
    }
  }

  return { overlap, gtOnly, predOnly }
}

function rulePredRects(rule: GroundTruthRuleMatch, predBboxesOverride: GroundingBbox[] = []): GroundingBbox[] {
  if (predBboxesOverride.length > 0) {
    return predBboxesOverride
  }
  if (rule.predicted_bboxes.length > 0) {
    return rule.predicted_bboxes
  }
  return rule.predicted_bbox ? [rule.predicted_bbox] : []
}

export function partitionGtOverlayRegions(
  rule: GroundTruthRuleMatch,
  predBboxesOverride: GroundingBbox[] = [],
): GtOverlayPartition {
  const gtRect = toRectEdges(rule.gt_bbox)
  if (!gtRect) {
    return { overlap: [], gtOnly: [], predOnly: [] }
  }
  const predRects = rulePredRects(rule, predBboxesOverride)
    .map(toRectEdges)
    .filter((rect): rect is RectEdges => rect !== null)

  if (predRects.length === 0) {
    return {
      overlap: [],
      gtOnly: [rule.gt_bbox],
      predOnly: [],
    }
  }

  return classifiedRects(gtRect, predRects)
}

export function computeGtOverlayMetrics(rule: GroundTruthRuleMatch): GtOverlayMetrics {
  const partition = partitionGtOverlayRegions(rule)
  const overlapRects = partition.overlap.map(toRectEdges).filter((rect): rect is RectEdges => rect !== null)
  const gtOnlyRects = partition.gtOnly.map(toRectEdges).filter((rect): rect is RectEdges => rect !== null)
  const predOnlyRects = partition.predOnly.map(toRectEdges).filter((rect): rect is RectEdges => rect !== null)

  const overlapArea = unionArea(overlapRects)
  const gtArea = overlapArea + unionArea(gtOnlyRects)
  const predArea = overlapArea + unionArea(predOnlyRects)
  const union = gtArea + predArea - overlapArea

  const precision = predArea > 0 ? overlapArea / predArea : null
  const recall = gtArea > 0 ? overlapArea / gtArea : null
  const f1 =
    precision !== null && recall !== null && precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : null
  const iou = union > 0 ? overlapArea / union : null

  return {
    precision,
    recall,
    f1,
    iou,
    gtArea,
    predArea,
    overlapArea,
  }
}

export function gtOverlayPredRects(
  rule: GroundTruthRuleMatch,
  predBboxesOverride: GroundingBbox[] = [],
): GroundingBbox[] {
  return rulePredRects(rule, predBboxesOverride).filter((bbox) => {
    const rect = toRectEdges(bbox)
    return rect !== null && rectArea(rect) > 0
  })
}
